import os
import sys
import json
import logging
import tempfile
import traceback
import torch
import numpy as np
from hdfs import InsecureClient

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import *
from graphframes import GraphFrame

# --- CONFIGURATION ---
CONFIG = {
    "app_name": "PyG_Hetero_Prep",
    "partitions": 3,
    "node_limit": 500000,
    "spark_master": "spark://spark-master:7077",
    "hdfs_uri": "http://namenode:50070",
    "output_path": "/tmp/pyg_recsys_hetero",
    "checkpoint_dir": "/tmp/spark_checkpoints_hetero",
    "packages": [
        "com.vesoft.nebula-spark-connector:nebula-spark-connector_2.2:3.4.0",
        "com.datastax.spark:spark-cassandra-connector_2.11:2.4.3",
        "graphframes:graphframes:0.8.1-spark2.4-s_2.11",
        "com.twitter:jsr166e:1.1.0"
    ],
    "nebula": {
        "meta_address": "metad0:9559,metad1:9559,metad2:9559",
        "space": "quiz_recsys",
        "user": "root", "password": "nebula"
    },
    "cassandra": {
        "host": "cassandra", "port": "9042", "keyspace": "feature_store"
    }
}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_spark_session(config):
    return (SparkSession.builder
        .appName(config["app_name"])
        .master(config["spark_master"])
        .config("spark.jars.packages", ",".join(config["packages"]))
        .config("spark.cassandra.connection.host", config["cassandra"]["host"])
        .config("spark.cassandra.connection.port", config["cassandra"]["port"])
        .config(f"spark.sql.shuffle.partitions", 18)
        .getOrCreate())

# ------------------------------------------------------------------------
# 1. LOAD & ENRICH (HETEROGENEOUS) - FULLY FIXED
# ------------------------------------------------------------------------
def load_and_enrich(spark, config):
    logger.info("Loading Heterogeneous Data...")

    # Helper for Nebula
    def read_neb(label, is_edge):
        t = "edge" if is_edge else "vertex"
        return spark.read.format("com.vesoft.nebula.connector.NebulaDataSource")\
            .option("type", t).option("spaceName", config['nebula']['space'])\
            .option("label", label).option("returnCols", "")\
            .option("metaAddress", config['nebula']['meta_address'])\
            .option("partitionNumber", config["partitions"]).load()

    # Helper to sanitize feature columns (Handle NULLs inside the vector)
    def safe_feat(col_name):
        return F.coalesce(F.col(col_name).cast("float"), F.lit(0.0))

    # Define default feature vector for missing rows (User/Quiz not in Cassandra)
    default_features = F.array(F.lit(0.0), F.lit(0.0), F.lit(0.0))

    # --- PROCESS NODES SEPARATELY ---

    # 1. Users
    users_raw = read_neb("user", False).select(F.col("_vertexId").alias("oid")).distinct()
    users_ids = users_raw.sort("oid").rdd.zipWithIndex()\
        .map(lambda x: (x[0][0], x[1])).toDF(["oid", "global_id"])

    u_feats = spark.read.format("org.apache.spark.sql.cassandra")\
        .options(table="user_features", keyspace=config["cassandra"]["keyspace"]).load()

    # FIX 1: Sanitize individual columns BEFORE creating the array
    u_df = u_feats.select(
        F.col("user_id").alias("oid"),
        F.array(
            safe_feat("skill_level"), 
            safe_feat("tenacity"), 
            safe_feat("activity_velocity")
        ).alias("features")
    )
    
    # FIX 2: Handle missing rows (Left Join nulls) using the default array
    final_users = users_ids.join(u_df, "oid", "left")\
        .withColumn("features", F.coalesce(F.col("features"), default_features))\
        .select("oid", "global_id", "features")

    # 2. Quizzes
    quizzes_raw = read_neb("Quiz", False).select(F.col("_vertexId").alias("oid")).distinct()
    quizzes_ids = quizzes_raw.sort("oid").rdd.zipWithIndex()\
        .map(lambda x: (x[0][0], x[1])).toDF(["oid", "global_id"])

    q_feats = spark.read.format("org.apache.spark.sql.cassandra")\
        .options(table="quiz_features", keyspace=config["cassandra"]["keyspace"]).load()

    # FIX 1: Sanitize individual columns
    q_df = q_feats.select(
        F.col("quiz_id").alias("oid"),
        F.array(
            safe_feat("popularity_score"), 
            safe_feat("actual_difficulty"), 
            safe_feat("quality_rating")
        ).alias("features")
    )

    # FIX 2: Handle missing rows
    final_quizzes = quizzes_ids.join(q_df, "oid", "left")\
        .withColumn("features", F.coalesce(F.col("features"), default_features))\
        .select("oid", "global_id", "features")

    # --- PROCESS EDGES SEPARATELY ---
    
    def process_edge(label, src_node_df, dst_node_df, edge_type_str):
        raw = read_neb(label, True).select(F.col("_srcId").alias("src_oid"), F.col("_dstId").alias("dst_oid"))
        
        with_src = raw.join(src_node_df.select(F.col("oid").alias("src_oid"), F.col("global_id").alias("src")), "src_oid")
        with_dst = with_src.join(dst_node_df.select(F.col("oid").alias("dst_oid"), F.col("global_id").alias("dst")), "dst_oid")
        
        res = with_dst.select("src", "dst").distinct().rdd.zipWithIndex()\
            .map(lambda x: (x[0][0], x[0][1], x[1])).toDF(["src", "dst", "edge_id"])
        
        return res.withColumn("edge_type", F.lit(edge_type_str))

    edges_took = process_edge("TOOK", users_ids, quizzes_ids, "User__TOOK__Quiz")
    edges_rated = process_edge("RATED", users_ids, quizzes_ids, "User__RATED__Quiz")

    return final_users, final_quizzes, edges_took, edges_rated

# ------------------------------------------------------------------------
# 2. PARTITIONING (Unified LPA)
# ------------------------------------------------------------------------
def partition_hetero(users, quizzes, edges_took, edges_rated, config):
    logger.info("Partitioning Heterogeneous Graph via GraphFrames...")

    # GraphFrames requires a single 'id' column. We create composite string IDs.
    # User 1 -> "U_1", Quiz 5 -> "Q_5"
    
    u_v = users.select(F.concat(F.lit("U_"), F.col("global_id")).alias("id"))
    q_v = quizzes.select(F.concat(F.lit("Q_"), F.col("global_id")).alias("id"))
    v = u_v.union(q_v)

    # Union edges for topology only
    e1 = edges_took.select(F.concat(F.lit("U_"), F.col("src")).alias("src"), 
                           F.concat(F.lit("Q_"), F.col("dst")).alias("dst"))
    e2 = edges_rated.select(F.concat(F.lit("U_"), F.col("src")).alias("src"), 
                            F.concat(F.lit("Q_"), F.col("dst")).alias("dst"))
    e = e1.union(e2)

    # Run LPA
    g = GraphFrame(v, e)
    res = g.labelPropagation(maxIter=1)

    # Map partition ID back to original DFs
    # "U_1" -> global_id: 1, partition: X
    part_map = res.withColumn("partition_id", (F.col("label") % config["partitions"]).cast("int"))
    
    # Split back to User/Quiz maps
    # We use RDD string parsing to be safe with older Spark versions
    def parse_pid(r):
        # r.id is "U_1" or "Q_5"
        parts = r.id.split("_")
        ntype = "User" if parts[0] == "U" else "Quiz"
        gid = int(parts[1])
        return (ntype, gid, r.partition_id)

    mapped_rdd = part_map.select("id", "partition_id").rdd.map(parse_pid).cache()
    
    u_pmap = mapped_rdd.filter(lambda x: x[0] == "User").map(lambda x: (x[1], x[2])).toDF(["global_id", "partition_id"])
    q_pmap = mapped_rdd.filter(lambda x: x[0] == "Quiz").map(lambda x: (x[1], x[2])).toDF(["global_id", "partition_id"])

    # Attach partition info
    p_users = users.join(u_pmap, "global_id")
    p_quizzes = quizzes.join(q_pmap, "global_id")

    return p_users, p_quizzes

# ------------------------------------------------------------------------
# 3. SAVE PARTITIONS (Hetero Compatible)
# ------------------------------------------------------------------------
def process_partition(iterator):
    """
    Constructs Hetero Data Dicts:
    node_feats = { "User": {ids, x}, "Quiz": {ids, x} }
    graph = { ("User", "TOOK", "Quiz"): {row, col, eid}, ... }
    """
    hdfs_url = "http://namenode:50070" 
    hdfs_root = "/tmp/pyg_recsys_hetero"
    
    # Storage buckets
    # key: node_type, value: list of rows
    nodes_bucket = {"User": [], "Quiz": []}
    # key: edge_type_str, value: list of rows
    edges_bucket = {"User__TOOK__Quiz": [], "User__RATED__Quiz": []}

    partition_id = -1

    for row in iterator:
        if partition_id == -1: partition_id = row.partition_id
        
        if row.row_kind == 'node':
            nodes_bucket[row.type_key].append(row)
        elif row.row_kind == 'edge':
            edges_bucket[row.type_key].append(row)

    if partition_id == -1: return

    client = InsecureClient(hdfs_url, user='root')
    p_dir = "{}/part_{}".format(hdfs_root, partition_id)

    # --- SAVE node_feats.pt ---
    # Structure: Dictionary[NodeType, Dict[str, Tensor]]
    node_feats_out = {}
    
    for ntype, rows in nodes_bucket.items():
        if not rows: continue
        # Sort by global ID
        rows.sort(key=lambda r: r.global_id)
        
        ids = torch.tensor([r.global_id for r in rows], dtype=torch.long)
        feats = torch.tensor([r.features for r in rows], dtype=torch.float)
        
        node_feats_out[ntype] = {
            'global_id': ids,
            'feats': {'x': feats}
        }
    
    if node_feats_out:
        with tempfile.NamedTemporaryFile() as tmp:
            torch.save(node_feats_out, tmp.name)
            client.upload("{}/node_feats.pt".format(p_dir), tmp.name, overwrite=True)

    # --- SAVE graph.pt ---
    # Structure: Dictionary[EdgeTypeTuple, Dict[str, Tensor]]
    graph_out = {}
    edge_feats_out = {} # Skeleton for IDs

    for etype_str, rows in edges_bucket.items():
        if not rows: continue
        
        # Parse "User__TOOK__Quiz" -> ("User", "TOOK", "Quiz")
        parts = etype_str.split("__")
        etype_tuple = (parts[0], parts[1], parts[2])
        
        srcs = torch.tensor([r.src for r in rows], dtype=torch.long)
        dsts = torch.tensor([r.dst for r in rows], dtype=torch.long)
        eids = torch.tensor([r.edge_id for r in rows], dtype=torch.long)
        
        # LocalGraphStore expects this dict structure
        graph_out[etype_tuple] = {
            'row': srcs,
            'col': dsts,
            'edge_id': eids,
            'size': (len(nodes_bucket[parts[0]]), len(nodes_bucket[parts[2]]))
        }
        
        # LocalFeatureStore expects global_id in edge_feats even if no features
        edge_feats_out[etype_tuple] = {
            'global_id': eids,
            'feats': {}
        }

    if graph_out:
        with tempfile.NamedTemporaryFile() as tmp:
            torch.save(graph_out, tmp.name)
            client.upload("{}/graph.pt".format(p_dir), tmp.name, overwrite=True)
            
        with tempfile.NamedTemporaryFile() as tmp:
            torch.save(edge_feats_out, tmp.name)
            client.upload("{}/edge_feats.pt".format(p_dir), tmp.name, overwrite=True)
            
    yield "Partition {} Success".format(partition_id)

def save_partitions_rdd(spark, users, quizzes, edges_took, edges_rated, config):
    logger.info("Saving Heterogeneous Partitions (With Halo Nodes)...")
    
    # 1. Get Partition Mappings (Who owns what?)
    #    We need these to know where edges "live" originally
    u_part = users.select(F.col("global_id").alias("uid"), F.col("partition_id").alias("u_pid"))
    q_part = quizzes.select(F.col("global_id").alias("qid"), F.col("partition_id").alias("q_pid"))

    # ---------------------------------------------------------
    # 2. Prepare Edges (Explode logic)
    #    If User is in P1 and Quiz is in P2, the edge goes to 
    #    BOTH P1 and P2 so both partitions can see the connection.
    # ---------------------------------------------------------
    def prep_edges(df, etype_str):
        # Join src to User Owner, dst to Quiz Owner
        joined = df.join(u_part, df.src == u_part.uid).join(q_part, df.dst == q_part.qid)
        
        # LOGIC CHANGE: 
        # Instead of filtering u_pid == q_pid, we create an array of [u_pid, q_pid]
        # and explode it. This duplicates the edge row for both partitions.
        # distinct() removes duplicates if u_pid == q_pid.
        expanded = joined.select(
            F.explode(F.array(F.col("u_pid"), F.col("q_pid"))).alias("partition_id"),
            "src", "dst", "edge_id"
        ).distinct()

        return expanded.select(
            "partition_id", 
            F.lit("edge").alias("row_kind"), 
            F.lit(etype_str).alias("type_key"),
            F.lit(None).cast("long").alias("global_id"), 
            F.lit(None).cast("array<float>").alias("features"),
            "src", "dst", "edge_id"
        )

    df_took = prep_edges(edges_took, "User__TOOK__Quiz")
    df_rated = prep_edges(edges_rated, "User__RATED__Quiz")
    
    # Cache edges as we need them to calculate required nodes
    all_edges = df_took.unionByName(df_rated)
    all_edges.cache()

    # ---------------------------------------------------------
    # 3. Prepare Nodes (Owned + Halo logic)
    #    A partition needs a node if:
    #    a) It owns the node (LPA result)
    #    b) An edge in this partition connects to that node (Halo)
    # ---------------------------------------------------------
    
    def get_needed_nodes(edge_df, full_node_df, id_col, node_type):
        # 1. Nodes strictly owned by this partition
        owned = full_node_df.select("partition_id", "global_id")
        
        # 2. Nodes referenced by edges existing in this partition (Halo)
        #    (If partition_id=0 has an edge src=10, we need Node 10 in part 0)
        referenced = edge_df.select("partition_id", F.col(id_col).alias("global_id"))
        
        # 3. Union and Distinct to get full list of needed IDs per partition
        needed_ids = owned.union(referenced).distinct()
        
        # 4. Join back to get features
        #    full_node_df contains [global_id, features, (partition_id - ignore this one)]
        #    We ignore the partition_id in full_node_df because we want the one from 'needed_ids'
        return needed_ids.join(full_node_df.drop("partition_id"), "global_id")\
            .select(
                "partition_id", 
                F.lit("node").alias("row_kind"), 
                F.lit(node_type).alias("type_key"), 
                "global_id", "features", 
                F.lit(None).cast("long").alias("src"), 
                F.lit(None).cast("long").alias("dst"), 
                F.lit(None).cast("long").alias("edge_id")
            )

    # Calculate needed Users (Sources in edges)
    # Note: users/quizzes DFs passed in already have features joined
    df_u = get_needed_nodes(all_edges, users, "src", "User")
    
    # Calculate needed Quizzes (Destinations in edges)
    df_q = get_needed_nodes(all_edges, quizzes, "dst", "Quiz")

    # 4. Union All
    combined = df_u.unionByName(df_q).unionByName(all_edges)

    # 5. Execute RDD
    #    Sort within partition by row_kind to ensure nodes processed before edges (optional but good practice)
    combined.repartition(config["partitions"], "partition_id")\
        .sortWithinPartitions("partition_id", "row_kind")\
        .rdd.mapPartitions(process_partition).collect()
        
    all_edges.unpersist()

# ------------------------------------------------------------------------
# 4. METADATA (Driver)
# ------------------------------------------------------------------------
def save_metadata(users, quizzes, edges_took, edges_rated, config):
    logger.info("Generating Hetero Metadata...")
    client = InsecureClient(config["hdfs_uri"], user='root')
    
    # 1. Collect Counts
    n_u = users.count()
    n_q = quizzes.count()
    n_e1 = edges_took.count()
    n_e2 = edges_rated.count()

    # 2. META.json
    meta = {
        "is_hetero": True,
        "is_sorted": False,
        "num_parts": config["partitions"],
        "num_nodes": n_u + n_q, # Total (informational)
        "num_edges": n_e1 + n_e2,
        "node_types": ["User", "Quiz"],
        "edge_types": [["User", "TOOK", "Quiz"], ["User", "RATED", "Quiz"]],
        "node_feat_schema": {
            "User": {"x": [3]}, # User dim
            "Quiz": {"x": [3]}  # Quiz dim
        },
        "edge_feat_schema": {
            ("User", "TOOK", "Quiz"): {},
            ("User", "RATED", "Quiz"): {}
        }
    }
    
    # Fix Tuple keys for JSON dump (JSON keys must be strings)
    # PyG loaders usually reconstruct tuples from specific formats or handle string keys.
    # We will stringify the tuple keys for standard JSON, PyG loader logic often adapts or we can adjust loader.
    # However, strictly speaking, standard META.json usually uses lists or simple dicts. 
    # For custom LocalFeatureStore, the critical part is 'is_hetero': True.
    # We'll transform tuple keys to strings for valid JSON.
    meta_json_safe = meta.copy()
    meta_json_safe['edge_feat_schema'] = {str(k): v for k, v in meta['edge_feat_schema'].items()}

    with tempfile.NamedTemporaryFile(mode='w') as tmp:
        json.dump(meta_json_safe, tmp)
        tmp.flush()
        client.upload("{}/META.json".format(config['output_path']), tmp.name, overwrite=True)

    # 3. node_map.pt (Dict[NodeType, Tensor])
    # Collect partition maps
    u_pd = users.select("global_id", "partition_id").toPandas()
    q_pd = quizzes.select("global_id", "partition_id").toPandas()
    
    u_map = torch.zeros(n_u, dtype=torch.long)
    u_map[u_pd['global_id'].values] = torch.tensor(u_pd['partition_id'].values, dtype=torch.long)
    
    q_map = torch.zeros(n_q, dtype=torch.long)
    q_map[q_pd['global_id'].values] = torch.tensor(q_pd['partition_id'].values, dtype=torch.long)
    
    node_map = { "User": u_map, "Quiz": q_map }
    
    with tempfile.NamedTemporaryFile() as tmp:
        torch.save(node_map, tmp.name)
        client.upload("{}/node_map.pt".format(config['output_path']), tmp.name, overwrite=True)

    # 4. edge_map.pt (Dict[EdgeType, Tensor])
    # We map edge -> source partition (User partition)
    # Join Edges with Users to get partition
    u_part = users.select(F.col("global_id").alias("src"), F.col("partition_id").alias("pid"))
    
    def make_emap(df):
        pdf = df.join(u_part, "src").select("edge_id", "pid").toPandas()
        emap = torch.zeros(len(pdf), dtype=torch.long)
        emap[pdf['edge_id'].values] = torch.tensor(pdf['pid'].values, dtype=torch.long)
        return emap

    edge_map = {
        ("User", "TOOK", "Quiz"): make_emap(edges_took),
        ("User", "RATED", "Quiz"): make_emap(edges_rated)
    }

    with tempfile.NamedTemporaryFile() as tmp:
        torch.save(edge_map, tmp.name)
        client.upload("{}/edge_map.pt".format(config['output_path']), tmp.name, overwrite=True)

# ------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------
def main():
    spark = create_spark_session(CONFIG)
    spark.sparkContext.setCheckpointDir(CONFIG["checkpoint_dir"])
    
    try:
        # 1. Load & Enrich
        users, quizzes, e_took, e_rated = load_and_enrich(spark, CONFIG)
        
        # 2. Partition (LPA on unified topology)
        p_users, p_quizzes = partition_hetero(users, quizzes, e_took, e_rated, CONFIG)
        p_users.cache()
        p_quizzes.cache()
        e_took.cache()
        e_rated.cache()

        # 3. Save Partitions (RDD)
        save_partitions_rdd(spark, p_users, p_quizzes, e_took, e_rated, CONFIG)
        
        # 4. Save Global Maps
        save_metadata(p_users, p_quizzes, e_took, e_rated, CONFIG)
        
        logger.info("Heterogeneous Processing Complete.")
    except Exception as e:
        logger.error(str(e))
        traceback.print_exc()
    finally:
        spark.stop()

if __name__ == "__main__":
    main()
