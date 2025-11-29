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
    "app_name": "PyG_Hetero_Prep_v2",
    "partitions": 1,  # Ensure this matches your --nproc_per_node in torchrun
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
# 1. LOAD & ENRICH
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

    def safe_feat(col_name):
        return F.coalesce(F.col(col_name).cast("float"), F.lit(0.0))

    default_features = F.array(F.lit(0.0), F.lit(0.0), F.lit(0.0))

    # --- NODES ---
    # 1. Users
    users_raw = read_neb("user", False).select(F.col("_vertexId").alias("oid")).distinct()
    users_ids = users_raw.sort("oid").rdd.zipWithIndex()\
        .map(lambda x: (x[0][0], x[1])).toDF(["oid", "global_id"])

    u_feats = spark.read.format("org.apache.spark.sql.cassandra")\
        .options(table="user_features", keyspace=config["cassandra"]["keyspace"]).load()

    u_df = u_feats.select(
        F.col("user_id").alias("oid"),
        F.array(safe_feat("skill_level"), safe_feat("tenacity"), safe_feat("activity_velocity")).alias("features")
    )
    
    final_users = users_ids.join(u_df, "oid", "left")\
        .withColumn("features", F.coalesce(F.col("features"), default_features))\
        .select("oid", "global_id", "features")

    # 2. Quizzes
    quizzes_raw = read_neb("Quiz", False).select(F.col("_vertexId").alias("oid")).distinct()
    quizzes_ids = quizzes_raw.sort("oid").rdd.zipWithIndex()\
        .map(lambda x: (x[0][0], x[1])).toDF(["oid", "global_id"])

    q_feats = spark.read.format("org.apache.spark.sql.cassandra")\
        .options(table="quiz_features", keyspace=config["cassandra"]["keyspace"]).load()

    q_df = q_feats.select(
        F.col("quiz_id").alias("oid"),
        F.array(safe_feat("popularity_score"), safe_feat("actual_difficulty"), safe_feat("quality_rating")).alias("features")
    )

    final_quizzes = quizzes_ids.join(q_df, "oid", "left")\
        .withColumn("features", F.coalesce(F.col("features"), default_features))\
        .select("oid", "global_id", "features")

    # --- EDGES (Forward) ---
    def process_edge(label, src_node_df, dst_node_df, edge_type_str):
        raw = read_neb(label, True).select(F.col("_srcId").alias("src_oid"), F.col("_dstId").alias("dst_oid"))
        
        with_src = raw.join(src_node_df.select(F.col("oid").alias("src_oid"), F.col("global_id").alias("src")), "src_oid")
        with_dst = with_src.join(dst_node_df.select(F.col("oid").alias("dst_oid"), F.col("global_id").alias("dst")), "dst_oid")
        
        res = with_dst.select("src", "dst").distinct().rdd.zipWithIndex()\
            .map(lambda x: (x[0][0], x[0][1], x[1])).toDF(["src", "dst", "edge_id"])
        
        return res.withColumn("edge_type", F.lit(edge_type_str))

    edges_took = process_edge("TOOK", users_ids, quizzes_ids, "User__TOOK__Quiz")
    edges_rated = process_edge("RATED", users_ids, quizzes_ids, "User__RATED__Quiz")

    # --- EDGES (Reverse) [NEW STEP] ---
    # We must generate reverse edges for bidirectional message passing
    # Swap src/dst and create new edge_type string
    def create_reverse(df, rev_type_str):
        return df.select(
            F.col("dst").alias("src"), # Swap
            F.col("src").alias("dst"), # Swap
            F.col("edge_id") # Keep ID same to track interaction
        ).withColumn("edge_type", F.lit(rev_type_str))

    edges_took_rev = create_reverse(edges_took, "Quiz__rev_TOOK__User")
    edges_rated_rev = create_reverse(edges_rated, "Quiz__rev_RATED__User")

    return final_users, final_quizzes, edges_took, edges_rated, edges_took_rev, edges_rated_rev

# ------------------------------------------------------------------------
# 2. PARTITIONING (Unified LPA)
# ------------------------------------------------------------------------
def partition_hetero(users, quizzes, edges_took, edges_rated, config):
    logger.info("Partitioning Heterogeneous Graph via GraphFrames...")

    # Node IDs for LPA
    u_v = users.select(F.concat(F.lit("U_"), F.col("global_id")).alias("id"))
    q_v = quizzes.select(F.concat(F.lit("Q_"), F.col("global_id")).alias("id"))
    v = u_v.union(q_v)

    # Union Forward edges only for topology (LPA works fine on forward edges typically)
    e1 = edges_took.select(F.concat(F.lit("U_"), F.col("src")).alias("src"), 
                           F.concat(F.lit("Q_"), F.col("dst")).alias("dst"))
    e2 = edges_rated.select(F.concat(F.lit("U_"), F.col("src")).alias("src"), 
                            F.concat(F.lit("Q_"), F.col("dst")).alias("dst"))
    e = e1.union(e2)

    g = GraphFrame(v, e)
    res = g.labelPropagation(maxIter=1)

    part_map = res.withColumn("partition_id", (F.col("label") % config["partitions"]).cast("int"))
    
    def parse_pid(r):
        parts = r.id.split("_")
        ntype = "User" if parts[0] == "U" else "Quiz"
        gid = int(parts[1])
        return (ntype, gid, r.partition_id)

    mapped_rdd = part_map.select("id", "partition_id").rdd.map(parse_pid).cache()
    
    u_pmap = mapped_rdd.filter(lambda x: x[0] == "User").map(lambda x: (x[1], x[2])).toDF(["global_id", "partition_id"])
    q_pmap = mapped_rdd.filter(lambda x: x[0] == "Quiz").map(lambda x: (x[1], x[2])).toDF(["global_id", "partition_id"])

    p_users = users.join(u_pmap, "global_id")
    p_quizzes = quizzes.join(q_pmap, "global_id")

    return p_users, p_quizzes

# ------------------------------------------------------------------------
# 3. SAVE PARTITIONS (Hetero Compatible)
# ------------------------------------------------------------------------
def process_partition(iterator):
    """
    Constructs Hetero Data Dicts and saves to HDFS.
    Returns a list of status messages (required for mapPartitions).
    """
    hdfs_url = "http://namenode:50070" 
    hdfs_root = "/tmp/pyg_recsys_hetero"
    
    # Initialize buckets
    nodes_bucket = {"User": [], "Quiz": []}
    edges_bucket = {
        "User__TOOK__Quiz": [], 
        "User__RATED__Quiz": [],
        "Quiz__rev_TOOK__User": [],
        "Quiz__rev_RATED__User": []
    }

    partition_id = -1

    # 1. Consume Iterator
    for row in iterator:
        if partition_id == -1: 
            partition_id = row.partition_id
        
        if row.row_kind == 'node':
            nodes_bucket[row.type_key].append(row)
        elif row.row_kind == 'edge':
            edges_bucket[row.type_key].append(row)

    # 2. Handle Empty Partition
    # If partition was empty, we return an empty list immediately.
    # The 'ensure_partitions_exist' function in the driver will handle creating
    # the empty placeholder files later.
    if partition_id == -1: 
        return [] 

    # 3. Save Data (Only if we found data)
    client = InsecureClient(hdfs_url, user='root')
    p_dir = "{}/part_{}".format(hdfs_root, partition_id)
    
    # Ensure directory exists (in case HDFS is strict)
    if not client.status(p_dir, strict=False):
        client.makedirs(p_dir)

    # --- SAVE node_feats.pt ---
    node_feats_out = {}
    for ntype, rows in nodes_bucket.items():
        if not rows: continue
        rows.sort(key=lambda r: r.global_id)
        ids = torch.tensor([r.global_id for r in rows], dtype=torch.long)
        feats = torch.tensor([r.features for r in rows], dtype=torch.float)
        node_feats_out[ntype] = {'global_id': ids, 'feats': {'x': feats}}
    
    if node_feats_out:
        with tempfile.NamedTemporaryFile() as tmp:
            torch.save(node_feats_out, tmp.name)
            client.upload("{}/node_feats.pt".format(p_dir), tmp.name, overwrite=True)

    # --- SAVE graph.pt & edge_feats.pt ---
    graph_out = {}
    edge_feats_out = {} 

    for etype_str, rows in edges_bucket.items():
        if not rows: continue
        
        parts = etype_str.split("__")
        etype_tuple = (parts[0], parts[1], parts[2])
        
        srcs = torch.tensor([r.src for r in rows], dtype=torch.long)
        dsts = torch.tensor([r.dst for r in rows], dtype=torch.long)
        eids = torch.tensor([r.edge_id for r in rows], dtype=torch.long)
        
        # Calculate sizes based on the node buckets in this partition
        src_size = len(nodes_bucket[parts[0]]) 
        dst_size = len(nodes_bucket[parts[2]])
        
        graph_out[etype_tuple] = {
            'row': srcs, 'col': dsts, 'edge_id': eids,
            'size': (src_size, dst_size)
        }
        edge_feats_out[etype_tuple] = {'global_id': eids, 'feats': {}}

    if graph_out:
        with tempfile.NamedTemporaryFile() as tmp:
            torch.save(graph_out, tmp.name)
            client.upload("{}/graph.pt".format(p_dir), tmp.name, overwrite=True)
            
        with tempfile.NamedTemporaryFile() as tmp:
            torch.save(edge_feats_out, tmp.name)
            client.upload("{}/edge_feats.pt".format(p_dir), tmp.name, overwrite=True)

    # 4. Return Success Message (Iterable)
    return ["Partition {} Success".format(partition_id)]

def save_partitions_rdd(spark, users, quizzes, edges_took, edges_rated, edges_took_rev, edges_rated_rev, config):
    logger.info("Saving Heterogeneous Partitions (With Halo Nodes)...")
    
    u_part = users.select(F.col("global_id").alias("uid"), F.col("partition_id").alias("u_pid"))
    q_part = quizzes.select(F.col("global_id").alias("qid"), F.col("partition_id").alias("q_pid"))

    def prep_edges(df, etype_str, src_part, dst_part, src_col, dst_col):
        # Generic joiner for both forward and reverse
        joined = df.join(src_part, df.src == src_col).join(dst_part, df.dst == dst_col)
        
        # src_part has column 'u_pid' or 'q_pid', we need to select dynamic column names
        spid = "u_pid" if "u_pid" in src_part.columns else "q_pid"
        dpid = "u_pid" if "u_pid" in dst_part.columns else "q_pid"

        expanded = joined.select(
            F.explode(F.array(F.col(spid), F.col(dpid))).alias("partition_id"),
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

    # Prep Forward Edges (User -> Quiz)
    # Src: User (u_part, uid), Dst: Quiz (q_part, qid)
    df_took = prep_edges(edges_took, "User__TOOK__Quiz", u_part, q_part, u_part.uid, q_part.qid)
    df_rated = prep_edges(edges_rated, "User__RATED__Quiz", u_part, q_part, u_part.uid, q_part.qid)
    
    # Prep Reverse Edges (Quiz -> User)
    # Src: Quiz (q_part, qid), Dst: User (u_part, uid)
    df_took_rev = prep_edges(edges_took_rev, "Quiz__rev_TOOK__User", q_part, u_part, q_part.qid, u_part.uid)
    df_rated_rev = prep_edges(edges_rated_rev, "Quiz__rev_RATED__User", q_part, u_part, q_part.qid, u_part.uid)

    all_edges = df_took.unionByName(df_rated).unionByName(df_took_rev).unionByName(df_rated_rev)
    all_edges.cache()

    # Get Needed Nodes (Sources AND Destinations)
    def get_needed_nodes(edge_df, full_node_df, id_col, node_type):
        owned = full_node_df.select("partition_id", "global_id")
        referenced = edge_df.select("partition_id", F.col(id_col).alias("global_id"))
        needed_ids = owned.union(referenced).distinct()
        
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

    df_u = get_needed_nodes(all_edges, users, "src", "User") # Technically Users appear as src in Fwd and dst in Rev
    # But because we unioned ALL edges, "src" column contains Quizzes for reverse edges.
    # We need to be careful here. 
    
    # Simplified Logic:
    # A partition needs User X if:
    # 1. User X is owned by Partition P
    # 2. User X is connected to ANY edge in Partition P (either as src or dst)
    
    # Filter edges where User is src (Forward types) OR User is dst (Reverse types)
    # Actually, simpler: Union (src of forward) + (dst of forward) covers everyone involved.
    # The reverse edges just flip src/dst, they don't introduce new IDs.
    # So we can just use the FORWARD edge definitions to calculate halo nodes.
    
    forward_edges = df_took.unionByName(df_rated)
    
    df_u_needed = get_needed_nodes(forward_edges, users, "src", "User")
    df_q_needed = get_needed_nodes(forward_edges, quizzes, "dst", "Quiz")

    combined = df_u_needed.unionByName(df_q_needed).unionByName(all_edges)

    combined.repartition(config["partitions"], "partition_id")\
        .sortWithinPartitions("partition_id", "row_kind")\
        .rdd.mapPartitions(process_partition).collect()
        
    all_edges.unpersist()

# ------------------------------------------------------------------------
# 4. METADATA (Updated for Directory Structure)
# ------------------------------------------------------------------------
def save_metadata(users, quizzes, edges_took, edges_rated, config):
    logger.info("Generating Hetero Metadata...")
    client = InsecureClient(config["hdfs_uri"], user='root')
    
    n_u = users.count()
    n_q = quizzes.count()
    n_e1 = edges_took.count()
    n_e2 = edges_rated.count()
    # Total edges = Forward + Reverse (since we materialized them)
    total_edges = (n_e1 + n_e2) * 2 

    meta = {
        "is_hetero": True,
        "is_sorted": False,
        "num_parts": config["partitions"],
        "num_nodes": n_u + n_q,
        "num_edges": total_edges,
        "node_types": ["User", "Quiz"],
        "edge_types": [
            ["User", "TOOK", "Quiz"], ["User", "RATED", "Quiz"],
            ["Quiz", "rev_TOOK", "User"], ["Quiz", "rev_RATED", "User"]
        ],
        "node_feat_schema": {
            "User": {"x": [3]}, 
            "Quiz": {"x": [3]} 
        },
        "edge_feat_schema": {
            "User__TOOK__Quiz": {}, "User__RATED__Quiz": {},
            "Quiz__rev_TOOK__User": {}, "Quiz__rev_RATED__User": {}
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w') as tmp:
        json.dump(meta, tmp)
        tmp.flush()
        client.upload("{}/META.json".format(config['output_path']), tmp.name, overwrite=True)

    # --- Node Map Directory ---
    client.makedirs(f"{config['output_path']}/node_map")
    
    def save_map(df, name):
        pd = df.select("global_id", "partition_id").toPandas()
        size = n_u if name == "User" else n_q
        imap = torch.zeros(size, dtype=torch.long)
        imap[pd['global_id'].values] = torch.tensor(pd['partition_id'].values, dtype=torch.long)
        with tempfile.NamedTemporaryFile() as tmp:
            torch.save(imap, tmp.name)
            client.upload(f"{config['output_path']}/node_map/{name}.pt", tmp.name, overwrite=True)

    save_map(users, "User")
    save_map(quizzes, "Quiz")

    # --- Edge Map Directory ---
    client.makedirs(f"{config['output_path']}/edge_map")
    
    u_part = users.select(F.col("global_id").alias("uid"), F.col("partition_id").alias("pid"))

    def save_edge_map_fwd(df, fname):
        # Forward edges owned by User (src)
        pdf = df.join(u_part, df.src == u_part.uid).select("edge_id", "pid").toPandas()
        emap = torch.zeros(len(pdf), dtype=torch.long)
        emap[pdf['edge_id'].values] = torch.tensor(pdf['pid'].values, dtype=torch.long)
        with tempfile.NamedTemporaryFile() as tmp:
            torch.save(emap, tmp.name)
            client.upload(f"{config['output_path']}/edge_map/{fname}.pt", tmp.name, overwrite=True)
    
    # Reverse edges are technically the same interactions, usually mapped to Dst (User)
    # PyG expects edge_map to define which partition 'owns' the edge. 
    # For reverse edges (Quiz->User), we can map them to the User partition too, 
    # or the Quiz partition. Standard practice for reverse edges often mirrors forward.
    # Let's map them based on their Source (Quiz) partition to be consistent with standard logic.
    
    q_part = quizzes.select(F.col("global_id").alias("qid"), F.col("partition_id").alias("pid"))
    
    def save_edge_map_rev(df, fname):
        # Reverse edges (src is Quiz) -> Join with Quiz Partition
        pdf = df.join(q_part, df.src == q_part.qid).select("edge_id", "pid").toPandas()
        emap = torch.zeros(len(pdf), dtype=torch.long)
        emap[pdf['edge_id'].values] = torch.tensor(pdf['pid'].values, dtype=torch.long)
        with tempfile.NamedTemporaryFile() as tmp:
            torch.save(emap, tmp.name)
            client.upload(f"{config['output_path']}/edge_map/{fname}.pt", tmp.name, overwrite=True)

    save_edge_map_fwd(edges_took, "User__TOOK__Quiz")
    save_edge_map_fwd(edges_rated, "User__RATED__Quiz")
    # Note: We reuse the original edge DFs for IDs, but the mapping logic uses the Reverse Source logic
    # Actually, we passed 'edges_took_rev' into save_partitions, but here we can just reuse the ID mapping
    # since IDs are shared. But we need to map to Partition.
    # For safety, let's map reverse edges to the USER partition as well (same as forward).
    # Why? because DistLinkNeighborLoader for (Quiz->User) will likely look for edges on the Quiz partition.
    # Let's map to Source (Quiz).
    
    # We need to construct the reverse DFs again locally if not cached, or just assume IDs match.
    # Let's accept that 'edges_took' has the IDs.
    # But wait, create_reverse kept the same edge_id.
    save_edge_map_rev(edges_took.withColumnRenamed("dst", "src_quiz"), "Quiz__rev_TOOK__User")
    save_edge_map_rev(edges_rated.withColumnRenamed("dst", "src_quiz"), "Quiz__rev_RATED__User")

# ------------------------------------------------------------------------
# 5. POST-PROCESS CHECK (Handle Empty Partitions)
# ------------------------------------------------------------------------
def ensure_partitions_exist(config):
    logger.info("Verifying all partitions exist...")
    client = InsecureClient(config["hdfs_uri"], user='root')
    
    empty_node_feats = {
        "User": {'global_id': torch.tensor([], dtype=torch.long), 'feats': {'x': torch.tensor([], dtype=torch.float)}},
        "Quiz": {'global_id': torch.tensor([], dtype=torch.long), 'feats': {'x': torch.tensor([], dtype=torch.float)}}
    }
    empty_graph = {
        ("User", "TOOK", "Quiz"): {'row': torch.tensor([], dtype=torch.long), 'col': torch.tensor([], dtype=torch.long), 'edge_id': torch.tensor([], dtype=torch.long), 'size': (0, 0)},
        ("User", "RATED", "Quiz"): {'row': torch.tensor([], dtype=torch.long), 'col': torch.tensor([], dtype=torch.long), 'edge_id': torch.tensor([], dtype=torch.long), 'size': (0, 0)},
        ("Quiz", "rev_TOOK", "User"): {'row': torch.tensor([], dtype=torch.long), 'col': torch.tensor([], dtype=torch.long), 'edge_id': torch.tensor([], dtype=torch.long), 'size': (0, 0)},
        ("Quiz", "rev_RATED", "User"): {'row': torch.tensor([], dtype=torch.long), 'col': torch.tensor([], dtype=torch.long), 'edge_id': torch.tensor([], dtype=torch.long), 'size': (0, 0)}
    }
    empty_edge_feats = { k: {'global_id': torch.tensor([], dtype=torch.long), 'feats': {}} for k in empty_graph.keys() }

    for pid in range(config["partitions"]):
        p_dir = f"{config['output_path']}/part_{pid}"
        if not client.status(p_dir, strict=False):
            logger.warning(f"Partition {pid} missing. Creating placeholder.")
            client.makedirs(p_dir)
            
            def upload(data, name):
                with tempfile.NamedTemporaryFile() as tmp:
                    torch.save(data, tmp.name)
                    client.upload(f"{p_dir}/{name}", tmp.name, overwrite=True)
            
            upload(empty_node_feats, "node_feats.pt")
            upload(empty_graph, "graph.pt")
            upload(empty_edge_feats, "edge_feats.pt")

# ------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------
def main():
    spark = create_spark_session(CONFIG)
    spark.sparkContext.setCheckpointDir(CONFIG["checkpoint_dir"])
    
    try:
        # 1. Load, Enrich, Generate Reverse Edges
        users, quizzes, e_took, e_rated, e_took_rev, e_rated_rev = load_and_enrich(spark, CONFIG)
        
        # 2. Partition
        p_users, p_quizzes = partition_hetero(users, quizzes, e_took, e_rated, CONFIG)
        p_users.cache()
        p_quizzes.cache()
        
        # 3. Save Partitions (Including Reverse Edges)
        save_partitions_rdd(spark, p_users, p_quizzes, e_took, e_rated, e_took_rev, e_rated_rev, CONFIG)
        
        # 4. Save Global Maps (Directory Structure)
        save_metadata(p_users, p_quizzes, e_took, e_rated, CONFIG)
        
        # 5. Safety Check
        ensure_partitions_exist(CONFIG)
        
        logger.info("Heterogeneous Processing Complete.")
    except Exception as e:
        logger.error(str(e))
        traceback.print_exc()
    finally:
        spark.stop()

if __name__ == "__main__":
    main()
