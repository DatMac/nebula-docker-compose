import os
import requests
import torch
import logging
import numpy as np # Import numpy for easier data handling
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from cassandra.cluster import Cluster
from nebula3.gclient.net import ConnectionPool
from nebula3.Config import Config
from nebula3.common.ttypes import Value
from prometheus_fastapi_instrumentator import Instrumentator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
NEBULA_HOST = os.environ.get("NEBULA_HOST", "graphd")
NEBULA_PORT = int(os.environ.get("NEBULA_PORT", 9669))
NEBULA_USER = os.environ.get("NEBULA_USER", "root")
NEBULA_PASSWORD = os.environ.get("NEBULA_PASSWORD", "nebula")
NEBULA_VERTEX_TAG = os.environ.get("NEBULA_VERTEX_TAG", "Customer")
NEBULA_LOOKUP_PROPERTY = os.environ.get("NEBULA_LOOKUP_PROPERTY", "cust_id")
CASSANDRA_HOSTS = os.environ.get("CASSANDRA_HOSTS", "cassandra").split(',')
CASSANDRA_KEYSPACE = os.environ.get("CASSANDRA_KEYSPACE", "feature_store")
CASSANDRA_TABLE = os.environ.get("CASSANDRA_TABLE", "customer_features")
TRITON_URL = os.environ.get("TRITON_URL", "http://triton-server:8000/v2/models/graphsage_model/infer")
NUM_HOPS = 2

app = FastAPI(title="Graph Inference Service")

try:
    print("--- ATTEMPTING TO INSTRUMENT APP ---")
    Instrumentator().instrument(app).expose(app)
    print("--- SUCCESSFULLY INSTRUMENTED APP AND EXPOSED /metrics ---")
except Exception as e:
    print(f"--- FAILED TO INSTRUMENT APP: {e} ---")

nebula_connection_pool = None
cassandra_session = None
cassandra_prepared_statement = None

_SVAL_TYPE = 5
_IVAL_TYPE = 3

def get_primitive_value(value: Value):
    value_type = value.getType()
    if value_type == _SVAL_TYPE:
        return value.get_sVal().decode('utf-8')
    elif value_type == _IVAL_TYPE:
        return value.get_iVal()
    raise TypeError(f"Unhandled Nebula value type: {value_type}")

@app.on_event("startup")
def startup_event():
    global nebula_connection_pool, cassandra_session, cassandra_prepared_statement
    try:
        logger.info(f"Initializing Nebula connection pool to {NEBULA_HOST}:{NEBULA_PORT}...")
        config = Config()
        config.max_connection_pool_size = 10
        nebula_connection_pool = ConnectionPool()
        if not nebula_connection_pool.init([(NEBULA_HOST, NEBULA_PORT)], config):
            raise ConnectionError("Failed to initialize Nebula connection pool.")
        logger.info("Nebula connection pool initialized.")
    except Exception as e:
        logger.error(f"FATAL: Failed to initialize Nebula connection pool: {e}")
        nebula_connection_pool = None
    try:
        logger.info(f"Connecting to Cassandra at {CASSANDRA_HOSTS}...")
        cluster = Cluster(CASSANDRA_HOSTS)
        cassandra_session = cluster.connect(CASSANDRA_KEYSPACE)
        query = f"SELECT cust_id, features FROM {CASSANDRA_TABLE} WHERE cust_id IN ?"
        cassandra_prepared_statement = cassandra_session.prepare(query)
        logger.info("Cassandra session connected and statement prepared.")
    except Exception as e:
        logger.error(f"FATAL: Failed to connect to Cassandra or prepare statement: {e}")
        cassandra_session = None

@app.on_event("shutdown")
def shutdown_event():
    if nebula_connection_pool:
        nebula_connection_pool.close()
    if cassandra_session:
        cassandra_session.shutdown()

class InferenceRequest(BaseModel):
    cust_id: str

# --- MODIFICATION 1: Update the function to return the target node's index ---
def prepare_tensors_from_db(start_cust_id: str):
    if not nebula_connection_pool or not cassandra_session or not cassandra_prepared_statement:
        raise HTTPException(status_code=503, detail="Database connections are not available.")

    with nebula_connection_pool.session_context(NEBULA_USER, NEBULA_PASSWORD) as session:
        session.execute('USE telecom;')

        lookup_query = f'LOOKUP ON {NEBULA_VERTEX_TAG} WHERE {NEBULA_VERTEX_TAG}.{NEBULA_LOOKUP_PROPERTY} == "{start_cust_id}" YIELD id(vertex)'
        result = session.execute(lookup_query)
        if not result.is_succeeded() or result.is_empty():
            raise HTTPException(status_code=404, detail=f"Customer ID '{start_cust_id}' not found in graph.")
        start_vertex_id = get_primitive_value(result.rows()[0].values[0])

        go_1_hop_query = f"GO 1 STEPS FROM '{start_vertex_id}' OVER * YIELD DISTINCT dst(edge)"
        result_1_hop = session.execute(go_1_hop_query)
        if not result_1_hop.is_succeeded():
            raise HTTPException(status_code=500, detail=f"Nebula 1-hop traversal failed.")
        level1_vertex_ids = {get_primitive_value(row.values[0]) for row in result_1_hop.rows()}

        level2_vertex_ids = set()
        if level1_vertex_ids:
            l1_vids_str = ", ".join([f'"{vid}"' for vid in level1_vertex_ids])
            # The GO query finds neighbors of L1 nodes, which are the L2 nodes
            go_2_hop_query = f"GO 1 STEPS FROM {l1_vids_str} OVER * YIELD DISTINCT dst(edge)"
            result_2_hop = session.execute(go_2_hop_query)
            if not result_2_hop.is_succeeded():
                 raise HTTPException(status_code=500, detail=f"Nebula 2-hop traversal failed.")
            # We must remove the start node and L1 nodes in case of cycles
            l2_candidates = {get_primitive_value(row.values[0]) for row in result_2_hop.rows()}
            level2_vertex_ids = l2_candidates - level1_vertex_ids - {start_vertex_id}

        subgraph_vertex_ids = {start_vertex_id}.union(level1_vertex_ids).union(level2_vertex_ids)
        sorted_vertex_ids = sorted(list(subgraph_vertex_ids))

        # --- Feature fetching logic remains the same ---
        fetch_vids_str = ", ".join([f'"{vid}"' for vid in sorted_vertex_ids])
        fetch_query = f"FETCH PROP ON {NEBULA_VERTEX_TAG} {fetch_vids_str} YIELD id(vertex), properties(vertex).{NEBULA_LOOKUP_PROPERTY}"
        result = session.execute(fetch_query)
        if not result.is_succeeded():
            raise HTTPException(status_code=500, detail=f"Nebula property fetch failed.")
        vertex_id_to_cust_id_map = {
            get_primitive_value(row.values[0]): get_primitive_value(row.values[1]) for row in result.rows()
        }
        subgraph_cust_ids = list(vertex_id_to_cust_id_map.values())

        if not subgraph_cust_ids:
            raise HTTPException(status_code=404, detail="Subgraph is empty or lacks feature keys.")

        rows = cassandra_session.execute(cassandra_prepared_statement, (subgraph_cust_ids,))
        cust_id_to_feature_map = {row.cust_id: row.features for row in rows}

        if len(cust_id_to_feature_map) != len(subgraph_cust_ids):
            missing = set(subgraph_cust_ids) - set(cust_id_to_feature_map.keys())
            raise HTTPException(status_code=404, detail=f"Feature data missing for CUST IDs: {missing}")

        vertex_id_to_idx_map = {vid: i for i, vid in enumerate(sorted_vertex_ids)}
        feature_list = [cust_id_to_feature_map[vertex_id_to_cust_id_map[vid]] for vid in sorted_vertex_ids]
        node_features = torch.tensor(feature_list, dtype=torch.float32)

        # --- OPTIMIZATION START: Replace the single inefficient edge query ---

        # Use a set to store edge tuples to automatically handle any duplicates
        edge_set = set()

        # Query 1: Get edges between the start node (L0) and its 1-hop neighbors (L1)
        if level1_vertex_ids:
            l1_vids_str = ", ".join([f'"{vid}"' for vid in level1_vertex_ids])
            edge_query_1 = (f"MATCH (v1)-[e]-(v2) WHERE id(v1) == '{start_vertex_id}' AND id(v2) IN [{l1_vids_str}] "
                            f"RETURN id(v1), id(v2)")
            result_edges_1 = session.execute(edge_query_1)
            if not result_edges_1.is_succeeded():
                raise HTTPException(status_code=500, detail="Nebula edge match failed for L0->L1.")
            for row in result_edges_1.rows():
                edge_set.add((get_primitive_value(row.values[0]), get_primitive_value(row.values[1])))

        # Query 2: Get edges between 1-hop neighbors (L1) and 2-hop neighbors (L2)
        if level1_vertex_ids and level2_vertex_ids:
            l1_vids_str = ", ".join([f'"{vid}"' for vid in level1_vertex_ids])
            l2_vids_str = ", ".join([f'"{vid}"' for vid in level2_vertex_ids])
            edge_query_2 = (f"MATCH (v1)-[e]-(v2) WHERE id(v1) IN [{l1_vids_str}] AND id(v2) IN [{l2_vids_str}] "
                            f"RETURN id(v1), id(v2)")
            result_edges_2 = session.execute(edge_query_2)
            if not result_edges_2.is_succeeded():
                raise HTTPException(status_code=500, detail="Nebula edge match failed for L1->L2.")
            for row in result_edges_2.rows():
                edge_set.add((get_primitive_value(row.values[0]), get_primitive_value(row.values[1])))

        # Convert the final set of required edges into the source and destination tensors
        source_nodes, dest_nodes = [], []
        for src_vid, dst_vid in edge_set:
            # Ensure both source and destination are part of the subgraph before adding
            if src_vid in vertex_id_to_idx_map and dst_vid in vertex_id_to_idx_map:
                source_nodes.append(vertex_id_to_idx_map[src_vid])
                dest_nodes.append(vertex_id_to_idx_map[dst_vid])

        edge_index = torch.tensor([source_nodes, dest_nodes], dtype=torch.int64)

        # --- OPTIMIZATION END ---

        # Find the index of our original start node in the final sorted list
        start_node_index = vertex_id_to_idx_map[start_vertex_id]

        return node_features, edge_index, start_node_index

# --- MODIFICATION 2: Update the /predict endpoint to handle the new return value ---
@app.post("/predict", summary="Get a model prediction for a given Customer ID")
async def predict(request: InferenceRequest):
    try:
        # Now returns three values
        node_features, edge_index, start_node_index = prepare_tensors_from_db(request.cust_id)
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An unexpected internal error occurred.")

    payload = {
        "inputs": [
            { "name": "x__0", "shape": list(node_features.shape), "datatype": "FP32", "data": node_features.flatten().tolist() },
            { "name": "edge_index__1", "shape": list(edge_index.shape), "datatype": "INT64", "data": edge_index.flatten().tolist() }
        ]
    }
    try:
        response = requests.post(TRITON_URL, json=payload, timeout=10)
        response.raise_for_status()
        triton_result = response.json()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=503, detail=f"Could not connect to Triton server: {e}")

    # --- NEW LOGIC: Extract the relevant data from Triton's response ---
    try:
        # Get the full output data and shape
        output = triton_result['outputs'][0]
        output_data = np.array(output['data'], dtype=np.float32)
        output_shape = output['shape']
        
        # Reshape the flat data list into a 2D matrix
        logits_matrix = output_data.reshape(output_shape)
        
        # Look up the specific row for our start node using the index
        start_node_logits = logits_matrix[start_node_index]
        
        # Return a clean JSON response with only the logits for the start node
        return {
            "cust_id": request.cust_id,
            "logits": start_node_logits.tolist(), # Convert numpy array back to a simple list
            "prediction": int(np.argmax(start_node_logits)) # Also return the final class prediction
        }
    except (KeyError, IndexError) as e:
        logger.error(f"Error parsing Triton response: {e}")
        raise HTTPException(status_code=500, detail="Could not parse response from model server.")
