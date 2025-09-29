import pendulum
from airflow.models.dag import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount

# This command now includes the export step
torch_and_hdfs_command = """
bash -c "
# --- Part 1: Training ---
torchrun \\
--nproc_per_node=3 \\
--nnodes=1 \\
--rdzv_backend=c10d \\
--rdzv_endpoint=localhost:29500 \\
/app/distributed_training.py \\
--dataset_root_dir /pyg_dataset \\
--num_epochs 10 \\
--batch_size 512 \\
--num_neighbors '10,5' \\
--progress_bar \\
--num_workers=0 \\
--model_save_path /app/my_models \\
&& \\
echo '--- Training finished, uploading model checkpoint to HDFS ---' && \\
hdfs dfs -mkdir -p /models/{{ ts_nodash }} && \\
hdfs dfs -put /app/my_models/* /models/{{ ts_nodash }} && \\
echo '--- Upload complete ---' && \\
echo '--- Exporting latest model for Triton Inference Server ---' && \\
python3 /app/export_for_triton.py \\
--model_name graphsage_model \\
--checkpoint_path /app/my_models/model_epoch_10.pt \\
--dataset_meta_path /pyg_dataset/META.json \\
--hdfs_repo_path /triton_models
"
"""

with DAG(
    dag_id="pytorch_distributed_training_and_serving_dag", # Renamed for clarity
    start_date=pendulum.datetime(2025, 1, 1, tz="UTC"),
    catchup=False,
    schedule=None,
    tags=["pytorch", "docker", "ml", "triton"],
    doc_md="""
    ### PyTorch Distributed Training and Serving DAG

    This DAG runs a distributed PyTorch training job, saves the final model
    checkpoint to HDFS, and then exports the model in a versioned format
    to an HDFS model repository for serving with Triton Inference Server.
    """,
) as dag:
    run_pytorch_training_and_export = DockerOperator(
        task_id="run_pytorch_training_and_export",
        image="pyg-node",
        docker_conn_id=None,
        command=torch_and_hdfs_command,
        auto_remove="force",
        network_mode="nebula-docker-compose_nebula-net", # Ensure this matches your network name
        environment={
            "JAVA_HOME": "/usr/lib/jvm/java-8-openjdk-amd64",
            "HADOOP_CONF_DIR": "/opt/hadoop-2.7.4/etc/hadoop",
            "HDFS_DATA_PATH": "/tmp/pyg_dataset",
            "LOCAL_DATA_PATH": "/pyg_dataset",
            "NUM_WORKERS": "3",
            "HDFS_COORD_PATH": "/job-coordination",
        },
        mounts=[
            Mount(
                source="/home/macquangdat2412/nebula-docker-compose/src",
                target="/app",
                type="bind"
            ),
            Mount(
                source="/home/macquangdat2412/nebula-docker-compose/hdfs-config/core-site.xml",
                target="/opt/hadoop-2.7.4/etc/hadoop/core-site.xml",
                type="bind"
            ),
            Mount(
                source="/home/macquangdat2412/nebula-docker-compose/hdfs-config/hdfs-site.xml",
                target="/opt/hadoop-2.7.4/etc/hadoop/hdfs-site.xml",
                type="bind"
            ),
        ],
        tty=True,
    )
