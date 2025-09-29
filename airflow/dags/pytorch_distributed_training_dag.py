import pendulum
from airflow.models.dag import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount

torch_and_hdfs_command = """
bash -c "
torchrun \
--nproc_per_node=3 \
--nnodes=1 \
--rdzv_backend=c10d \
--rdzv_endpoint=localhost:29500 \
/app/distributed_training.py \
--dataset_root_dir /pyg_dataset \
--num_epochs 10 \
--batch_size 512 \
--num_neighbors '10,5' \
--progress_bar \
--num_workers=0 \
--model_save_path /app/my_models \
&& \
echo '--- Training finished, uploading models to HDFS ---' && \
hdfs dfs -mkdir -p /models/{{ ts_nodash }} && \
hdfs dfs -put /app/my_models/* /models/{{ ts_nodash }} && \
echo '--- Upload complete ---'
"
"""

with DAG(
    dag_id="pytorch_distributed_training_dag",
    start_date=pendulum.datetime(2025, 1, 1, tz="UTC"),
    catchup=False,
    schedule=None,
    tags=["pytorch", "docker", "ml"],
    doc_md="""
    ### PyTorch Distributed Training DAG

    This DAG uses the DockerOperator to run a distributed PyTorch training job.
    It relies on a pre-built Docker image named `pyg-node`. The operator is
    configured to connect to a specific Docker network and mount necessary
    volumes for scripts, data, and model outputs.

    **Important:** You must replace the placeholder paths in the `mounts`
    parameter with the actual paths on your Airflow worker host.
    """,
) as dag:
    run_pytorch_training = DockerOperator(
        task_id="run_pytorch_training_in_container",
        image="pyg-node",
        docker_conn_id=None,
        command=torch_and_hdfs_command,
        auto_remove="force",
        network_mode="nebula-docker-compose_nebula-net",
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
