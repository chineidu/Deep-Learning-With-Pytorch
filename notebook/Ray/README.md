# Ray

## Table of Contents

- [Ray](#ray)
  - [Table of Contents](#table-of-contents)
  - [GCP Setup](#gcp-setup)
    - [Create Ray Cluster](#create-ray-cluster)
    - [View Ray Dashboard](#view-ray-dashboard)
    - [Submit A Ray Job](#submit-a-ray-job)
    - [SSH Into The Cluster](#ssh-into-the-cluster)
    - [Shutdown The Cluster](#shutdown-the-cluster)

## GCP Setup

- The full config can be found here: [Github repo](https://github.com/ray-project/ray/tree/master/python/ray/autoscaler/gcp)
- Simple config

```yaml
# ===========================
# config.yaml
# ===========================
# A unique identifier for the head node and workers of this cluster.
cluster_name: test-cluster

# The maximum number of workers nodes to launch in addition to the head
# node.
max_workers: 2

auth:
  ssh_user: ubuntu

# If a node is idle for this many minutes, it will be removed.
idle_timeout_minutes: 5

# Cloud-provider specific configuration.
provider:
  project_id: ml-project-419115 # TODO: set your GCP project ID here
  type: gcp
  availability_zone: europe-west2-b
  region: europe-west2

available_node_types:
  ray_head_default:
    # The resources provided by this node type.
    resources: { "CPU": 2 }
    # Provider-specific config for this node type, e.g. instance type. By default
    # Ray will auto-configure unspecified fields such as subnets and ssh-keys.
    # For more documentation on available fields, see:
    # https://cloud.google.com/compute/docs/reference/rest/v1/instances/insert
    node_config:
      machineType: e2-medium
      disks:
        - boot: true
          autoDelete: true
          type: PERSISTENT
          initializeParams:
            diskSizeGb: 50
            # See https://cloud.google.com/compute/docs/images for more images
            sourceImage: projects/deeplearning-platform-release/global/images/family/common-cpu

setup_commands:
- sleep 5
- sudo apt-get update
- sudo apt-get install -y python3-pip python-is-python3
- pip install ray[default] numpy pandas rich

# Specify the node type of the head node (as configured above).
head_node_type: ray_head_default
```

### Create Ray Cluster

```sh
ray up -y <config-filename> --no-config-cache
# e.g.
ray up -y config.yaml --no-config-cache
```

- Output

```text
2024-04-16 22:11:35,959 - INFO - Note: NumExpr detected 12 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 8.
2024-04-16 22:11:35,959 - INFO - NumExpr defaulting to 8 threads.
Cluster: test-cluster

2024-04-16 22:11:36,464 INFO util.py:382 -- setting max workers for head node type to 0
Checking GCP environment settings
2024-04-16 22:11:44,259 INFO config.py:579 -- _configure_key_pair: Private key not specified in config, using/Users/neidu/.ssh/ray-autoscaler_gcp_europe-west2_ml-project-419115_ubuntu_0.pem
No head node found. Launching a new cluster. Confirm [y/N]: y [automatic, due to --yes]

Usage stats collection is enabled. To disable this, add `--disable-usage-stats` to the command that starts the cluster, or run the following command: `ray disable-usage-stats` before starting the cluster. See https://docs.ray.io/en/master/cluster/usage-stats.html for more details.
```

### View Ray Dashboard

```sh
ray dashboard <config-filename>
# e.g.
ray dashboard config.yaml
```

### Submit A Ray Job

- Run this Python file on the created `ray` cluster.

```py
# ===========================
# script.py
# ===========================
import time
from typing import Any
import ray
import numpy as np
import pandas as pd
from rich.console import Console
from rich.theme import Theme

custom_theme = Theme(
    {
        "info": "#76FF7B",
        "warning": "#FBDDFE",
        "error": "#FF0000",
    }
)
console = Console(theme=custom_theme)

# Initialize Ray and automatically connect to a running cluster.
ray.init(address="auto")


def load_model():
    # A dummy model.
    def model(batch: pd.DataFrame) -> pd.DataFrame:

        # Dummy payload so copying the model will actually copy some data
        # across nodes.
        model.payload = np.zeros(10)
        time.sleep(2)
        return pd.DataFrame({"default": batch["passenger_count"] % 2 == 0})

    return model


# Load data
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    np.random.seed(42)
    input_df_1: pd.DataFrame = pd.DataFrame(
        {"passenger_count": np.random.choice(a=np.arange(1, 10), size=50_000)}
    )
    input_df_2: pd.DataFrame = pd.DataFrame(
        {"passenger_count": np.random.choice(a=np.arange(2, 9), size=55_000)}
    )
    return input_df_1, input_df_2


@ray.remote(num_cpus=2, max_retries=2)
def make_prediction(model, data: pd.DataFrame):
    result = model(data)

    return result.shape


def batch_prediction():
    # ray.put() the model just once to local object store, and then pass the
    # reference to the remote tasks.
    model = load_model()
    model_ref = ray.put(model)

    # Load data
    input_df_1, input_df_2 = load_data()

    result_refs = []

    # Launch all prediction tasks.
    for data in [input_df_1, input_df_2]:
        # Launch a prediction task by passing model reference and the data to it.
        # NOTE: it would be highly inefficient if you are passing the model itself
        # like make_prediction.remote(model, data), which in order to pass the model
        # to remote node will ray.put(model) for each task, potentially overwhelming
        # the local object store and causing out-of-disk error.
        result_refs.append(make_prediction.remote(model_ref, data))

    # Wait for all the tasks to complete.
    results = ray.get(result_refs)

    # Let's check prediction output size.
    for r in results:
        console.print(f"Predictions: {r}", style="bold red")


if __name__ == "__main__":
    batch_prediction()

```

- Submit the job to the running cluster.

```sh
ray job submit --address http://localhost:<your_port> --working-dir . -- python <filename.py>
# e.g.
ray job submit --address http://localhost:8265 --working-dir . -- python script.py

# OR
ray submit <config-filename> <filename.py>
ray submit config.yaml script.py
```

### SSH Into The Cluster

```sh
ray attach <config-filename>
# e.g.
ray attach config.yaml
```

### Shutdown The Cluster

```sh
ray down -y <config-filename>

# e.g.
ray down -y config.yaml
```
