# type: ignore
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


def retrieve(item: int, db: list[str] = None) -> tuple[int, str]:
    time.sleep(item / 10.0)
    return item, db[item]


def get_database():
    database: list[str] = [
        "Learning",
        "Ray",
        "Flexible",
        "Distributed",
        "Python",
        "for",
        "Machine",
        "Learning",
    ]
    return database


def print_runtime(input_data: list[Any], start_time: time.time):
    print(f"Runtime: {time.time() - start_time:.2f} seconds, \ndata:")
    print(*input_data, sep="\n")


@ray.remote
def retrieve_task(item: int, db: list[str] = None) -> tuple[int, str]:
    return retrieve(item, db)


def main() -> None:
    start = time.time()
    database = get_database()
    # Put the objects in the object store.
    database_ref = ray.put(database)

    object_references = [retrieve_task.remote(item, database_ref) for item in range(8)]
    data = ray.get(object_references)
    print_runtime(data, start)


def main_with_wait() -> None:
    start = time.time()
    database = get_database()
    # Put the objects in the object store.
    database_ref = ray.put(database)

    all_data: list[Any] = []

    object_references = [retrieve_task.remote(item, database_ref) for item in range(8)]

    while len(object_references) > 0:
        finished_refs, object_references = ray.wait(object_references)
        # Retrieve the results of the finished tasks.
        data = ray.get(finished_refs)
        all_data.extend(data)

    print_runtime(all_data, start)


if __name__ == "__main__":
    main()
    console.print()
    main_with_wait()
