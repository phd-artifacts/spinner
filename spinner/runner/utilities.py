import itertools
import os
import pickle
from functools import partial

import pandas as pd
import yaml
from rich import print as rprint
from rich.progress import Progress
from runner.instance_runner import InstanceRunner


def run_benchmarks(config, output, extra_args):
    """
    Generate execution matrix from input configuration and run all benchmarks.
    """
    bench_config = yaml.safe_load(open(config))
    bench_metadata = bench_config["metadata"]
    bench_metadata["start_timestamp"] = str(pd.Timestamp.now())
    bench_metadata.update(extra_args)
    bench_metadata["runner_hostname"] = str(os.uname()[1])
    bench_metadata["start_env"] = str(os.environ.copy())

    # Iterate over settings in bench_config, excluding metadata
    bench_names = [
        bench_name for bench_name in bench_config if bench_name != "metadata"
    ]

    # Get all parameters from each bench and create a list of columns
    parameters = list(
        itertools.chain(*[list(bench_config[bench].keys()) for bench in bench_names])
    )

    # List parameters generated by output capture, if any
    capture_parameters = []
    for bench_name in bench_names:
        if "output" in bench_config["metadata"][bench_name]:
            capture_parameters.append("output")
            rprint(f"-> Capturing output from execution")

            for filter in bench_config["metadata"][bench_name]["output"]:
                if "to_float" in filter:
                    rprint(f"-> Capturing parameter {filter['to_float']['name']}")
                    capture_parameters.append(filter["to_float"]["name"])

    columns = list(itertools.chain(["name"], parameters, capture_parameters, ["time"]))

    execution_df = pd.DataFrame(columns=columns)

    with Progress() as progress:
        # Helper function to update progress
        def _update_progress(
            progress, task, value=None, total=None, increment_total=None
        ):
            if total is not None:
                progress.update(task, total=total)
            if value is not None:
                progress.advance(task, advance=value)
            if increment_total is not None:
                progress.update(
                    task, total=progress.tasks[task].total + increment_total
                )

        # Main progress bar
        bench_task = progress.add_task(
            "[cyan]Running benchmarks",
        )

        # Handle to progress bar updates
        progress_callback = partial(
            _update_progress,
            progress,
            bench_task,
            value=None,
            total=None,
            increment_total=None,
        )

        for bench_name in bench_names:
            runner_class = InstanceRunner

            instance_runner = runner_class(
                bench_name,
                execution_df,
                bench_config["metadata"],
                bench_config[bench_name],
                progress_callback,
            )

            instance_runner.run()

            progress_callback()

    bench_metadata["end_timestamp"] = str(pd.Timestamp.now())
    bench_metadata["end_env"] = str(os.environ.copy())

    rprint(execution_df)
    with open(output, "wb") as f:
        pickle.dump({"metadata": bench_metadata, "dataframe": execution_df}, f)
