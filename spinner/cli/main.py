import importlib.resources
import os

import click
import exporter.exporter as exporter
from runner.utilities import run_benchmarks


def parse_extra_args(extra_args):
    args_dict = {}
    if extra_args:
        # Split by ';' first to handle key-value pairs
        pairs = extra_args.split(";")
        for pair in pairs:
            if "=" in pair:
                # Split by '=' to separate key and value
                # The '1' ensures only the first '=' is split
                key, value = pair.split("=", 1)
                args_dict[key.strip()] = value.strip()
    return args_dict


@click.command()
@click.option(
    "--config",
    "-c",
    default="bench_settings.json",
    type=str,
    help="Benchmark configuration file",
)
@click.option(
    "--run",
    "-r",
    default=False,
    is_flag=True,
    show_default=True,
    help="Run all benchmarks",
)
@click.option(
    "--export",
    "-e",
    default=False,
    is_flag=True,
    show_default=True,
    help="Export results to report.html",
)
@click.option(
    "--output", "-o", default="bench_metadata.pkl", type=str, help="Output File (.pkl)"
)
@click.option(
    "--extra-args",
    "-ea",
    default="",
    type=str,
    help="Extra arguments as key=value pairs, separated by semicolons.",
)
def cli(run, export, config, output, extra_args):
    extra_args_d = parse_extra_args(extra_args)
    main(run, export, config, output, extra_args_d)


def main(run, export, config, output, extra_args):
    if run:
        run_benchmarks(config, output, extra_args)

    if export:
        reporter_path = importlib.resources.files("exporter") / "reporter.ipynb"
        exporter.run_reporter(reporter_path, "report.html", pkl_db_path=os.getcwd())


if __name__ == "__main__":
    cli()
