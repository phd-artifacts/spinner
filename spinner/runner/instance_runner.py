import itertools
import os
import shutil
import signal
import subprocess
import time

import pandas as pd
from jinja2 import Environment, Template, Undefined
from rich import print as rprint


class StrictUndefined(Undefined):
    def __getattr__(self, name):
        raise NameError(f"'{name}' is undefined")

    def __getitem__(self, name):
        raise NameError(f"'{name}' is undefined")

    def __str__(self):
        raise NameError(f"'{self._undefined_name}' is undefined")


class InstanceRunner:
    """
    Base class to handle the general case execution of benchmarks.
    """

    def __init__(
        self,
        bench_name,
        execution_df,
        metadata,
        sweep_parameters,
        progress_callback,
        runner_env=None,
    ):
        self.bench_name = bench_name
        self.execution_df = execution_df
        self.metadata = metadata
        self.sweep_parameters = sweep_parameters
        self.progress_callback = progress_callback
        self.runner_env = runner_env

    def get_parameter_combinations(self):
        parameter_list = self.sweep_parameters.keys()
        parameter_combinations = list(
            itertools.product(
                *[self.sweep_parameters[param] for param in parameter_list]
            )
        )
        parameter_combinations = (
            dict(zip(parameter_list, comb)) for comb in parameter_combinations
        )

        return parameter_combinations

    def get_run_instruction(self) -> list:
        instructions = []

        parameter_combinations = self.get_parameter_combinations()

        for parameters in list(parameter_combinations):
            rprint(f"self.metadata: {self.metadata}")
            command_info = self.metadata[self.bench_name]["command"]
            template_string = command_info["template"]
            rprint(f"command_template: {template_string}")

            env = Environment(undefined=StrictUndefined)
            template = env.from_string(template_string)

            command = template.render(
                self.metadata | parameters | self.metadata[self.bench_name]
            )
            rprint(f"rendered command: {command}")

            instructions.append(
                {
                    "cmd": command,
                    "parameters": parameters,
                }
            )

        return instructions

    def check_output(self, output) -> bool:
        # TODO check the output
        rprint("[red] Check output not implemented!")
        return True

    def run(self):
        """Runs the exercise and stores the results in self.results"""
        run_cmds = self.get_run_instruction()
        self.progress_callback(increment_total=(len(run_cmds) * self.metadata["runs"]))

        for cmd in run_cmds:
            for i in range(self.metadata["runs"]):
                self.progress_callback(value=1)

                curr_df_entry = {
                    "name": self.bench_name,
                    **cmd["parameters"],
                    "time": None,
                }

                rprint(curr_df_entry)

                output, error, elapsed_time, return_code = self.run_command(
                    cmd, self.metadata["timeout"]
                )

                if not return_code == 0:
                    rprint(f"-> Error running command: {cmd['cmd']}\n{error}")
                    continue

                if not self.check_output(output):
                    rprint(f"-> Output incorrect: {output}")
                    continue

                # Append run to execution_df
                curr_df_entry["time"] = elapsed_time
                self.execution_df.loc[len(self.execution_df)] = curr_df_entry

    def run_command(self, cmd, timeout):
        """Runs a command and returns the output, time and return code"""
        rprint(f"-> Running command: {cmd['cmd']}")
        rprint(f"-> Path: {os.getcwd()}")

        start_time = time.time()
        try:
            process = subprocess.Popen(
                cmd["cmd"],
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                # cwd=cwd,
                env=self.runner_env,
            )
            stdout, stderr = process.communicate(timeout)
            returncode = process.returncode
            stderr = stderr.decode()
            stdout = stdout.decode()

        except subprocess.TimeoutExpired as e:
            rprint(f"-> Command timeout after {timeout} seconds")
            process.kill()
            process.communicate()
            returncode = -1  # Typically, -1 indicates a timeout error
            stderr = f"Timeout error (limit: {timeout}s)"
            stdout = ""
            rprint(f"-> Killing children group {process.pid}")
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            except ProcessLookupError:
                rprint(
                    f"-> Failed to kill children group {process.pid}, probably already dead"
                )
            rprint(f"-> Done killing children group {process.pid}")

        end_time = time.time()
        elapsed_time = end_time - start_time
        rprint(f"-> stderr = {stderr}")
        rprint(f"-> stdout = {stdout}")
        rprint(f"-> elapsed time = {elapsed_time}")
        rprint(f"-> return code = {returncode}")

        return (
            stdout,
            stderr,
            elapsed_time,
            returncode,
        )
