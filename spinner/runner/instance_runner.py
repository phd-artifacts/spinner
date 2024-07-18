import subprocess
import os
import signal
from rich import print as rprint
import pandas as pd
import time


class InstanceRunner:
    """
    Base class to handle the general case execution of benchmarks.
    """

    def __init__(
        self,
        execution_df,
        metadata,
        sweep_parameters,
        progress_callback,
        runner_env=None,
    ):
        self.execution_df = execution_df
        self.metadata = metadata
        self.sweep_parameters = sweep_parameters
        self.progress_callback = progress_callback
        self.runner_env = runner_env

    def get_bench_name(self) -> str:
        raise NotImplementedError

    def get_run_instruction(self) -> list:
        raise NotImplementedError

    def check_output(self, output) -> bool:
        # TODO check the output
        rprint("[red] Check output not implemented!")
        return True

    def run(self):
        """Runs the exercise and stores the results in self.results"""
        run_cmds = self.get_run_instruction()
        self.progress_callback(increment_total=(len(run_cmds) * self.metadata["runs"]))

        # Run serial commands first (so we can calculate speedup)
        for cmd in run_cmds:
            for i in range(self.metadata["runs"]):
                self.progress_callback(value=1)

                curr_df_entry = {
                    "name": self.get_bench_name(),
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
