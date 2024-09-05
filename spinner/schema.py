from __future__ import annotations

import ast
import itertools as it
from functools import cache, cached_property
from typing import Annotated, Any, Literal, Self

from jinja2 import Environment, Template, meta
from pydantic import (
    BaseModel,
    Field,
    PositiveFloat,
    PositiveInt,
    RootModel,
    field_validator,
    model_validator,
)
from pydantic_core import ErrorDetails

# ==============================================================================
# GLOBALS
# ==============================================================================

# Used to aggregating errors. The first element of the outer tuple os a tuple
# representing the location of the error in the config (e.g.
# `applications.name.output.0`) and the second is the error message to be displayed to
# the user.
_LocationMessagePair = tuple[tuple[int | str, ...], str]

# ==============================================================================
# MODELS
# ==============================================================================


class SpinnerMetadata(BaseModel):
    """The metadata section of the config."""

    description: str
    version: str = Field(pattern=r"^v?\d+\.\d+(\.\d+)?$")
    runs: int = Field(gt=0)
    timeout: PositiveFloat | None = Field(default=None, gt=0.0)
    retry: bool = Field(default=False)
    retry_limit: PositiveInt = Field(default=1, ge=0)


class SpinnerCommand(RootModel):
    """A command template to execute an application."""

    root: str

    def __hash__(self) -> int:
        return hash(self.template)

    @cache
    def template(self, env: Environment | None = None) -> Template:
        if not env:
            env = Environment()
        return env.from_string(self.root)

    @cache
    def render(self, *args, **kwargs) -> str:
        return self.template().render(*args, **kwargs)

    @cached_property
    def placeholders(self) -> set[str]:
        return meta.find_undeclared_variables(self.template())

    def __str__(self) -> str:
        return str(self.root)

    def __repr__(self) -> str:
        return repr(self.root)


class SpinnerLambda(RootModel):
    """A Python lambda function to parse the command output."""

    root: str

    @field_validator("root", mode="after")
    def validate_lambda_python_code(cls, root: str) -> str:
        root = root.strip()
        try:
            module = ast.parse(source=root)
            assert len(module.body) == 1, "only one expression is allowed"
            body = module.body
            assert isinstance(
                body[0].value, ast.Lambda
            ), "must be a lambda expression"
            value = body[0].value
            assert len(value.args.args) == 1, "lambda must receive a single argument"
        except SyntaxError as e:
            raise ValueError(f"syntax error: {e}") from e
        except AssertionError as e:
            raise ValueError(f"{e}") from e
        return root

    def __call__(self, *args, **kwargs) -> Any:
        return eval(self.root)(*args, **kwargs)


class SpinnerOutputAll(BaseModel):
    type: Literal["all"]
    name: str
    func: SpinnerLambda = Field(alias="lambda")


class SpinnerOutputMatches(BaseModel):
    type: Literal["matches"]
    name: str
    pattern: str
    func: SpinnerLambda = Field(alias="lambda")


SpinnerOutput = Annotated[
    SpinnerOutputAll | SpinnerOutputMatches, Field(discriminator="type")
]


class SpinnerPlot(BaseModel):
    title: str = Field(default="")
    x_axis: str
    y_axis: str
    group_by: str | list[str] | None = Field(default=None)


class SpinnerApplication(BaseModel):
    command: SpinnerCommand
    outputs: list[SpinnerOutput] = Field(default_factory=list)
    plots: list[SpinnerPlot] = Field(default_factory=list)

    def _validate_plot(self, plot: SpinnerPlot) -> tuple[tuple[Any], str]:
        errors = []
        variables = self.variables
        if plot.x_axis not in variables:
            errors.append((("plot", "x_axis"), f"undefined x-axis {plot.x_axis!r}"))
        if plot.y_axis not in variables:
            errors.append((("plot", "y_axis"), f"undefined y-axis {plot.y_axis!r}"))
        if plot.group_by and plot.group_by not in variables:
            errors.append(
                (("plot", "group_by"), f"undefined group-by {plot.group_by!r}")
            )
        return errors

    @model_validator(mode="after")
    def plot_axes_are_valid(self) -> Self:
        errors = []
        for plot in self.plots:
            errors += self._validate_plot(plot)
        if errors:
            raise ValidationError.from_exception_data(
                title="SpinnerApplication",
                line_errors=[
                    ErrorDetails(
                        type="value_error", loc=loc, ctx={"error": ValueError(msg)}
                    )
                    for loc, msg in errors
                ],
            )
        return self

    @cached_property
    def output_variables(self) -> set[str]:
        return set(x.name for x in self.outputs)

    @cached_property
    def variables(self) -> set[str]:
        return self.command.placeholders | self.output_variables


class SpinnerApplications(RootModel):
    root: dict[str, SpinnerApplication] = Field(default_factory=dict)

    def items(self):
        return self.root.items()

    def __iter__(self):
        return iter(self.root)

    def __getitem__(self, key) -> SpinnerApplication | None:
        return self.root.get(key)


class SpinnerBenchmark(RootModel):
    root: dict[str, list[Any]] = Field(default_factory=dict)

    @cached_property
    def parameters(self) -> set[str]:
        return set(self.root.keys())

    def sweep(self) -> list[dict[str, Any]]:
        keys = self.root.keys()
        values = it.product(*[values for values in self.root.values()])
        return list(dict(zip(keys, x)) for x in values)

    def items(self):
        return self.root.items()

    def __iter__(self):
        return iter(self.root)

    def __getitem__(self, key) -> list[Any] | None:
        return self.root.get(key)


class SpinnerBenchmarks(RootModel):
    root: dict[str, SpinnerBenchmark] = Field(default_factory=dict)

    def items(self):
        return self.root.items()

    def __iter__(self):
        return iter(self.root)

    def __getitem__(self, key) -> SpinnerApplication | None:
        return self.root.get(key)


class SpinnerConfig(BaseModel):
    metadata: SpinnerMetadata
    applications: SpinnerApplications = Field(default_factory=dict)
    benchmarks: SpinnerBenchmarks = Field(default_factory=dict)

    def validate_benchmark_keys(self) -> _LocationMessagePair:
        errors = []

        benchmarks = set(self.benchmarks)
        applications = set(self.applications)

        if difference := benchmarks - applications:
            for benchmark in difference:
                errors.append(
                    (("benchmarks", benchmark), f"benchmark {benchmark!r} is undefined")
                )

        return errors

    def validate_application_placeholders(self) -> _LocationMessagePair:
        errors = []

        for name, application in self.applications.items():
            placeholders = application.command.placeholders
            if name not in self.benchmarks:
                # TODO: Issue a warning when the application has no corresponding
                # benchmark.
                continue

            # Which placeholders that are *not* in the benchmark parameters
            if difference := placeholders - self.benchmarks[name].parameters:
                errors.append(
                    (
                        ("applications", name, "command"),
                        f"placeholders {difference} are undefined",
                    )
                )

        return errors

    @model_validator(mode="after")
    def validate(self) -> Self:
        errors = []
        errors += self.validate_benchmark_keys()
        errors += self.validate_application_placeholders()

        if errors:
            # Create a `ValidationError` that aggregates errors from all validators
            raise ValidationError.from_exception_data(
                title="Spinner Config",
                line_errors=[
                    ErrorDetails(
                        type="value_error", loc=loc, ctx={"error": ValueError(msg)}
                    )
                    for loc, msg in errors
                ],
            )

        return self
