import pytest
from pydantic import ValidationError

from spinner.schema import SpinnerMetadata

# TEST: metadata -----------------------------------------------------------------------


@pytest.fixture
def metadata():
    return {
        "description": "Lorem ipsum dolor sit amet.",
        "version": "1.0",
        "runs": 2,
        "timeout": 5,
        "retry": True,
        "retry_limit": 1,
    }


def test_metadata_valid(metadata):
    """This test should always pass."""
    SpinnerMetadata(**metadata)


# TEST: metadata.version ---------------------------------------------------------------


@pytest.fixture(params=["", "1", "1."])
def metadata_invalid_version(metadata, request):
    metadata["version"] = request.param
    return metadata


def test_metadata_version_invalid(metadata_invalid_version):
    with pytest.raises(ValidationError) as _error:
        SpinnerMetadata(**metadata_invalid_version)


# TEST: metadata.runs ------------------------------------------------------------------


@pytest.fixture(params=[0, -1])
def metadata_invalid_runs(metadata, request):
    metadata["runs"] = request.param
    return metadata


def test_metadata_runs_invalid(metadata_invalid_runs):
    with pytest.raises(ValidationError) as _error:
        SpinnerMetadata(**metadata_invalid_runs)


# TEST: metadata.timeout ---------------------------------------------------------------


@pytest.fixture(params=[0.0, -1.0])
def metadata_invalid_timeout(metadata, request):
    metadata["timeout"] = request.param
    return metadata


def test_metadata_timeout_none(metadata):
    metadata["timeout"] = None
    SpinnerMetadata(**metadata)


def test_metadata_timeout_invalid(metadata_invalid_timeout):
    with pytest.raises(ValidationError) as _error:
        SpinnerMetadata(**metadata_invalid_timeout)


# TEST: metadata.retry -----------------------------------------------------------------


@pytest.fixture
def metadata_no_retry(metadata):
    metadata.pop("retry", None)
    return metadata


def test_metadata_retry_missing(metadata_no_retry):
    SpinnerMetadata(**metadata_no_retry)


# TEST: metadata.retry_limit -----------------------------------------------------------


@pytest.fixture(params=[-1])
def metadata_invalid_retry_limit(metadata, request):
    metadata["retry_limit"] = request.param
    return metadata


def test_metadata_retry_limit_invalid(metadata_invalid_retry_limit):
    with pytest.raises(ValidationError) as _error:
        SpinnerMetadata(**metadata_invalid_retry_limit)
