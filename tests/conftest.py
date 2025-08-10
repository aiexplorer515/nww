import os, json, pytest

@pytest.fixture(scope="session")
def repo_paths():
    # assume tests run at repo root
    return {
        "schemas": os.path.abspath("schemas"),
        "golden": os.path.abspath("tests/golden")
    }
