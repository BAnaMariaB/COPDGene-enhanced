"""Shared test setup for the ingestion tests.

Puts the DAGs folder on the import path so `import copd_ingestion` works, and
points Airflow / the raw zone at throwaway temp dirs so tests never touch your
real `~/airflow` or `data/raw`.
"""

import os
import sys
import tempfile

# Repo root = one level up from tests/.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DAGS_DIR = os.path.join(REPO_ROOT, "airflow", "dags")

# Import path so the DAG modules are importable by name.
if DAGS_DIR not in sys.path:
    sys.path.insert(0, DAGS_DIR)

# Isolate Airflow + the raw landing zone into temp dirs (set before importing the
# DAG module, since RAW_ROOT is computed from these at import time).
os.environ.setdefault("AIRFLOW_HOME", tempfile.mkdtemp(prefix="copd_airflow_home_"))
os.environ.setdefault("COPD_RAW_ROOT", tempfile.mkdtemp(prefix="copd_raw_"))


def underlying(task_callable):
    """Return the plain Python function wrapped by an Airflow @task decorator.

    Lets us call the task body directly in a unit test without a running Airflow.
    """
    for attr in ("function", "__wrapped__"):
        fn = getattr(task_callable, attr, None)
        if callable(fn):
            return fn
    raise AttributeError("could not unwrap the underlying function from the task")
