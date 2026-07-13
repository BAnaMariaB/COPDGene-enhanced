"""
COPD model training / validation / testing DAG.

SCOPE: placeholder orchestration only. This DAG is a scaffold for future
modeling work and intentionally contains empty task bodies to be filled in.
The intended flow is:

  start -> train -> validate -> test -> select_champion -> complete

No modeling logic is implemented yet.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow.sdk import dag, task


# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------

default_args = {
    "owner": "ml",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


@dag(
    dag_id="copd_train_validate_test",
    description="Placeholder DAG for training, validating, testing COPD models, and selecting a champion.",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    tags=["copd", "ml", "training", "validation", "testing"],
)
def copd_train_validate_test():
    @task
    def start() -> None:
        """Explicit start marker for readability in the task graph."""
        pass

    @task
    def train_model() -> None:
        """Train candidate model(s)."""
        pass

    @task
    def validate_model() -> None:
        """Validate trained model(s)."""
        pass

    @task
    def test_model() -> None:
        """Run final model test evaluation."""
        pass

    @task
    def select_champion() -> None:
        """Select the champion model from evaluated candidates."""
        pass

    @task
    def complete() -> None:
        """Final marker task for downstream dependencies."""
        pass

    start_task = start()
    train_task = train_model()
    validate_task = validate_model()
    test_task = test_model()
    champion_task = select_champion()
    complete_task = complete()

    start_task >> train_task >> validate_task >> test_task >> champion_task >> complete_task


copd_dag = copd_train_validate_test()

