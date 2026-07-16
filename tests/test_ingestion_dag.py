"""DAG-level tests: the DAG parses and is wired correctly.

The key structural guarantee for our ingestion-only scope: the two context
sources are landed raw and gate completion, but are NOT merged into the
sid-level preprocessing (they have no join key).
"""

import os

from airflow.models import DagBag

import copd_ingestion as ci


def test_dagbag_imports_cleanly():
    bag = DagBag(os.path.dirname(ci.__file__))
    assert bag.import_errors == {}
    assert "copd_ingestion" in bag.dags


def test_expected_tasks_present():
    dag = ci.copd_dag
    task_ids = set(dag.task_dict)
    for expected in (
        "ingest_cdc_copd_prevalence",
        "ingest_smoking_prevalence",
        "preprocessing",
        "ingestion_complete",
    ):
        assert expected in task_ids, f"missing task {expected}"


def test_context_sources_gate_completion_but_are_not_merged():
    dag = ci.copd_dag
    for context_id in ("ingest_cdc_copd_prevalence", "ingest_smoking_prevalence"):
        task = dag.get_task(context_id)
        # Gates the completion marker...
        assert "ingestion_complete" in task.downstream_task_ids
        # ...but is deliberately NOT fed into the sid-level merge.
        assert "preprocessing" not in task.downstream_task_ids


def test_preprocessing_consumes_only_the_three_core_sources():
    dag = ci.copd_dag
    pre = dag.get_task("preprocessing")
    assert len(pre.upstream_task_ids) == 3
    assert "ingest_cdc_copd_prevalence" not in pre.upstream_task_ids
    assert "ingest_smoking_prevalence" not in pre.upstream_task_ids
