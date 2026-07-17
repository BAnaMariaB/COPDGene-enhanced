-- Model registry: written by the modeling DAG, read by the FastAPI backend.
--
-- `is_champion` is scoped PER TARGET, not globally, so a second target can be
-- added later without a migration.
--
-- Targets today:
--   gold_copd   the repo's decided target (eda_feature_engineering/)
--   gold_stage  planned severity cascade; not built yet
--
-- No CHECK on target_name: CHANGES_MADE.md calls the same quantity
-- `copd_diagnosis`, and the naming isn't settled. The backend reads whatever
-- name COPD_DIAGNOSIS_TARGET is set to (default `gold_copd`).
CREATE TABLE IF NOT EXISTS model_registry (
    id              SERIAL PRIMARY KEY,
    model_name      TEXT        NOT NULL,
    mlflow_run_id   TEXT        NOT NULL,
    artifact_uri    TEXT        NOT NULL,
    target_name     TEXT        NOT NULL,
    class_labels    JSONB       NOT NULL,
    metrics         JSONB       NOT NULL,
    is_champion     BOOLEAN     NOT NULL DEFAULT FALSE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- At most one champion PER TARGET. Partial unique index: rows with
-- is_champion = FALSE are unconstrained, so history is retained.
CREATE UNIQUE INDEX IF NOT EXISTS one_champion_per_target
    ON model_registry (target_name) WHERE is_champion;

CREATE INDEX IF NOT EXISTS model_registry_run_id ON model_registry (mlflow_run_id);

-- Promotion is a single transaction, scoped to one target:
--   BEGIN;
--   UPDATE model_registry SET is_champion = FALSE
--     WHERE is_champion AND target_name = :target;
--   UPDATE model_registry SET is_champion = TRUE
--     WHERE mlflow_run_id = :run_id AND target_name = :target;
--   COMMIT;
--
-- class_labels order MUST match the column order of predict_proba, e.g.
--   gold_copd  -> ["no_copd", "copd"]   (or ["false", "true"] — gold_copd is boolean)
--   gold_stage -> ["gold_1", "gold_2", "gold_3", "gold_4"]
