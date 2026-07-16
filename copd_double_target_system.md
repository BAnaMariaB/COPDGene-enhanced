# COPDDoubleTargetSystem — Low-level architecture and training/evaluation details
This document describes the implemented ML system used by the Airflow DAG `copd_train_validate_test` (`airflow/dags/copd_train_validate_test.py`) on the preprocessed NHANES dataset.
## 1) Problem & targets
The system predicts two targets from the NHANES-derived cohort.
### Target A: `copd_diagnosis` (binary)
Definition (clinical GOLD screening criterion):
- `copd_diagnosis = 1` if `fev1_fvc_ratio < 0.70`
- `copd_diagnosis = 0` otherwise
In the preprocessed dataset this target is stored as integer values `0/1`.
### Target B: `gold_stage` (multiclass, 5 labels)
Definition:
- `gold_stage = 0` means **no COPD** (ratio ≥ 0.70)
- For COPD-positive samples, GOLD stage is assigned from `fev1_pct_predicted`:
  - GOLD 1: ≥ 80%
  - GOLD 2: 50–79%
  - GOLD 3: 30–49%
  - GOLD 4: < 30%
In the dataset, `gold_stage` is stored as integer values `0..4`.

Important training detail:
- The GOLD ensemble is trained only on true COPD rows, and only on stages GOLD_1..GOLD_4 (internally a 4-class problem).
- During inference/evaluation, GOLD_0 is treated as a sentinel for “no COPD”.
## 2) Input dataset & leakage control
### Preprocessed dataset path
The DAG reads:
- `data/preprocessed/<ds>/central_preprocessed_dataset.csv`
For this run, `ds=2026-07-16`.
### Columns
The CSV contains engineered demographic/exposure features plus spirometry columns.
### Leakage guard (per target)
Both targets are *defined* from spirometry-derived quantities. To prevent the model from simply reusing the already-derived definitions, the training code always drops the *target-definition* columns from features in `_build_targets()`:
- `fev1_fvc_ratio`
- `fev1_pct_predicted`

Raw spirometry measurements are handled differently per target:
- Diagnosis (`copd_diagnosis`): by default, raw spirometry is also dropped to avoid trivially reconstructing the ratio from `fev1_ml` and `fvc_ml`.
  - Controlled by `COPD_DIAGNOSIS_DROP_RAW_SPIROMETRY` (default: `1`)
- GOLD stage (`gold_stage`): raw spirometry is allowed by default because GOLD staging is clinically defined from spirometry.
  - Controlled by `COPD_GOLD_DROP_RAW_SPIROMETRY` (default: `0`)

Columns considered “raw spirometry”:
- `fev1_ml`
- `fvc_ml`
## 3) System architecture
The deployed model is one composite object:
- `COPDDoubleTargetSystem`
  - owns a diagnosis ensemble: `COPDEnsembleClassifier`
  - owns a GOLD-stage ensemble: `COPDEnsembleClassifier`
The two ensembles are *connected* via one additional feature:
- `diagnosis_proba_copd` = probability of COPD produced by the diagnosis ensemble.
This feature is appended to the feature vector used by the GOLD ensemble.
### 3.1) Ensemble architecture (COPDEnsembleClassifier)
Each `COPDEnsembleClassifier` is a stacking ensemble with two modes:
- Default (held-out stacking):
  1. Fit all base models on **(X_train, y_train)**.
  2. Compute meta-features on **X_val**:
     - `meta_features = hstack([base_i.predict_proba(X_val) for each base_i])`
  3. Fit meta-model on **(meta_features, y_val)**.
- Optional OOF stacking (recommended):
  - Enabled with `COPD_OOF_FOLDS > 1`.
  - Uses `StratifiedKFold` on the training split to generate out-of-fold (OOF) base-model probabilities for every training row.
  - Fits the meta-model on OOF meta-features for **X_train**.

During inference, both modes use the same flow:
- base models → `predict_proba` blocks → meta-model → final `predict_proba`.
### 3.2) Double-target connection
Training order:
1. Fit diagnosis ensemble.
2. Compute `diagnosis_proba_copd` for both train and validation splits.
3. Fit GOLD-stage ensemble on augmented features.
Inference order:
1. Run diagnosis ensemble to get `P(COPD)`.
2. Augment input features with `diagnosis_proba_copd`.
3. Run GOLD-stage ensemble.
## 4) Label encoding and COPD probability column
Targets are converted to string labels before encoding:
- `copd_diagnosis`: `0 → "no_copd"`, `1 → "copd"`
- `gold_stage`: `0..4 → "GOLD_0".."GOLD_4"`
Then `sklearn.preprocessing.LabelEncoder` is fit for each target.
Important detail:
- `LabelEncoder` sorts classes lexicographically, so the integer codes may not match the original dataset integers.
- The system therefore determines the correct COPD class index by locating the string label `"copd"` in `diagnosis_label_encoder.classes_`.
- `COPDDoubleTargetSystem._diagnosis_proba()` uses that index to extract the correct probability column.
This prevents a subtle but serious bug where `predict_proba()[:, 1]` might refer to `no_copd` depending on encoder ordering.
## 5) Model configurations (hyperparameters)
All model configs are defined in `DEFAULT_BASE_MODELS` and `DEFAULT_META_MODEL`.
### 5.1) Base model 1: CatBoostClassifier
Configured parameters:
- `iterations = 800` (boosting rounds)
- `learning_rate = 0.05`
- `depth = 6`
- `loss_function = "MultiClass"` (used for both targets; 2-class is treated as multiclass with 2 labels)
- `random_seed = 42`
- `verbose = False`
- `allow_writing_files = False`
Regularization:
- Not explicitly set here (CatBoost defaults apply).
### 5.2) Base model 2: XGBClassifier (XGBoost)
Configured parameters:
- `n_estimators = 300`
- `learning_rate = 0.05`
- `max_depth = 5`
- `subsample = 0.8`
- `colsample_bytree = 0.8`
- `random_state = 42`
- `n_jobs = 2`
Objective/num_class behavior:
- `objective` is intentionally **not hardcoded**.
- XGBoost auto-infers:
  - binary → `binary:logistic`
  - multiclass → `multi:softprob` (+ correct `num_class`)
This avoids an XGBoost 2.1.x failure when `multi:softprob` is forced for a binary task.
Regularization:
- Not explicitly set here (XGBoost defaults apply, e.g. L2 `reg_lambda`).
### 5.3) Base model 3: LogisticRegression
Configured parameters:
- `max_iter = 2000`
- `random_state = 42`
Regularization:
- Default scikit-learn regularization is used (L2 penalty with default `C=1.0`, unless scikit-learn defaults change).
### 5.4) Meta-model: LogisticRegression
Configured parameters:
- `max_iter = 2000`
- `random_state = 42`
Input dimension:
- If a target has `K` classes and there are 3 base models, the meta-feature vector has size `3*K`.
## 6) Training procedure (train/val/test)
### Split sizes
- `TEST_SIZE = 0.15`
- `VAL_SIZE = 0.15`
- `RANDOM_STATE = 42`
Split logic:
1. Split (train+val) vs test using stratification on `copd_diagnosis`.
2. Split train vs val (within train+val) stratified on `copd_diagnosis`.
Rationale:
- `gold_stage` is highly imbalanced (e.g. GOLD_4 has very few samples), so stratifying on GOLD as well is not stable.
### Training
- Base models are trained on `X_train`.
- Meta-model training depends on stacking mode:
  - Default: fit on base-model probabilities on `X_val`.
  - OOF (`COPD_OOF_FOLDS>1`): fit on OOF base-model probabilities on `X_train`.
- Both ensembles use balanced `sample_weight` (when supported by the estimator) via `compute_sample_weight(class_weight="balanced", ...)`.
- GOLD ensemble additionally receives `diagnosis_proba_copd` as a feature.
- GOLD ensemble is trained only on true COPD rows (GOLD_1..GOLD_4).
### “Epochs”
There are no neural-network epochs.
- CatBoost uses `iterations=800` boosting rounds.
- XGBoost uses `n_estimators=300` boosting rounds.
- LogisticRegression uses iterative optimization capped by `max_iter=2000`.
## 7) Evaluation metrics
Computed by `_classification_metrics()`:
- `accuracy`
- `precision_macro`
- `recall_macro`
- `f1_macro`
- `roc_auc_ovr` (macro, one-vs-rest) only when it can be computed and is finite.
Implementation note (MLflow SQLite constraint):
- When ROC AUC cannot be computed (e.g. a class absent in `y_true`), the metric is omitted rather than logging NaN.
- Logging NaNs for many metrics at the same timestamp can violate MLflow’s SQLite unique constraint.
## 8) Artifacts, logging, and champion records
### Local artifacts
Per partition date `<ds>`, outputs are written under:
- `data/artifacts/<ds>/splits/` (X_train/X_val/X_test and y arrays)
- `data/artifacts/<ds>/models/copd_double_target_system/`
- `data/artifacts/<ds>/test_metrics.json`
- `data/artifacts/<ds>/champion_diagnosis.json`
- `data/artifacts/<ds>/champion_gold_stage.json`
### MLflow
Tracking:
- Tracking URI: `MLFLOW_TRACKING_URI` (defaults to `sqlite:///.../mlflow.db`)
- Experiment: `copd_nhanes_double_target_classification`
Logged items:
- Parameters: split sizes, class lists, base model names, etc.
- Validation and test metrics for:
  - full diagnosis ensemble
  - full gold ensemble
  - each base model in each ensemble
- Model artifacts:
  - the system directory is logged as an artifact
  - meta-models are also logged via `mlflow.sklearn.log_model`
### Champion selection semantics
Current behavior:
- The pipeline writes champion JSON records for the **current run**.
- It tags the current MLflow run with `champion=true`.
- It does not compare against historical champions (database registration is intentionally not implemented).
In other words, “champion” here means “this run’s registered model artifact”, not “best-ever across runs”.
## 9) Inference path (how predictions are produced)
### Diagnosis
1. Optionally drop raw spirometry columns (`fev1_ml`, `fvc_ml`) depending on `COPD_DIAGNOSIS_DROP_RAW_SPIROMETRY`.
2. Reindex input `X` to the feature order captured during training.
3. For each base model, compute `predict_proba(X)`.
4. Concatenate all base probabilities into `meta_features`.
5. Meta-model outputs final probabilities and predictions.
### GOLD stage
1. Compute `P(COPD)` via the diagnosis ensemble.
2. Optionally drop raw spirometry columns depending on `COPD_GOLD_DROP_RAW_SPIROMETRY`.
3. Append `diagnosis_proba_copd` to the feature vector.
4. If diagnosis predicts no-COPD, output GOLD_0.
5. Otherwise run the GOLD ensemble inference (4-class GOLD_1..GOLD_4) and shift the predicted class back into 1..4.
## 10) Latest validated run
Partition:
- `ds=2026-07-16`
Settings:
- `COPD_OOF_FOLDS=3`
- `COPD_DIAGNOSIS_DROP_RAW_SPIROMETRY=1`
- `COPD_GOLD_DROP_RAW_SPIROMETRY=0`
- `COPD_DIAGNOSIS_USE_SAMPLE_WEIGHTS=0`
- `COPD_GOLD_USE_SAMPLE_WEIGHTS=1`
- `COPD_TUNE_DIAGNOSIS_THRESHOLD=1`
- tuned `diagnosis_threshold=0.15`
MLflow run id:
- `8084ae58171344ab985f1b6190fb7b6c`
Test metrics (full double-target system):
- Diagnosis:
  - macro: `accuracy ≈ 0.8311`, `precision_macro ≈ 0.6536`, `recall_macro ≈ 0.7385`, `f1_macro ≈ 0.6782`
  - weighted: `f1_weighted ≈ 0.8491`
- GOLD stage:
  - macro (strict over GOLD_0..GOLD_4): `accuracy ≈ 0.8190`, `precision_macro ≈ 0.4310`, `recall_macro ≈ 0.4770`, `f1_macro ≈ 0.4407`
  - weighted: `f1_weighted ≈ 0.8420`

Notes on interpretation:
- Diagnosis performance is not artificially perfect because the diagnosis model does not get raw spirometry by default.
- GOLD staging is reported as a 5-label output (GOLD_0..GOLD_4), but the GOLD ensemble itself is trained only on true COPD rows (GOLD_1..GOLD_4). Overall GOLD metrics therefore depend on both diagnosis gating and staging accuracy.
- The diagnosis threshold is tuned on the validation split to improve downstream GOLD-stage metrics (because it controls how many samples are routed to GOLD_1..GOLD_4 vs set to GOLD_0).
