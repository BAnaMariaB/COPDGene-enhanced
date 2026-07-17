import React, { useEffect, useState } from "react";
import { getChampions, getSchema, postBatch, postPredict } from "./api.js";
import PredictForm from "./components/PredictForm.jsx";
import Readout from "./components/Readout.jsx";

const TARGET_LABEL = {
  gold_copd: "Screening",
  copd_diagnosis: "Screening",
  gold_stage: "Staging",
};

function ChampionPlate({ champions }) {
  const targets = champions?.targets ?? {};
  const entries = Object.entries(targets);

  return (
    <dl className="plate">
      <div className="plate-title">Models in service</div>
      {entries.length === 0 ? (
        <div className="plate-row">
          <dt>Status</dt>
          <dd>none registered</dd>
        </div>
      ) : (
        entries.map(([target, c]) => (
          <div className="plate-row" key={target}>
            <dt>{TARGET_LABEL[target] || target}</dt>
            <dd>
              {c.model_name}
              {typeof c.metrics?.f1_macro === "number"
                ? ` · F1 ${c.metrics.f1_macro.toFixed(3)}`
                : ""}
            </dd>
          </div>
        ))
      )}
    </dl>
  );
}

export default function App() {
  const [schema, setSchema] = useState(null);
  const [champions, setChampions] = useState(null);
  const [championError, setChampionError] = useState(null);

  const [values, setValues] = useState({});
  const [errors, setErrors] = useState({});
  const [result, setResult] = useState(null);
  const [predictError, setPredictError] = useState(null);
  const [busy, setBusy] = useState(false);

  const [batch, setBatch] = useState(null);
  const [batchBusy, setBatchBusy] = useState(false);

  // The schema always loads; champions may legitimately not exist yet.
  useEffect(() => {
    getSchema().then(setSchema).catch((e) => setPredictError(e.message));
    getChampions()
      .then(setChampions)
      .catch((e) => setChampionError(e.message));
  }, []);

  const handleChange = (name, value) => {
    setValues((v) => ({ ...v, [name]: value }));
    setErrors((e) => (e[name] ? { ...e, [name]: undefined } : e));
  };

  const handleSubmit = async () => {
    setBusy(true);
    setPredictError(null);
    setErrors({});
    try {
      const filled = Object.fromEntries(
        Object.entries(values).filter(([, v]) => v !== "" && v != null)
      );
      setResult(await postPredict(filled));
    } catch (err) {
      setResult(null);
      if (err.fieldErrors) setErrors(err.fieldErrors);
      else setPredictError(err.message);
    } finally {
      setBusy(false);
    }
  };

  const handleReset = () => {
    setValues({});
    setErrors({});
    setResult(null);
    setPredictError(null);
  };

  const handleBatch = async (file) => {
    if (!file) return;
    setBatchBusy(true);
    setBatch(null);
    try {
      setBatch(await postBatch(file));
    } catch (err) {
      setBatch({ error: err.message });
    } finally {
      setBatchBusy(false);
    }
  };

  return (
    <div className="shell">
      <header className="masthead">
        <div>
          <h1>
            COPD
            <br />
            Screening
          </h1>
          <p>
            Estimates airflow obstruction under the GOLD criterion — FEV1/FVC
            below 0.70 — from demographics, smoking history, vitals and CT
            imaging.
          </p>
        </div>
        <ChampionPlate champions={champions} />
      </header>

      {championError ? (
        <div className="notice">
          <strong>No models are being served</strong>
          {championError}
        </div>
      ) : null}

      <div className="columns">
        <main>
          {schema ? (
            <PredictForm
              schema={schema}
              values={values}
              errors={errors}
              busy={busy}
              onChange={handleChange}
              onSubmit={handleSubmit}
              onReset={handleReset}
            />
          ) : (
            <p className="empty">Loading the input schema…</p>
          )}

          <fieldset style={{ marginTop: 32 }}>
            <legend>Batch</legend>
            <p className="batch-head">Score a CSV</p>
            <p className="empty" style={{ marginBottom: 12 }}>
              One row per subject, with a header row using the field names above.
              Rows that fail validation are reported individually — they don't
              stop the run.
            </p>
            <label className="sr-only" htmlFor="batch-file">
              CSV file
            </label>
            <input
              id="batch-file"
              type="file"
              accept=".csv"
              disabled={batchBusy}
              onChange={(e) => handleBatch(e.target.files?.[0])}
            />
            {batchBusy ? <p className="batch-summary">Scoring…</p> : null}
            {batch?.error ? (
              <p className="batch-summary" style={{ color: "var(--flag)" }}>
                {batch.error}
              </p>
            ) : null}
            {batch && !batch.error ? (
              <p className="batch-summary">
                {batch.n_rows - batch.n_failed} of {batch.n_rows} rows scored
                {batch.n_failed ? ` · ${batch.n_failed} rejected` : ""}
                {" · "}
                {batch.results.filter((r) => r.gold_stage).length} staged
              </p>
            ) : null}
          </fieldset>
        </main>

        <aside>
          <Readout result={result} error={predictError} />
        </aside>
      </div>
    </div>
  );
}
