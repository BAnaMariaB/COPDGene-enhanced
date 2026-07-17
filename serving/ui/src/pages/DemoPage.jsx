import React, { useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import { predict, predictCsv, triggerDag } from '../api';
import ChampionPanel from '../components/ChampionPanel';

const EXAMPLES = {
  scenarioA: {
    age: 58,
    sex: 1,
    race_ethnicity: 3,
    height_cm: 172,
    weight_kg: 82,
    bmi: 27.7,
    pack_years: 20,
    smoking_status: 'former',
    fev1_ml: 2100,
    fvc_ml: 3200,
  },
  scenarioB: {
    age: 67,
    sex: 0,
    race_ethnicity: 2,
    height_cm: 165,
    weight_kg: 74,
    bmi: 27.2,
    pack_years: 35,
    smoking_status: 'current',
    fev1_ml: 1450,
    fvc_ml: 2450,
  },
};

const STAGE_LABELS = ['No COPD', 'GOLD 1', 'GOLD 2', 'GOLD 3', 'GOLD 4'];

function formatPercent(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return 'n/a';
  return `${(numeric * 100).toFixed(1)}%`;
}

function PredictionInsights({ result }) {
  if (!result) {
    return <div className="empty-state">Prediction results will appear here after the model runs.</div>;
  }

  const probabilities = Array.isArray(result.gold_stage_proba) ? result.gold_stage_proba : [];
  const threshold = Number(result?.champion?.diagnosis_threshold);
  const hasCopd = Number(result.copd_pred) === 1;
  const stageIndex = Number(result.gold_stage_pred) || 0;

  return (
    <div className="result-panel">
      <div className="result-grid">
        <article className="result-card result-card-strong">
          <span className="result-label">COPD risk</span>
          <strong>{formatPercent(result.copd_proba)}</strong>
          <span className={`status-pill ${hasCopd ? 'status-alert' : 'status-ok'}`}>
            {hasCopd ? 'COPD predicted' : 'No COPD predicted'}
          </span>
        </article>
        <article className="result-card">
          <span className="result-label">Predicted GOLD stage</span>
          <strong>{STAGE_LABELS[stageIndex] ?? `Stage ${stageIndex}`}</strong>
          <span className="result-muted">
            Threshold {Number.isFinite(threshold) ? threshold.toFixed(2) : 'n/a'}
          </span>
        </article>
      </div>

      <div className="distribution-card">
        <div className="distribution-head">
          <h3>Stage distribution</h3>
          <span>{hasCopd ? 'Conditional GOLD distribution' : 'Diagnosis gate held at GOLD 0'}</span>
        </div>
        <div className="distribution-list">
          {STAGE_LABELS.map((label, index) => (
            <div className="distribution-row" key={label}>
              <span>{label}</span>
              <div className="distribution-bar">
                <div
                  className="distribution-fill"
                  style={{ width: `${Math.max(4, Math.min(100, Number(probabilities[index] || 0) * 100))}%` }}
                />
              </div>
              <strong>{formatPercent(probabilities[index])}</strong>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function PredictionSection() {
  const [form, setForm] = useState(EXAMPLES.scenarioA);
  const [result, setResult] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');

  const fields = useMemo(
    () => [
      ['age', 'Age'],
      ['sex', 'Sex'],
      ['race_ethnicity', 'Race / ethnicity'],
      ['height_cm', 'Height (cm)'],
      ['weight_kg', 'Weight (kg)'],
      ['bmi', 'BMI'],
      ['pack_years', 'Pack-years'],
      ['smoking_status', 'Smoking status', 'text'],
      ['fev1_ml', 'FEV1 (ml)'],
      ['fvc_ml', 'FVC (ml)'],
    ],
    [],
  );

  async function onSubmit(event) {
    event.preventDefault();
    setBusy(true);
    setError('');
    try {
      const payload = Object.fromEntries(
        Object.entries(form).map(([key, value]) => [key, value === '' || Number.isNaN(value) ? null : value]),
      );
      const data = await predict(payload);
      setResult(data);
    } catch (requestError) {
      setError(requestError.message);
      setResult(null);
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="demo-card">
      <div className="section-kicker">Patient demo</div>
      <div className="card-headline">
        <h2>Run a single-patient respiratory assessment</h2>
        <p>Use one of the prepared cases or enter values manually.</p>
      </div>

      <form className="form-grid" onSubmit={onSubmit}>
        {fields.map(([key, label, type = 'number']) => (
          <label key={key} className="field">
            <span>{label}</span>
            <input
              type={type}
              value={form[key] ?? ''}
              onChange={(event) =>
                setForm({
                  ...form,
                  [key]: type === 'number'
                    ? (event.target.value === '' ? '' : event.target.valueAsNumber)
                    : event.target.value,
                })
              }
            />
          </label>
        ))}
        <div className="demo-actions">
          <button className="button button-primary" type="submit" disabled={busy}>
            {busy ? 'Running...' : 'Run prediction'}
          </button>
          <button className="button button-secondary" type="button" onClick={() => setForm(EXAMPLES.scenarioA)}>
            Load example A
          </button>
          <button className="button button-secondary" type="button" onClick={() => setForm(EXAMPLES.scenarioB)}>
            Load example B
          </button>
        </div>
      </form>

      {error ? <div className="error-box">{error}</div> : null}
      <PredictionInsights result={result} />
    </section>
  );
}

function BatchSection() {
  const [file, setFile] = useState(null);
  const [status, setStatus] = useState('');
  const [downloadUrl, setDownloadUrl] = useState('');
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);

  async function onSubmit(event) {
    event.preventDefault();
    if (!file) return;
    setBusy(true);
    setError('');
    setStatus('');
    try {
      const blob = await predictCsv(file);
      if (downloadUrl) {
        URL.revokeObjectURL(downloadUrl);
      }
      const url = URL.createObjectURL(blob);
      setDownloadUrl(url);
      setStatus(`Predictions generated for ${file.name}`);
    } catch (requestError) {
      setError(requestError.message);
      setDownloadUrl('');
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="demo-card">
      <div className="section-kicker">Batch scoring</div>
      <div className="card-headline">
        <h2>Score a cohort from CSV</h2>
        <p>Upload a file, append prediction columns, and download the scored output.</p>
      </div>

      <form className="upload-grid" onSubmit={onSubmit}>
        <label className="upload-field">
          <span>Input CSV</span>
          <input type="file" accept=".csv" onChange={(event) => setFile(event.target.files?.[0] ?? null)} />
        </label>
        <div className="demo-actions">
          <button className="button button-primary" type="submit" disabled={busy || !file}>
            {busy ? 'Scoring...' : 'Run batch'}
          </button>
          <a className="button button-secondary" href="/examples/batch_prediction_example.csv">
            Download example CSV
          </a>
        </div>
      </form>

      {status ? <div className="success-box">{status}</div> : null}
      {error ? <div className="error-box">{error}</div> : null}
      {downloadUrl ? (
        <div className="download-row">
          <a className="button button-secondary" href={downloadUrl} download="predictions.csv">
            Download predictions
          </a>
        </div>
      ) : null}
    </section>
  );
}

function OperationsSection() {
  const [ds, setDs] = useState('');
  const [response, setResponse] = useState(null);
  const [error, setError] = useState('');
  const [busyAction, setBusyAction] = useState('');

  async function run(path) {
    setBusyAction(path);
    setError('');
    try {
      const data = await triggerDag(path, { ds: ds || null, conf: {} });
      setResponse(data);
    } catch (requestError) {
      setError(requestError.message);
      setResponse(null);
    } finally {
      setBusyAction('');
    }
  }

  return (
    <section className="demo-card">
      <div className="section-kicker">Pipeline control</div>
      <div className="card-headline">
        <h2>Trigger ingestion or champion refresh</h2>
        <p>The API forwards each request to Airflow and returns the DAG trigger response.</p>
      </div>

      <div className="ops-layout">
        <label className="field">
          <span>Logical date</span>
          <input value={ds} onChange={(event) => setDs(event.target.value)} placeholder="2026-07-17" type="text" />
        </label>
        <div className="demo-actions">
          <button
            className="button button-secondary"
            type="button"
            disabled={busyAction === 'trigger_ingestion'}
            onClick={() => run('trigger_ingestion')}
          >
            {busyAction === 'trigger_ingestion' ? 'Triggering...' : 'Trigger ingestion'}
          </button>
          <button
            className="button button-secondary"
            type="button"
            disabled={busyAction === 'trigger_training'}
            onClick={() => run('trigger_training')}
          >
            {busyAction === 'trigger_training' ? 'Triggering...' : 'Trigger training'}
          </button>
        </div>
      </div>

      {error ? <div className="error-box">{error}</div> : null}
      <div className="ops-response">
        {response ? (
          <div className="ops-grid">
            {Object.entries(response).map(([key, value]) => (
              <div className="ops-item" key={key}>
                <span>{key}</span>
                <strong>{String(value)}</strong>
              </div>
            ))}
          </div>
        ) : (
          <div className="empty-state">No DAG trigger has been sent yet.</div>
        )}
      </div>
    </section>
  );
}

export default function DemoPage() {
  return (
    <div className="demo-page">
      <header className="demo-hero">
        <div className="demo-hero-copy">
          <div className="section-kicker">Interactive workspace</div>
          <h1>Prediction, cohort scoring, and pipeline actions in one separate route.</h1>
          <p>This page is lazy-loaded as its own chunk and keeps the actual product workflow away from the welcome screen.</p>
        </div>
        <div className="demo-nav">
          <Link className="button button-secondary" to="/welcome">
            Back to welcome
          </Link>
        </div>
      </header>

      <main className="demo-main">
        <ChampionPanel compact />
        <PredictionSection />
        <BatchSection />
        <OperationsSection />
      </main>
    </div>
  );
}
