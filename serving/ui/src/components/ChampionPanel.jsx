import React, { useEffect, useState } from 'react';
import { getChampion } from '../api';

function ChampionRow({ label, value, subvalue }) {
  return (
    <div className="champion-row">
      <div className="champion-label">{label}</div>
      <div className="champion-value" title={String(value ?? '')}>{value ?? 'n/a'}</div>
      {subvalue ? <div className="champion-subvalue">{subvalue}</div> : null}
    </div>
  );
}

function formatMetric(champion) {
  if (!champion?.metric_name) return 'n/a';
  const value = typeof champion.metric_value === 'number'
    ? champion.metric_value.toFixed(4)
    : champion.metric_value;
  return `${champion.metric_name} ${value ?? ''}`.trim();
}

export default function ChampionPanel({ compact = false }) {
  const [champion, setChampion] = useState(null);

  useEffect(() => {
    let mounted = true;
    getChampion()
      .then((data) => {
        if (mounted) setChampion(data);
      })
      .catch((error) => {
        if (mounted) setChampion({ error: error.message });
      });
    return () => {
      mounted = false;
    };
  }, []);

  return (
    <section className={compact ? 'champion-panel champion-panel-compact' : 'champion-panel'}>
      <div className="section-kicker">Active champion</div>
      {!champion ? <div className="muted">Loading champion metadata...</div> : null}
      {champion?.error ? <div className="error-box">{champion.error}</div> : null}
      {champion && !champion.error ? (
        <div className="champion-grid">
          <ChampionRow label="Model" value={champion.model_name} subvalue={champion.target} />
          <ChampionRow label="Metric" value={formatMetric(champion)} subvalue="Champion selection metric" />
          <ChampionRow label="Threshold" value={champion.diagnosis_threshold} subvalue="COPD decision gate" />
          <ChampionRow label="Run" value={champion.mlflow_run_id} subvalue={champion.registered_at} />
        </div>
      ) : null}
    </section>
  );
}
