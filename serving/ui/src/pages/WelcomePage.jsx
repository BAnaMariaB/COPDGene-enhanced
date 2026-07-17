import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { getChampion } from '../api';
import ChampionPanel from '../components/ChampionPanel';

export default function WelcomePage() {
  const navigate = useNavigate();
  const [threshold, setThreshold] = useState('...');
  const thresholdValue = Number(threshold);
  const thresholdWidth = Number.isFinite(thresholdValue)
    ? `${Math.max(8, Math.min(100, thresholdValue * 100))}%`
    : '12%';

  useEffect(() => {
    let mounted = true;
    getChampion()
      .then((data) => {
        if (mounted) {
          const value = Number(data?.diagnosis_threshold);
          setThreshold(Number.isFinite(value) ? value.toFixed(2) : 'n/a');
        }
      })
      .catch(() => {
        if (mounted) setThreshold('n/a');
      });
    return () => {
      mounted = false;
    };
  }, []);

  return (
    <div className="welcome-page">
      <header className="welcome-hero">
        <div className="welcome-copy">
          <div className="section-kicker">COPDGene Enhanced</div>
          <h1>Respiratory risk assessment with a cleaner clinical front door.</h1>
          <p>
            The landing page stays deliberate. The interactive workspace lives on its own route and loads
            independently, with single-patient scoring, CSV batch inference, and DAG operations behind one entry point.
          </p>
          <div className="welcome-actions">
            <button className="button button-primary button-large" onClick={() => navigate('/demo')}>
              Try Demo
            </button>
          </div>
        </div>
        <div className="welcome-highlight">
          <div className="signal-card">
            <div className="signal-card-head">
              <span>Active champion gate</span>
              <span>Live</span>
            </div>
            <div className="signal-metric">{threshold}</div>
            <div className="signal-label">Current COPD decision threshold</div>
            <div className="signal-rail">
              <div className="signal-rail-fill" style={{ width: thresholdWidth }} />
            </div>
            <div className="signal-footnote">
              Threshold tuning is optimized in the training DAG before the champion row is published.
            </div>
          </div>
        </div>
      </header>

      <main className="welcome-main">
        <ChampionPanel />
      </main>
    </div>
  );
}
