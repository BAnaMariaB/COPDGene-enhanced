import React from "react";

/**
 * The result readout, drawn as pen deflections on the chart paper.
 *
 * Two stages, mirroring the pipeline: screening, then staging. Staging only
 * appears for screen-positive subjects, because the gold_stage model is trained
 * exclusively on COPD-positive rows and has no concept of a negative.
 *
 * Class names come from the champion rows in Postgres, not from here, so the
 * labels stay correct if the team renames or re-orders them.
 */

// gold_copd is a boolean target: a champion may register classes as
// ["false","true"], ["no_copd","copd"] or ["0","1"]. Treat all negatives alike.
const NEGATIVE = new Set(["no_copd", "false", "0", "none", "negative", "normal"]);
const isFlagged = (label) => !NEGATIVE.has(String(label).toLowerCase());
const pretty = (label) => String(label).replace(/_/g, " ");

const SKIP_COPY = {
  screen_negative:
    "No staging. GOLD severity is only defined for subjects who screen positive — a staging model is trained on COPD-positive cases only and never sees a negative example.",
  no_gold_stage_champion:
    "No staging. Severity staging isn't built yet — it needs age/sex/height-adjusted % predicted FEV1 from reference equations. Screening stands on its own.",
};

function Trace({ probabilities, top, flagged }) {
  if (!probabilities) {
    return (
      <p className="empty">
        This model returned a label without probabilities. Wrap it so
        <code>predict()</code> returns <code>predict_proba</code> output to see the
        distribution here.
      </p>
    );
  }
  const rows = Object.entries(probabilities).sort((a, b) => b[1] - a[1]);
  return (
    <div>
      {rows.map(([label, p]) => {
        const isMax = label === top;
        return (
          <div
            key={label}
            className={`trace-row ${isMax ? "is-max" : ""} ${
              isMax && flagged ? "flagged" : ""
            }`}
          >
            <div>
              <div className="trace-label">{pretty(label)}</div>
              <div className="trace-track">
                <div
                  className="trace-fill"
                  style={{ width: `${Math.max(p * 100, 0.5)}%` }}
                />
              </div>
            </div>
            <div className="trace-value">{(p * 100).toFixed(1)}%</div>
          </div>
        );
      })}
    </div>
  );
}

function Stage({ eyebrow, stage }) {
  const flagged = isFlagged(stage.prediction);
  return (
    <div className="stage">
      <p className="stage-eyebrow">{eyebrow}</p>
      <p className={`verdict ${flagged ? "flagged" : ""}`}>{pretty(stage.prediction)}</p>
      <p className="verdict-sub">
        {stage.model_name} · {stage.mlflow_run_id?.slice(0, 12)}
      </p>
      <Trace
        probabilities={stage.probabilities}
        top={stage.prediction}
        flagged={flagged}
      />
    </div>
  );
}

export default function Readout({ result, error }) {
  if (error) {
    return (
      <section className="readout" aria-live="polite">
        <div className="readout-head">Readout</div>
        <div className="readout-body">
          <div className="notice" style={{ margin: 0 }}>
            <strong>Can't predict yet</strong>
            {error}
          </div>
        </div>
      </section>
    );
  }

  if (!result) {
    return (
      <section className="readout" aria-live="polite">
        <div className="readout-head">Readout</div>
        <div className="readout-body">
          <p className="empty">
            Fill in the subject's details and run the screen. The result appears
            here: whether the subject screens positive for COPD under the GOLD
            criterion, and how confident the model is.
          </p>
        </div>
      </section>
    );
  }

  const skip = result.staging_skipped_reason;

  return (
    <section className="readout" aria-live="polite">
      <div className="readout-head">Readout</div>
      <div className="readout-body">
        <Stage eyebrow="Step 1 · Screening" stage={result.diagnosis} />

        <hr className="hair" />

        {result.gold_stage ? (
          <Stage eyebrow="Step 2 · Severity staging" stage={result.gold_stage} />
        ) : (
          <div className="stage">
            <p className="stage-eyebrow">Step 2 · Severity staging</p>
            <p className="empty">{SKIP_COPY[skip] || "Staging unavailable."}</p>
          </div>
        )}

        <p className="empty" style={{ marginTop: 20, fontSize: 11.5 }}>
          Screening target is the GOLD criterion (FEV1/FVC &lt; 0.70). Research
          output from a student project — not a diagnostic device, and not for
          clinical use.
        </p>
      </div>
    </section>
  );
}
