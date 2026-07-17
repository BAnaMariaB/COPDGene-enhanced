import React from "react";

/**
 * The input form renders entirely from GET /model/schema. Nothing about the
 * feature set is hardcoded here, so when Hamza's NHANES columns and derived
 * features land, this component doesn't change — only features.py does.
 */

function Field({ spec, value, error, onChange }) {
  const id = `f-${spec.name}`;
  return (
    <div
      className={`field ${error ? "invalid" : ""} ${
        spec.type === "multi_categorical" ? "field-wide" : ""
      }`}
    >
      <label htmlFor={id} id={`${id}-label`}>
        {spec.label}
        {spec.unit ? <span className="unit"> ({spec.unit})</span> : null}
        {spec.required ? <span className="req" aria-hidden="true">*</span> : null}
      </label>

      {spec.type === "multi_categorical" ? (
        <div className="checks" role="group" aria-labelledby={`${id}-label`}>
          {spec.options.map((o) => {
            const picked = Array.isArray(value) && value.includes(o);
            return (
              <label key={o} className="check">
                <input
                  type="checkbox"
                  checked={picked}
                  onChange={() => {
                    const current = Array.isArray(value) ? value : [];
                    onChange(
                      spec.name,
                      picked ? current.filter((v) => v !== o) : [...current, o]
                    );
                  }}
                />
                <span>{o}</span>
              </label>
            );
          })}
        </div>
      ) : spec.type === "categorical" ? (
        <select
          id={id}
          value={value ?? ""}
          onChange={(e) => onChange(spec.name, e.target.value)}
          aria-invalid={!!error}
          aria-describedby={error ? `${id}-err` : undefined}
        >
          <option value="">—</option>
          {spec.options.map((o) => (
            <option key={o} value={o}>
              {spec.option_labels?.[o] ?? o}
            </option>
          ))}
        </select>
      ) : (
        <input
          id={id}
          type="number"
          inputMode="decimal"
          step="any"
          min={spec.min}
          max={spec.max}
          value={value ?? ""}
          onChange={(e) => onChange(spec.name, e.target.value)}
          aria-invalid={!!error}
          aria-describedby={error ? `${id}-err` : undefined}
        />
      )}

      {error ? (
        <p className="err" id={`${id}-err`}>
          {error}
        </p>
      ) : spec.help ? (
        <p className="help">{spec.help}</p>
      ) : null}
    </div>
  );
}

export default function PredictForm({
  schema,
  values,
  errors,
  busy,
  onChange,
  onSubmit,
  onReset,
}) {
  const byGroup = schema.groups.map((g) => ({
    name: g,
    fields: schema.features.filter((f) => f.group === g),
  }));

  return (
    <div>
      {byGroup.map((group) => (
        <fieldset key={group.name}>
          <legend>{group.name}</legend>
          <div className="grid">
            {group.fields.map((spec) => (
              <Field
                key={spec.name}
                spec={spec}
                value={values[spec.name]}
                error={errors[spec.name]}
                onChange={onChange}
              />
            ))}
          </div>
        </fieldset>
      ))}

      <div className="actions">
        <button className="btn-primary" onClick={onSubmit} disabled={busy}>
          {busy ? "Screening…" : "Run screen"}
        </button>
        <button className="btn-ghost" onClick={onReset} disabled={busy}>
          Clear
        </button>
      </div>
    </div>
  );
}
