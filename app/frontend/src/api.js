// All calls go to /api/*, which vite (dev) or nginx (prod) proxies to FastAPI.
// Keeping one origin means no CORS config to get wrong at deploy time.

const BASE = import.meta.env.VITE_API_BASE || "/api";

async function request(path, options = {}) {
  const res = await fetch(`${BASE}${path}`, options);
  let body = null;
  try {
    body = await res.json();
  } catch {
    body = null;
  }
  if (!res.ok) {
    const err = new Error(
      (body && (typeof body.detail === "string" ? body.detail : body.detail?.message)) ||
        `Request failed (${res.status})`
    );
    err.status = res.status;
    err.fieldErrors = body?.detail?.field_errors ?? null;
    throw err;
  }
  return body;
}

export const getSchema = () => request("/model/schema");
export const getChampions = () => request("/model/champions");

export const postPredict = (features) =>
  request("/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ features }),
  });

export const postBatch = (file) => {
  const form = new FormData();
  form.append("file", file);
  return request("/predict/batch", { method: "POST", body: form });
};
