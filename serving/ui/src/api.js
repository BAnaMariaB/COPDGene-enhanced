const API_BASE = '';

async function readJson(response) {
  const text = await response.text();
  try {
    return JSON.parse(text);
  } catch {
    return { raw: text };
  }
}

async function readJsonOrThrow(response) {
  const payload = await readJson(response);
  if (!response.ok) {
    const detail = payload?.detail ?? payload?.raw ?? `Request failed with ${response.status}`;
    throw new Error(String(detail));
  }
  return payload;
}

export async function getChampion() {
  const response = await fetch(`${API_BASE}/model`);
  return readJsonOrThrow(response);
}

export async function predict(payload) {
  const response = await fetch(`${API_BASE}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  return readJsonOrThrow(response);
}

export async function predictCsv(file) {
  const form = new FormData();
  form.append('file', file);
  const response = await fetch(`${API_BASE}/predict_csv`, {
    method: 'POST',
    body: form,
  });
  if (!response.ok) {
    throw new Error(await response.text());
  }
  return response.blob();
}

export async function triggerDag(path, payload) {
  const response = await fetch(`${API_BASE}/${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  return readJsonOrThrow(response);
}
