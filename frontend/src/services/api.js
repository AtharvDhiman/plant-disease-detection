/**
 * Typed-ish wrapper around the FastAPI backend.
 *
 * Every call funnels through `request`, which normalises the backend's error
 * envelope (`{error: {code, message, hint}}`) into a thrown `ApiError`. That
 * keeps error handling in the components down to one `catch` per call instead
 * of a status-code ladder in each one.
 */

const BASE_URL = import.meta.env.VITE_API_BASE_URL || '';

export class ApiError extends Error {
  constructor(message, { code, status, hint } = {}) {
    super(message);
    this.name = 'ApiError';
    this.code = code || 'unknown_error';
    this.status = status;
    this.hint = hint;
  }
}

async function request(path, options = {}) {
  const { timeout = 60000, ...rest } = options;
  // Long-running predictions still need a ceiling, otherwise a hung backend
  // leaves the UI spinning forever with no way to recover.
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeout);

  let response;
  try {
    response = await fetch(`${BASE_URL}${path}`, { ...rest, signal: controller.signal });
  } catch (error) {
    clearTimeout(timer);
    if (error.name === 'AbortError') {
      throw new ApiError('The request timed out.', {
        code: 'timeout',
        hint: 'The server may be busy. Try again in a moment.',
      });
    }
    throw new ApiError('Could not reach the server.', {
      code: 'network_error',
      hint: 'Check that the backend is running on port 8000.',
    });
  }
  clearTimeout(timer);

  if (response.status === 204) return null;

  const contentType = response.headers.get('content-type') || '';
  const body = contentType.includes('application/json') ? await response.json() : null;

  if (!response.ok) {
    const detail = body?.error || {};
    throw new ApiError(detail.message || `Request failed (${response.status})`, {
      code: detail.code,
      status: response.status,
      hint: detail.hint,
    });
  }
  return body;
}

function query(params = {}) {
  const search = new URLSearchParams();
  Object.entries(params).forEach(([key, value]) => {
    if (value !== undefined && value !== null && value !== '') search.append(key, value);
  });
  const string = search.toString();
  return string ? `?${string}` : '';
}

/* --------------------------------------------------------------- prediction */

export const api = {
  health: () => request('/api/health', { timeout: 10000 }),

  modelHealth: () => request('/api/health/model', { timeout: 15000 }),

  analyzeImage: (file) => {
    const form = new FormData();
    form.append('file', file);
    return request('/api/analyze-image', { method: 'POST', body: form, timeout: 30000 });
  },

  predict: (file, { topK = 5, explain = true, save = true } = {}) => {
    const form = new FormData();
    form.append('file', file);
    return request(`/api/predict${query({ top_k: topK, explain, save })}`, {
      method: 'POST',
      body: form,
      // Grad-CAM adds a backward pass; on CPU-only hosts this can take a while.
      timeout: 120000,
    });
  },

  /* ------------------------------------------------------------- history */

  listPredictions: (params) => request(`/api/predictions${query(params)}`),
  getPrediction: (id) => request(`/api/predictions/${id}`),
  deletePrediction: (id) => request(`/api/predictions/${id}`, { method: 'DELETE' }),
  clearPredictions: () => request('/api/predictions?confirm=true', { method: 'DELETE' }),

  /* ------------------------------------------------------------ research */

  dashboard: (days = 30) => request(`/api/dashboard${query({ days })}`),
  models: () => request('/api/models'),
  metrics: (group) => request(`/api/metrics${query({ group })}`),
  ablation: () => request('/api/metrics/ablation'),
  perClass: (experimentId) => request(`/api/metrics/per-class${query({ experiment_id: experimentId })}`),
  curves: (experimentId) => request(`/api/metrics/curves/${experimentId}`),
  calibration: () => request('/api/metrics/calibration'),
  performance: () => request('/api/metrics/performance'),
  experiments: () => request('/api/experiments'),
  datasetStats: () => request('/api/dataset-stats'),
  diseases: (params) => request(`/api/diseases${query(params)}`),
  disease: (className) => request(`/api/diseases/${encodeURIComponent(className)}`),
  figures: () => request('/api/figures'),
};

/** Absolute URL of an image served by the backend. */
export const imageUrl = (path) => (path?.startsWith('http') ? path : `${BASE_URL}${path || ''}`);

/** Absolute URL of a rendered training figure. */
export const figureUrl = (group, filename) =>
  `${BASE_URL}/api/figures/${group}/${encodeURIComponent(filename)}`;
