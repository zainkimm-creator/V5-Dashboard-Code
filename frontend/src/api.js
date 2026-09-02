const LOCAL_DASHBOARD_API_BASE =
  typeof window !== 'undefined' &&
  ['127.0.0.1', 'localhost'].includes(window.location.hostname) &&
  window.location.port === '5202'
    ? 'http://127.0.0.1:8016'
    : 'http://127.0.0.1:8014';

export const DEFAULT_API_BASE =
  import.meta.env.VITE_API_BASE_URL ??
  import.meta.env.VITE_API_BASE ??
  LOCAL_DASHBOARD_API_BASE;

export async function apiGet(baseUrl, path) {
  const response = await fetch(`${baseUrl}${path}`);
  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText}`);
  }
  return response.json();
}

export async function apiPost(baseUrl, path, body = {}, options = {}) {
  const response = await fetch(`${baseUrl}${path}`, {
    ...options,
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...(options.headers ?? {}) },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(`${response.status} ${response.statusText}: ${text}`);
  }
  return response.json();
}

export function artifactUrl(baseUrl, artifactPath) {
  if (!artifactPath) return null;
  if (artifactPath.startsWith('http://') || artifactPath.startsWith('https://')) {
    return artifactPath;
  }
  return `${baseUrl}${artifactPath}`;
}
