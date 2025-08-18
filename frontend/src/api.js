const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";

export async function analyzeHeadline(headline, maxTickers = 10) {
  const res = await fetch(`${API_BASE}/analyze_headline`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ headline, max_tickers: maxTickers })
  });
  if (!res.ok) throw new Error(await res.text() || `HTTP ${res.status}`);
  return res.json();
}

export async function health() {
  const res = await fetch(`${API_BASE}/health`);
  return res.json();
}
