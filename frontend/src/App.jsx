import React, { useEffect, useState } from "react";
import { analyzeHeadline, health } from "./api";

const Arrow = ({ dir }) => {
  if (dir === "up") return <span title="Predicted up">▲</span>;
  if (dir === "down") return <span title="Predicted down">▼</span>;
  return <span title="Neutral">→</span>;
};

export default function App() {
  const [headline, setHeadline] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [apiInfo, setApiInfo] = useState(null);

  useEffect(() => {
    health().then(setApiInfo).catch(() => {});
  }, []);

  async function onSubmit(e) {
    e.preventDefault();
    setError(""); setResult(null);
    const text = headline.trim();
    if (!text) { setError("Please type a headline."); return; }
    try {
      setLoading(true);
      const r = await analyzeHeadline(text, 10);
      setResult(r);
    } catch (err) {
      setError(err.message || "Request failed.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="container">
      <header>
        <h1>Headline Impact Analyzer</h1>
        <p className="sub">
          Type a headline → model sentiment/impact + matched tickers + live quotes + a hypothetical projection.
        </p>
        {apiInfo && (
          <p className="health">
            {/* Overlay details removed */}
            API OK • Headline model: {String(apiInfo.headline_model).split("\\").pop()} • Device: {apiInfo.device}
          </p>
        )}
      </header>

      <form onSubmit={onSubmit} className="card input-card">
        <label htmlFor="headline">Headline</label>
        <textarea
          id="headline"
          rows={3}
          placeholder="e.g., Apple recalls iPhones due to battery fires; $AAPL plunges"
          value={headline}
          onChange={(e) => setHeadline(e.target.value)}
        />
        <button type="submit" disabled={loading}>{loading ? "Analyzing..." : "Analyze"}</button>
        {error && <div className="error">{error}</div>}
      </form>

      {result && (
        <section className="card">
          <h2>Result</h2>
          <div className="result-row">
            <div>
              <div className="label">Predicted Label</div>
              <div className="badge">{result.sentiment}</div>
              <div className="tiny">confidence: {(result.confidence * 100).toFixed(1)}%</div>
            </div>
            <div>
              <div className="label">Impact</div>
              <div className="impact"><Arrow dir={result.predicted_impact} /> {result.predicted_impact}</div>
            </div>
            <div>
              <div className="label">Matched Tickers</div>
              <div className="chips">
                {result.tickers?.length ? result.tickers.map(t => <span key={t} className="chip">{t}</span>) : <span className="tiny">none</span>}
              </div>
            </div>
          </div>

          {result.quotes?.length ? (
            <>
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>Ticker</th>
                      <th>Price</th>
                      <th>Change</th>
                      <th>Change %</th>
                      <th>Projected (model)</th>
                      <th>As of</th>
                    </tr>
                  </thead>
                  <tbody>
                  {result.quotes.map(q => (
                    <tr key={q.symbol}>
                      {/* removed the tooltip that exposed overlay numbers */}
                      <td>{q.symbol}</td>
                      <td>{q.price}</td>
                      <td className={q.change >= 0 ? "pos" : "neg"}>{q.change >= 0 ? `+${q.change}` : q.change}</td>
                      <td className={q.change_pct >= 0 ? "pos" : "neg"}>{q.change_pct >= 0 ? `+${q.change_pct}%` : `${q.change_pct}%`}</td>
                      <td className={q.predicted_change_pct >= 0 ? "pos" : "neg"}>
                        {q.predicted_change_pct >= 0 ? `+${q.predicted_change_pct}%` : `${q.predicted_change_pct}%`}
                        {" → "}{q.projected_price}
                      </td>
                      <td>
                        <span className="tiny">
                          {new Date(q.asof).toLocaleString()}
                          {q.is_stale ? " • market closed?" : ""}
                        </span>
                      </td>
                    </tr>
                  ))}
                  </tbody>
                </table>
              </div>
              <div className="tiny" style={{marginTop:4}}>
                Projection is hypothetical, based on model confidence/keywords/volatility (not financial advice).
              </div>
            </>
          ) : <div className="tiny">No live quotes found for matched tickers.</div>}

          <details style={{marginTop:"1rem"}}>
            <summary>Scores (debug)</summary>
            <pre className="pre">{JSON.stringify(result.scores, null, 2)}</pre>
          </details>
        </section>
      )}

      <footer><div className="tiny">Data via Yahoo Finance (may be delayed). Not financial advice.</div></footer>
    </div>
  );
}
