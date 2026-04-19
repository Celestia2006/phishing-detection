import { useState } from "react";

const styles = `
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&display=swap');

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg: #0a0a0f;
    --surface: #111118;
    --border: #1e1e2e;
    --accent: #00ff88;
    --accent-dim: #00ff8822;
    --accent-mid: #00ff8855;
    --danger: #ff3b5c;
    --danger-dim: #ff3b5c22;
    --warn: #ffb800;
    --text: #e8e8f0;
    --muted: #5a5a7a;
    --mono: 'Space Mono', monospace;
    --sans: 'Syne', sans-serif;
  }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--sans);
    min-height: 100vh;
    overflow-x: hidden;
  }

  .noise {
    position: fixed;
    inset: 0;
    pointer-events: none;
    z-index: 0;
    opacity: 0.03;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)'/%3E%3C/svg%3E");
    background-size: 200px 200px;
  }

  .grid-bg {
    position: fixed;
    inset: 0;
    pointer-events: none;
    z-index: 0;
    background-image:
      linear-gradient(var(--border) 1px, transparent 1px),
      linear-gradient(90deg, var(--border) 1px, transparent 1px);
    background-size: 40px 40px;
    opacity: 0.4;
    mask-image: radial-gradient(ellipse 80% 80% at 50% 0%, black 40%, transparent 100%);
  }

  .glow-orb {
    position: fixed;
    width: 600px;
    height: 600px;
    border-radius: 50%;
    background: radial-gradient(circle, #00ff8815 0%, transparent 70%);
    top: -200px;
    left: 50%;
    transform: translateX(-50%);
    pointer-events: none;
    z-index: 0;
  }

  .app {
    position: relative;
    z-index: 1;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
  }

  /* NAV */
  nav {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 20px 48px;
    border-bottom: 1px solid var(--border);
    backdrop-filter: blur(12px);
    background: #0a0a0f88;
    position: sticky;
    top: 0;
    z-index: 10;
  }

  .logo {
    font-family: var(--mono);
    font-size: 13px;
    font-weight: 700;
    color: var(--accent);
    letter-spacing: 2px;
    text-transform: uppercase;
  }

  .logo span {
    color: var(--muted);
  }

  .nav-badge {
    font-family: var(--mono);
    font-size: 11px;
    color: var(--muted);
    border: 1px solid var(--border);
    padding: 4px 12px;
    border-radius: 100px;
    letter-spacing: 1px;
  }

  /* HERO */
  .hero {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 80px 24px 60px;
    text-align: center;
  }

  .eyebrow {
    font-family: var(--mono);
    font-size: 11px;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    gap: 10px;
  }

  .eyebrow::before,
  .eyebrow::after {
    content: '';
    display: block;
    width: 32px;
    height: 1px;
    background: var(--accent);
    opacity: 0.5;
  }

  h1 {
    font-size: clamp(36px, 6vw, 72px);
    font-weight: 800;
    line-height: 1.05;
    letter-spacing: -2px;
    margin-bottom: 20px;
    max-width: 800px;
  }

  h1 em {
    font-style: normal;
    color: var(--accent);
    position: relative;
  }

  .subtitle {
    font-family: var(--mono);
    font-size: 13px;
    color: var(--muted);
    max-width: 500px;
    line-height: 1.8;
    margin-bottom: 52px;
  }

  /* INPUT CARD */
  .input-card {
    width: 100%;
    max-width: 680px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 28px;
    position: relative;
    transition: border-color 0.3s;
  }

  .input-card:focus-within {
    border-color: var(--accent-mid);
    box-shadow: 0 0 40px var(--accent-dim);
  }

  .input-label {
    font-family: var(--mono);
    font-size: 11px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 12px;
    display: block;
  }

  .input-row {
    display: flex;
    gap: 12px;
    align-items: stretch;
  }

  .url-input {
    flex: 1;
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 14px 18px;
    font-family: var(--mono);
    font-size: 13px;
    color: var(--text);
    outline: none;
    transition: border-color 0.2s, box-shadow 0.2s;
    min-width: 0;
  }

  .url-input::placeholder { color: var(--muted); }

  .url-input:focus {
    border-color: var(--accent);
    box-shadow: 0 0 0 3px var(--accent-dim);
  }

  .scan-btn {
    background: var(--accent);
    color: var(--bg);
    border: none;
    border-radius: 10px;
    padding: 14px 24px;
    font-family: var(--sans);
    font-size: 14px;
    font-weight: 700;
    cursor: pointer;
    letter-spacing: 0.5px;
    transition: all 0.2s;
    white-space: nowrap;
    position: relative;
    overflow: hidden;
  }

  .scan-btn::after {
    content: '';
    position: absolute;
    inset: 0;
    background: white;
    opacity: 0;
    transition: opacity 0.2s;
  }

  .scan-btn:hover::after { opacity: 0.15; }
  .scan-btn:active { transform: scale(0.97); }

  .scan-btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  /* SCANNING ANIMATION */
  .scanning {
    margin-top: 20px;
    display: flex;
    align-items: center;
    gap: 10px;
    font-family: var(--mono);
    font-size: 12px;
    color: var(--accent);
  }

  .scanning-dots span {
    display: inline-block;
    animation: blink 1.2s infinite;
  }
  .scanning-dots span:nth-child(2) { animation-delay: 0.2s; }
  .scanning-dots span:nth-child(3) { animation-delay: 0.4s; }

  @keyframes blink {
    0%, 100% { opacity: 0.2; }
    50% { opacity: 1; }
  }

  /* RESULT CARD */
  .result-card {
    margin-top: 20px;
    border-radius: 12px;
    padding: 20px;
    border: 1px solid;
    animation: slideUp 0.4s ease;
  }

  @keyframes slideUp {
    from { opacity: 0; transform: translateY(12px); }
    to { opacity: 1; transform: translateY(0); }
  }

  .result-card.safe {
    background: #00ff8808;
    border-color: #00ff8840;
  }

  .result-card.danger {
    background: #ff3b5c08;
    border-color: #ff3b5c40;
  }

  .result-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 14px;
  }

  .result-icon {
    font-size: 22px;
    line-height: 1;
  }

  .result-title {
    font-size: 16px;
    font-weight: 700;
  }

  .result-card.safe .result-title { color: var(--accent); }
  .result-card.danger .result-title { color: var(--danger); }

  .result-url {
    font-family: var(--mono);
    font-size: 11px;
    color: var(--muted);
    margin-top: 2px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .trust-bar-wrap {
    margin-top: 4px;
  }

  .trust-label {
    display: flex;
    justify-content: space-between;
    font-family: var(--mono);
    font-size: 11px;
    color: var(--muted);
    margin-bottom: 6px;
  }

  .trust-bar {
    height: 6px;
    background: var(--border);
    border-radius: 100px;
    overflow: hidden;
  }

  .trust-fill {
    height: 100%;
    border-radius: 100px;
    transition: width 0.8s cubic-bezier(0.16, 1, 0.3, 1);
  }

  .result-card.safe .trust-fill { background: var(--accent); }
  .result-card.danger .trust-fill { background: var(--danger); }

  /* STATS ROW */
  .stats-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
    margin-top: 32px;
    width: 100%;
    max-width: 680px;
  }

  .stat-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px;
    text-align: left;
  }

  .stat-value {
    font-family: var(--mono);
    font-size: 28px;
    font-weight: 700;
    color: var(--accent);
    line-height: 1;
    margin-bottom: 6px;
  }

  .stat-label {
    font-size: 12px;
    color: var(--muted);
    font-family: var(--mono);
    letter-spacing: 0.5px;
  }

  /* FEATURES */
  .features {
    display: flex;
    gap: 12px;
    margin-top: 48px;
    flex-wrap: wrap;
    justify-content: center;
    max-width: 680px;
  }

  .feature-chip {
    font-family: var(--mono);
    font-size: 11px;
    color: var(--muted);
    border: 1px solid var(--border);
    padding: 8px 14px;
    border-radius: 100px;
    letter-spacing: 0.5px;
    display: flex;
    align-items: center;
    gap: 6px;
    transition: all 0.2s;
  }

  .feature-chip:hover {
    border-color: var(--accent-mid);
    color: var(--accent);
  }

  .feature-chip-dot {
    width: 5px;
    height: 5px;
    border-radius: 50%;
    background: var(--accent);
    opacity: 0.6;
  }

  /* FOOTER */
  footer {
    padding: 20px 48px;
    border-top: 1px solid var(--border);
    display: flex;
    align-items: center;
    justify-content: space-between;
  }

  .footer-text {
    font-family: var(--mono);
    font-size: 11px;
    color: var(--muted);
    letter-spacing: 1px;
  }

  .model-badges {
    display: flex;
    gap: 8px;
  }

  .model-badge {
    font-family: var(--mono);
    font-size: 10px;
    color: var(--muted);
    border: 1px solid var(--border);
    padding: 3px 10px;
    border-radius: 4px;
    letter-spacing: 0.5px;
  }

  @media (max-width: 600px) {
    nav { padding: 16px 20px; }
    .hero { padding: 60px 20px 40px; }
    .stats-row { grid-template-columns: 1fr 1fr; }
    footer { flex-direction: column; gap: 12px; padding: 20px; }
    .input-row { flex-direction: column; }
    .scan-btn { width: 100%; }
  }
`;

const FEATURES = [
  "SHAP Explainability",
  "WHOIS Analysis",
  "Trust Score",
  "Real-Time Warnings",
  "Adaptive Learning",
  "Model Comparison",
];

const MOCK_RESULT = {
  url: "http://paypal-secure-login.xyz/verify",
  safe: false,
  trust: 12,
  label: "Phishing Detected",
};

export default function App() {
  const [url, setUrl] = useState("");
  const [scanning, setScanning] = useState(false);
  const [result, setResult] = useState(null);

  const handleScan = async () => {
    if (!url.trim()) return;
    setResult(null);
    setScanning(true);
    try {
      const response = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url }),
      });
      const data = await response.json();
      setResult({
        url,
        safe: !data.is_phishing,
        trust: data.trust_score,
        label: data.is_phishing ? "Phishing Detected" : "Legitimate Website",
        warning: data.warning,
      });
    } catch (error) {
      setResult({
        url,
        safe: false,
        trust: 0,
        label: "Error — Could not reach backend",
        warning: null,
      });
    } finally {
      setScanning(false);
    }
  };

  const handleKey = (e) => {
    if (e.key === "Enter") handleScan();
  };

  return (
    <>
      <style>{styles}</style>
      <div className="noise" />
      <div className="grid-bg" />
      <div className="glow-orb" />

      <div className="app">
        {/* NAV */}
        <nav>
          <div className="logo">
            PHISH<span>///</span>GUARD
          </div>
          <div className="nav-badge">v1.0 · BETA</div>
        </nav>

        {/* HERO */}
        <main className="hero">
          <div className="eyebrow">Threat Intelligence System</div>

          <h1>
            Detect <em>Phishing</em>
            <br />
            Before It Strikes
          </h1>

          <p className="subtitle">
            AI-powered URL analysis using Logistic Regression, Random Forest &
            XGBoost — with explainable predictions and domain intelligence.
          </p>

          {/* INPUT */}
          <div className="input-card">
            <span className="input-label">Enter URL to scan</span>
            <div className="input-row">
              <input
                className="url-input"
                type="text"
                placeholder="https://example.com"
                value={url}
                onChange={(e) => setUrl(e.target.value)}
                onKeyDown={handleKey}
              />
              <button
                className="scan-btn"
                onClick={handleScan}
                disabled={scanning || !url.trim()}
              >
                {scanning ? "Scanning..." : "Scan URL →"}
              </button>
            </div>

            {scanning && (
              <div className="scanning">
                <span>Analyzing threat vectors</span>
                <span className="scanning-dots">
                  <span>.</span>
                  <span>.</span>
                  <span>.</span>
                </span>
              </div>
            )}

            {result && (
              <div className={`result-card ${result.safe ? "safe" : "danger"}`}>
                <div className="result-header">
                  <span className="result-icon">
                    {result.safe ? "🛡️" : "⚠️"}
                  </span>
                  <div>
                    <div className="result-title">{result.label}</div>
                    <div className="result-url">{result.url}</div>
                  </div>
                </div>
                <div className="trust-bar-wrap">
                  <div className="trust-label">
                    <span>Trust Score</span>
                    <span>{result.trust}%</span>
                  </div>
                  <div className="trust-bar">
                    <div
                      className="trust-fill"
                      style={{ width: `${result.trust}%` }}
                    />
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* STATS */}
          <div className="stats-row">
            <div className="stat-card">
              <div className="stat-value">3</div>
              <div className="stat-label">ML Models</div>
            </div>
            <div className="stat-card">
              <div className="stat-value">30</div>
              <div className="stat-label">URL Features</div>
            </div>
            <div className="stat-card">
              <div className="stat-value">97%</div>
              <div className="stat-label">Accuracy</div>
            </div>
          </div>

          {/* FEATURE CHIPS */}
          <div className="features">
            {FEATURES.map((f) => (
              <div className="feature-chip" key={f}>
                <span className="feature-chip-dot" />
                {f}
              </div>
            ))}
          </div>
        </main>

        {/* FOOTER */}
        <footer>
          <span className="footer-text">
            DEPT. OF COMPUTER SCIENCE & ENGINEERING
          </span>
          <div className="model-badges">
            <span className="model-badge">LR</span>
            <span className="model-badge">RF</span>
            <span className="model-badge">XGB</span>
          </div>
        </footer>
      </div>
    </>
  );
}
