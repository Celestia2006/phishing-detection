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
    --warn-dim: #ffb80022;
    --text: #e8e8f0;
    --muted: #babfdc;
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

  .nav-left {
    display: flex;
    flex-direction: column;
  }

  .logo {
    font-family: var(--mono);
    font-size: 18px;
    font-weight: 700;
    color: var(--accent);
    letter-spacing: 2px;
    text-transform: uppercase;
  }

  .tagline {
    font-family: var(--mono);
    font-size: 10px;
    color: var(--muted);
    letter-spacing: 1px;
    margin-top: 2px;
  }

  .dark-toggle {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 8px;
    cursor: pointer;
    color: var(--text);
    transition: all 0.2s;
  }

  .dark-toggle:hover {
    border-color: var(--accent-mid);
    box-shadow: 0 0 20px var(--accent-dim);
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

  .hero-title {
    font-size: clamp(36px, 6vw, 72px);
    font-weight: 800;
    line-height: 1.05;
    letter-spacing: -2px;
    margin-bottom: 20px;
    max-width: 800px;
  }

  .hero-title em {
    font-style: normal;
    color: var(--accent);
    position: relative;
  }

  .hero-subtitle {
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
    backdrop-filter: blur(10px);
  }

  .input-card:focus-within {
    border-color: var(--accent-mid);
    box-shadow: 0 0 40px var(--accent-dim);
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
    position: relative;
  }

  .url-input::placeholder { color: var(--muted); }

  .url-input:focus {
    border-color: var(--accent);
    box-shadow: 0 0 0 3px var(--accent-dim);
  }

  .input-icon {
    position: absolute;
    right: 18px;
    top: 50%;
    transform: translateY(-50%);
    color: var(--muted);
    font-size: 16px;
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

  /* DASHBOARD */
  .dashboard {
    width: 100%;
    max-width: 1200px;
    margin: 0 auto;
    padding: 40px 24px;
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 24px;
  }

  .card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 24px;
    backdrop-filter: blur(10px);
    transition: all 0.3s;
  }

  .card:hover {
    border-color: var(--accent-mid);
    box-shadow: 0 0 40px var(--accent-dim);
  }

  .card-title {
    font-size: 16px;
    font-weight: 700;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .card-icon {
    font-size: 18px;
  }

  /* STATUS CARD */
  .status-card {
    grid-column: span 2;
  }

  .status-display {
    text-align: center;
    margin-bottom: 20px;
  }

  .status-label {
    font-size: 48px;
    font-weight: 800;
    margin-bottom: 8px;
  }

  .status-safe { color: var(--accent); }
  .status-phishing { color: var(--danger); }
  .status-suspicious { color: var(--warn); }

  .confidence {
    font-family: var(--mono);
    font-size: 14px;
    color: var(--muted);
  }

  .status-icon {
    font-size: 64px;
    margin-bottom: 16px;
  }

  /* TRUST SCORE CARD */
  .trust-card {
    position: relative;
  }

  .trust-meter {
    width: 120px;
    height: 120px;
    margin: 0 auto 16px;
    position: relative;
  }

  .trust-circle-bg {
    width: 100%;
    height: 100%;
    border-radius: 50%;
    background: conic-gradient(from 0deg, var(--border) 0deg, var(--border) 360deg);
    position: relative;
  }

  .trust-circle-fill {
    position: absolute;
    top: 8px;
    left: 8px;
    width: calc(100% - 16px);
    height: calc(100% - 16px);
    border-radius: 50%;
    background: var(--bg);
    display: flex;
    align-items: center;
    justify-content: center;
  }

  .trust-score {
    font-size: 32px;
    font-weight: 700;
  }

  .trust-label {
    font-family: var(--mono);
    font-size: 12px;
    color: var(--muted);
    text-align: center;
    margin-top: 8px;
  }

  /* WHOIS CARD */
  .whois-table {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
  }

  .whois-item {
    display: flex;
    justify-content: space-between;
    padding: 8px 0;
    border-bottom: 1px solid var(--border);
  }

  .whois-label {
    font-family: var(--mono);
    font-size: 12px;
    color: var(--muted);
  }

  .whois-value {
    font-size: 12px;
    text-align: right;
  }

  /* SHAP CARD */
  .shap-list {
    display: flex;
    flex-direction: column;
    gap: 12px;
  }

  .shap-item {
    display: flex;
    align-items: center;
    gap: 12px;
  }

  .shap-bar {
    flex: 1;
    height: 8px;
    background: var(--border);
    border-radius: 4px;
    overflow: hidden;
  }

  .shap-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.5s;
  }

  .shap-fill.positive { background: var(--danger); }
  .shap-fill.negative { background: var(--accent); }

  .shap-text {
    font-family: var(--mono);
    font-size: 11px;
    min-width: 120px;
  }

  /* FEATURES CARD */
  .features-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
  }

  .feature-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 12px;
    border-radius: 8px;
    background: var(--bg);
    border: 1px solid var(--border);
  }

  .feature-name {
    font-family: var(--mono);
    font-size: 11px;
  }

  .feature-risk {
    font-size: 12px;
    font-weight: 600;
  }

  .risk-low { color: var(--accent); }
  .risk-medium { color: var(--warn); }
  .risk-high { color: var(--danger); }

  /* ANALYSIS PANEL */
  .analysis-panel {
    grid-column: span 2;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 24px;
  }

  .analysis-steps {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .analysis-step {
    display: flex;
    align-items: center;
    gap: 12px;
    font-family: var(--mono);
    font-size: 12px;
    color: var(--muted);
    opacity: 0;
    animation: fadeIn 0.3s forwards;
  }

  .analysis-step.completed {
    color: var(--accent);
  }

  .step-icon {
    width: 16px;
    height: 16px;
    border-radius: 50%;
    background: var(--border);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 10px;
  }

  .step-icon.completed {
    background: var(--accent);
    color: var(--bg);
  }

  @keyframes fadeIn {
    to { opacity: 1; }
  }

  /* FEEDBACK CARD */
  .feedback-buttons {
    display: flex;
    gap: 12px;
    margin-top: 16px;
  }

  .feedback-btn {
    flex: 1;
    padding: 12px;
    border: 1px solid var(--border);
    border-radius: 8px;
    background: var(--bg);
    color: var(--text);
    font-family: var(--sans);
    font-size: 14px;
    cursor: pointer;
    transition: all 0.2s;
  }

  .feedback-btn:hover {
    border-color: var(--accent-mid);
  }

  .feedback-btn.safe:hover {
    border-color: var(--accent);
    box-shadow: 0 0 20px var(--accent-dim);
  }

  .feedback-btn.phishing:hover {
    border-color: var(--danger);
    box-shadow: 0 0 20px var(--danger-dim);
  }

  .feedback-message {
    margin-top: 12px;
    padding: 8px 12px;
    border-radius: 6px;
    font-family: var(--mono);
    font-size: 12px;
    text-align: center;
  }

  .feedback-message.success {
    background: var(--accent-dim);
    color: var(--accent);
  }

  /* ANIMATIONS */
  .card-enter {
    animation: cardEnter 0.6s ease forwards;
  }

  @keyframes cardEnter {
    from {
      opacity: 0;
      transform: translateY(20px);
    }
    to {
      opacity: 1;
      transform: translateY(0);
    }
  }

  /* RESPONSIVE */
  @media (max-width: 768px) {
    nav { padding: 16px 20px; }
    .nav-left { flex-direction: row; align-items: center; gap: 12px; }
    .tagline { margin-top: 0; }
    .hero { padding: 60px 20px 40px; }
    .dashboard { grid-template-columns: 1fr; padding: 20px 12px; }
    .status-card { grid-column: span 1; }
    .analysis-panel { grid-column: span 1; }
    .input-row { flex-direction: column; }
    .scan-btn { width: 100%; }
    .whois-table { grid-template-columns: 1fr; }
    .features-grid { grid-template-columns: 1fr; }
  }
`;

const ANALYSIS_STEPS = [
  "Checking WHOIS...",
  "Extracting URL features...",
  "Running ML model...",
  "Generating SHAP explanations...",
  "Calculating trust score...",
];

export default function App() {
  const [url, setUrl] = useState("");
  const [scanning, setScanning] = useState(false);
  const [result, setResult] = useState(null);
  const [currentStep, setCurrentStep] = useState(0);
  const [feedback, setFeedback] = useState(null);

  const handleScan = async () => {
    if (!url.trim()) return;
    setResult(null);
    setScanning(true);
    setCurrentStep(0);
    setFeedback(null);

    // Simulate step-by-step analysis
    const stepInterval = setInterval(() => {
      setCurrentStep(prev => {
        if (prev < ANALYSIS_STEPS.length - 1) {
          return prev + 1;
        } else {
          clearInterval(stepInterval);
          return prev;
        }
      });
    }, 500);

    try {
      const response = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url }),
      });
      const data = await response.json();
      setResult(data);
    } catch (error) {
      setResult({
        prediction: {
          label: "Error",
          is_phishing: true,
          confidence: 0,
          trust_score: 0,
          warning: "Could not reach backend",
        },
        whois: {},
        explanation: { local_features: [] },
        features: {},
      });
    } finally {
      setScanning(false);
      clearInterval(stepInterval);
    }
  };

  const handleKey = (e) => {
    if (e.key === "Enter") handleScan();
  };

  const handleFeedback = async (isCorrect) => {
    if (!result) return;
    try {
      await fetch("http://127.0.0.1:8000/feedback", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          url,
          predicted_phishing: result.prediction.is_phishing,
          actual_phishing: !isCorrect,
          confidence: result.prediction.confidence,
        }),
      });
      setFeedback("Feedback recorded for adaptive retraining.");
    } catch (error) {
      setFeedback("Failed to submit feedback.");
    }
  };

  const getStatusInfo = () => {
  if (!result) return { label: "", color: "", icon: "" };
  const { is_phishing, confidence } = result.prediction;

  if (!is_phishing) {
    return { label: "SAFE", color: "safe", icon: "🛡️" };
  } else if (confidence < 70) {
    return { label: "SUSPICIOUS", color: "suspicious", icon: "⚡" };
  } else {
    return { label: "PHISHING", color: "phishing", icon: "⚠️" };
  }
};

  const getRiskLevel = (score) => {
    if (score >= 80) return { label: "Safe", color: "low" };
    if (score >= 50) return { label: "Medium Risk", color: "medium" };
    return { label: "High Risk", color: "high" };
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
          <div className="nav-left">
            <div className="logo">PhishGuard AI</div>
            <div className="tagline">Real-Time AI Powered Phishing Detection</div>
          </div>
        </nav>

        {/* HERO */}
        {!result && (
          <main className="hero">
            <h1 className="hero-title">
              Detect <em>Phishing</em> Threats
            </h1>
            <p className="hero-subtitle">
              Advanced AI analysis with real-time WHOIS, SHAP explanations, and multi-model predictions.
            </p>

            {/* INPUT */}
            <div className="input-card">
              <div className="input-row">
                <div style={{ position: 'relative', flex: 1 }}>
                  <input
                    className="url-input"
                    type="text"
                    placeholder="Enter website URL to analyze"
                    value={url}
                    onChange={(e) => setUrl(e.target.value)}
                    onKeyDown={handleKey}
                  />
                  <span className="input-icon">🔍</span>
                </div>
                <button
                  className="scan-btn"
                  onClick={handleScan}
                  disabled={scanning || !url.trim()}
                >
                  {scanning ? "Analyzing..." : "Analyze Website"}
                </button>
              </div>

              {/* LIVE ANALYSIS PANEL */}
              {scanning && (
                <div className="analysis-panel">
                  <div className="card-title">
                    <span className="card-icon">⚙️</span>
                    Live Analysis
                  </div>
                  <div className="analysis-steps">
                    {ANALYSIS_STEPS.map((step, index) => (
                      <div
                        key={index}
                        className={`analysis-step ${index <= currentStep ? 'completed' : ''}`}
                        style={{ animationDelay: `${index * 0.1}s` }}
                      >
                        <div className={`step-icon ${index <= currentStep ? 'completed' : ''}`}>
                          {index <= currentStep ? '✓' : '○'}
                        </div>
                        {step}
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </main>
        )}

        {/* DASHBOARD */}
        {result && (
          <main className="dashboard">
            {/* STATUS CARD */}
            <div className="card status-card card-enter">
              <div className="card-title">
                <span className="card-icon">{getStatusInfo().icon}</span>
                Detection Status
              </div>
              <div className="status-display">
                <div className={`status-icon status-${getStatusInfo().color}`}>
                  {getStatusInfo().icon}
                </div>
                <div className={`status-label status-${getStatusInfo().color}`}>
                  {getStatusInfo().label}
                </div>
                <div className="confidence">
                  {result.prediction.confidence}% Confidence
                </div>
              </div>
            </div>

            {/* TRUST SCORE CARD */}
            <div className="card trust-card card-enter" style={{ animationDelay: '0.1s' }}>
              <div className="card-title">
                <span className="card-icon">📊</span>
                Trust Score
              </div>
              <div className="trust-meter">
                <div className="trust-circle-bg" style={{
                  background: `conic-gradient(from 0deg, var(--accent) ${result.prediction.trust_score * 3.6}deg, var(--border) ${result.prediction.trust_score * 3.6}deg)`
                }}>
                  <div className="trust-circle-fill">
                    <div className="trust-score">{result.prediction.trust_score}</div>
                  </div>
                </div>
              </div>
              <div className="trust-label">
                {getRiskLevel(result.prediction.trust_score).label}
              </div>
            </div>

            {/* WHOIS CARD */}
            <div className="card card-enter" style={{ animationDelay: '0.2s' }}>
              <div className="card-title">
                <span className="card-icon">🌐</span>
                WHOIS Domain Analysis
              </div>
              <div className="whois-table">
                <div className="whois-item">
                  <span className="whois-label">Domain Age</span>
                  <span className="whois-value">{result.whois?.domain_age_days || 'N/A'} days</span>
                </div>
                <div className="whois-item">
                  <span className="whois-label">Registrar</span>
                  <span className="whois-value">{result.whois?.registrar || 'N/A'}</span>
                </div>
                <div className="whois-item">
                  <span className="whois-label">Expiry Date</span>
                  <span className="whois-value">{result.whois?.expiry_date || 'N/A'}</span>
                </div>
                <div className="whois-item">
                  <span className="whois-label">Hosting Country</span>
                  <span className="whois-value">{result.whois?.country || 'N/A'}</span>
                </div>
                <div className="whois-item">
                  <span className="whois-label">SSL Status</span>
                  <span className="whois-value">Valid</span>
                </div>
              </div>
            </div>
           {/* SHAP CARD */}
            <div className="card card-enter" style={{ animationDelay: '0.3s' }}>
            <div className="card-title">
            <span className="card-icon">🤖</span>
            AI Explainability
            </div>

            {/* use global_features; fall back to local_features if absent */}
            {(() => {
            const features = result.explanation?.global_features?.length
              ? result.explanation.global_features
              : result.explanation?.local_features || [];

            if (!features.length) {
              return <p style={{ fontFamily: 'var(--mono)', fontSize: '12px', color: 'var(--muted)' }}>No explanation data available.</p>;
            }

            const maxAbs = Math.max(...features.map(f => Math.abs(f.shap_value)), 0.001);
            const isPhishing = result.prediction.is_phishing;

            return (
              <>
                <p style={{ fontFamily: 'var(--mono)', fontSize: '11px', color: 'var(--muted)', marginBottom: '16px', lineHeight: '1.6' }}>
                  Features ranked by influence on this prediction. Red bars pushed toward phishing, green toward safe.
                </p>
                <div className="shap-list">
                  {features.map((feature, index) => {
                    const pct = (Math.abs(feature.shap_value) / maxAbs) * 100;
                    // For global features: high shap_value = important for phishing detection
                    // Color depends on whether this feature is a risk signal or a safety signal
                    const isRisk = isPhishing
                      ? feature.shap_value > 0
                      : feature.shap_value < 0;

                    return (
                      <div key={index} style={{ marginBottom: '10px' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                          <span style={{ fontFamily: 'var(--mono)', fontSize: '11px', color: 'var(--text)' }}>
                            {feature.label}
                          </span>
                          <span style={{
                            fontFamily: 'var(--mono)',
                            fontSize: '10px',
                            color: isRisk ? 'var(--danger)' : 'var(--accent)',
                            fontWeight: '700'
                          }}>
                            {isRisk ? '▲ Risk' : '✓ Safe'}
                          </span>
                        </div>
                        <div style={{
                          height: '8px',
                          background: 'var(--border)',
                          borderRadius: '4px',
                          overflow: 'hidden'
                        }}>
                          <div style={{
                            width: `${pct}%`,
                            height: '100%',
                            borderRadius: '4px',
                            background: isRisk ? 'var(--danger)' : 'var(--accent)',
                            transition: 'width 0.6s ease',
                          }} />
                        </div>
                        <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '3px' }}>
                          <span style={{ fontFamily: 'var(--mono)', fontSize: '10px', color: 'var(--muted)' }}>
                            importance: {(Math.abs(feature.shap_value) * 100).toFixed(1)}%
                          </span>
                        </div>
                      </div>
                    );
                  })}
                </div>
                <div style={{
                  marginTop: '16px',
                  padding: '10px 12px',
                  borderRadius: '8px',
                  background: 'var(--bg)',
                  border: '1px solid var(--border)',
                  fontFamily: 'var(--mono)',
                  fontSize: '11px',
                  color: 'var(--muted)',
                  lineHeight: '1.6'
                }}>
                  Base rate: {(result.explanation.base_value * 100).toFixed(0)}% phishing probability before features applied.
                  Model: {result.explanation.model_used}
                </div>
              </>
            );
            })()}
            </div>

            {/* FEATURES CARD */}
            <div className="card card-enter" style={{ animationDelay: '0.4s' }}>
              <div className="card-title">
                <span className="card-icon">📋</span>
                Feature Risk Breakdown
              </div>
              <div className="features-grid">
                {Object.entries(result.features || {}).slice(0, 8).map(([key, value]) => (
                  <div key={key} className="feature-item">
                    <span className="feature-name">{key.replace(/_/g, ' ')}</span>
                    <span className={`feature-risk risk-${value > 0.5 ? 'high' : value > 0.3 ? 'medium' : 'low'}`}>
                      {value > 0.5 ? 'High' : value > 0.3 ? 'Med' : 'Low'}
                    </span>
                  </div>
                ))}
              </div>
            </div>
            {/* FEEDBACK CARD */}
            <div className="card card-enter" style={{ animationDelay: '0.5s', justifySelf: 'center', width: '100%', maxWidth: '400px' }}>
              <div className="card-title">
                <span className="card-icon">💬</span>
                User Feedback
              </div>
              <div className="feedback-buttons">
                <button
                  className="feedback-btn safe"
                  onClick={() => handleFeedback(true)}
                >
                  Mark as Safe
                </button>
                <button
                  className="feedback-btn phishing"
                  onClick={() => handleFeedback(false)}
                >
                  Report Phishing
                </button>
              </div>
              {feedback && (
                <div className="feedback-message success">
                  {feedback}
                </div>
              )}
            </div>
            <div></div>
            <button
              onClick={() => { setResult(null); setUrl(""); setFeedback(null); }}
              style={{
                background: 'var(--bg)',
                border: '1px solid var(--border)',
                borderRadius: '8px',
                padding: '6px 14px',
                color: 'var(--accent)',
                fontFamily: 'var(--mono)',
                fontSize: '21px',
                cursor: 'pointer',
                // display: 'flex',
                alignItems: 'center',
                gap: '6px',
                transition: 'all 0.2s',
              }}
              onMouseEnter={e => { e.currentTarget.style.borderColor = 'var(--accent-mid)'; e.currentTarget.style.color = 'var(--accent)'; }}
              onMouseLeave={e => { e.currentTarget.style.borderColor = 'var(--border)'; e.currentTarget.style.color = 'var(--accent)'; }}
            >
              ← New Scan
            </button>
          </main>
        )}
      </div>
    </>
  );
};                  
