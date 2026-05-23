import { useState } from "react";

const ANALYSIS_STEPS = [
  "Checking WHOIS...",
  "Extracting URL features...",
  "Running ML model...",
  "Generating SHAP explanations...",
  "Calculating trust score...",
];

// ─── Plain-English SHAP labels ────────────────────────────────────────────────
const SHAP_LABELS = {
  ssl_certificate: {
    safe: "Has a valid SSL certificate.",
    risk: "No SSL certificate detected.",
  },
  suspicious_anchor: {
    safe: "Links inside the page look normal.",
    risk: "Links inside the page are suspicious.",
  },
  url_length: { safe: "URL length is normal.", risk: "URL is unusually long." },
  having_ip_address: {
    safe: "Uses a real domain name.",
    risk: "Uses a raw IP instead of a domain.",
  },
  shortening_service: {
    safe: "No link shortener used.",
    risk: "URL was shortened — origin is hidden.",
  },
  having_at_symbol: {
    safe: "No '@' symbol in the URL.",
    risk: "Has '@' in URL — a known phishing trick.",
  },
  double_slash: {
    safe: "URL structure looks normal.",
    risk: "Has '//' redirect in the URL.",
  },
  prefix_suffix: {
    safe: "Domain has no suspicious dashes.",
    risk: "Domain uses dashes — common in fakes.",
  },
  sub_domain: {
    safe: "Domain depth is normal.",
    risk: "Too many subdomains — looks suspicious.",
  },
  domain_age: {
    safe: "Domain has been around a while.",
    risk: "Domain is very new — under 6 months.",
  },
  favicon: {
    safe: "Favicon loads from the same domain.",
    risk: "Favicon loads from a different domain.",
  },
  request_url: {
    safe: "Page content loads locally.",
    risk: "Page loads content from other sites.",
  },
  abnormal_url: {
    safe: "URL matches WHOIS records.",
    risk: "URL doesn't match WHOIS records.",
  },
};

// ─── Model metadata ───────────────────────────────────────────────────────────
// Keys match the `name` field returned by the backend in model_comparison[]
const MODEL_INFO = {
  "Logistic Regression": {
    icon: "📈",
    desc: "Fast, linear classifier. Great baseline.",
  },
  SVM: {
    icon: "🔷",
    desc: "Margin-based classifier. Strong on high-dimensional data.",
  },
  KNN: {
    icon: "🟠",
    desc: "Instance-based learner. Classifies by nearest neighbors.",
  },
  "Random Forest": {
    icon: "🌲",
    desc: "Ensemble of decision trees. Handles noisy features well.",
  },
  XGBoost: {
    icon: "⚡",
    desc: "Boosted trees. Highest accuracy on structured data.",
  },
};

// ─── Helpers ──────────────────────────────────────────────────────────────────
function whoisStatusColor(key, value) {
  if (!value || value === "N/A") return { color: "var(--muted)", icon: "—" };
  if (key === "domain_age_days") {
    const days = parseInt(value);
    if (days > 365) return { color: "var(--accent)", icon: "✓" };
    if (days > 90) return { color: "var(--warn)", icon: "!" };
    return { color: "var(--danger)", icon: "✗" };
  }
  if (key === "days_until_expiry") {
    const days = parseInt(value);
    if (days > 180) return { color: "var(--accent)", icon: "✓" };
    if (days > 0) return { color: "var(--warn)", icon: "!" };
    return { color: "var(--danger)", icon: "✗" };
  }
  if (key === "privacy_protected") {
    return value === true
      ? { color: "var(--warn)", icon: "!" }
      : { color: "var(--accent)", icon: "✓" };
  }
  return { color: "var(--muted)", icon: "•" };
}

// ─── Component ────────────────────────────────────────────────────────────────
export default function ScanPage({ onNewScan }) {
  const [url, setUrl] = useState("");
  const [scanning, setScanning] = useState(false);
  const [result, setResult] = useState(null);
  const [currentStep, setCurrentStep] = useState(0);
  const [feedback, setFeedback] = useState(null);
  const [copied, setCopied] = useState(false);

  const resetScan = () => {
    setResult(null);
    setUrl("");
    setFeedback(null);
    if (onNewScan) onNewScan();
  };

  const handleCopy = () => {
    if (!result) return;
    const p = result.prediction;
    const summary = [
      `PhishGuard AI — Scan Summary`,
      `─────────────────────────────`,
      `URL      : ${url}`,
      `Verdict  : ${p.is_phishing ? "⚠ Phishing" : "✓ Safe"}`,
      `Confidence: ${(p.confidence * 100).toFixed(1)}%`,
      `Trust Score: ${p.trust_score}/100`,
      `Model Used: ${p.model_used}`,
      `─────────────────────────────`,
      `Model Votes:`,
      ...(result.model_comparison || []).map(
        (m) => `  ${m.name}: ${m.label} (${Math.round(m.confidence * 100)}%)`,
      ),
    ].join("\n");
    navigator.clipboard.writeText(summary).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  };

  const handleScan = async () => {
    if (!url.trim()) return;

    // Validate URL format before hitting the backend
    try {
      const testUrl =
        url.startsWith("http://") || url.startsWith("https://")
          ? url
          : "http://" + url;
      new URL(testUrl);
    } catch {
      alert("Please provide a proper URL");
      return;
    }

    setResult(null);
    setScanning(true);
    setCurrentStep(0);
    setFeedback(null);

    const stepInterval = setInterval(() => {
      setCurrentStep((prev) => {
        if (prev < ANALYSIS_STEPS.length - 1) return prev + 1;
        clearInterval(stepInterval);
        return prev;
      });
    }, 500);

    try {
      const response = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url }),
      });

      // Handle backend validation errors (e.g. 422 Invalid URL)
      if (!response.ok) {
        alert("Please provide a proper URL");
        setScanning(false);
        clearInterval(stepInterval);
        return;
      }

      const data = await response.json();
      setResult(data);
    } catch {
      setResult({
        prediction: {
          label: "Error",
          is_phishing: true,
          confidence: 0,
          trust_score: 0,
          warning: "Could not reach backend.",
        },
        whois: { lookup_success: false, risk_flags: [] },
        explanation: { local_features: [], global_features: [] },
        features: {},
        model_comparison: [],
      });
    } finally {
      setScanning(false);
      clearInterval(stepInterval);
    }
  };

  const handleKey = (e) => {
    if (e.key === "Enter") handleScan();
  };

  // ── Feedback ──
  // Backend expects: { url, predicted_label, correct_label }
  const handleFeedback = async (correctLabel) => {
    if (!result) return;
    try {
      await fetch("http://127.0.0.1:8000/feedback", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          url,
          predicted_label: result.prediction.label, // "phishing" or "legitimate"
          correct_label: correctLabel, // what the user says it is
        }),
      });
      setFeedback("✓ Feedback recorded. Thanks for improving the model!");
    } catch {
      setFeedback("Failed to submit feedback.");
    }
  };

  const getStatusInfo = () => {
    if (!result) return { label: "", color: "", icon: "" };
    const { is_phishing, confidence } = result.prediction;
    if (!is_phishing) return { label: "SAFE", color: "safe", icon: "🛡️" };
    if (confidence < 0.7)
      return { label: "SUSPICIOUS", color: "suspicious", icon: "⚡" };
    return { label: "PHISHING", color: "phishing", icon: "⚠️" };
  };

  const getRiskLevel = (score) => {
    if (score >= 80) return "Safe";
    if (score >= 50) return "Medium Risk";
    return "High Risk";
  };

  // ─── RENDER ───────────────────────────────────────────────────────────────
  return (
    <>
      {/* ─── SCAN INPUT ─── */}
      {!result && (
        <main className="hero">
          <h1 className="hero-title">
            Detect <em>Phishing</em> Threats
          </h1>
          <p className="hero-subtitle">
            AI analysis with real-time WHOIS, SHAP explanations, and multi-model
            predictions.
          </p>

          <div className="input-card" style={{ maxWidth: "860px" }}>
            <div className="input-row">
              <div style={{ position: "relative", flex: 1 }}>
                <input
                  className="url-input"
                  type="text"
                  placeholder="Enter website URL to analyze 🔍"
                  value={url}
                  onChange={(e) => setUrl(e.target.value)}
                  onKeyDown={handleKey}
                />
              </div>
              <button
                className="nav-link cta"
                onClick={handleScan}
                disabled={scanning || !url.trim()}
              >
                {scanning ? "Analyzing..." : "Analyze Website"}
              </button>
            </div>

            {scanning && (
              <div
                style={{
                  marginTop: "20px",
                  paddingTop: "20px",
                  borderTop: "1px solid var(--border)",
                }}
              >
                <div className="card-title">
                  <span className="card-icon">⚙️</span>
                  Live Analysis
                </div>
                <div className="analysis-steps">
                  {ANALYSIS_STEPS.map((step, i) => (
                    <div
                      key={i}
                      className={`analysis-step ${i <= currentStep ? "completed" : ""}`}
                      style={{ animationDelay: `${i * 0.1}s` }}
                    >
                      <div
                        className={`step-icon ${i <= currentStep ? "completed" : ""}`}
                      >
                        {i <= currentStep ? "✓" : "○"}
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

      {/* ─── RESULTS DASHBOARD ─── */}
      {result && (
        <main className="dashboard">
          {/* STATUS CARD */}
          <div className="card status-card card-enter">
            <div className="card-title">
              <span className="card-icon">{getStatusInfo().icon}</span>
              Detection Status
            </div>
            {/* Warning banner — only shown when phishing */}
            {result.prediction.warning && (
              <div
                style={{
                  background: "var(--danger-dim)",
                  border: "1px solid var(--danger)",
                  borderRadius: "8px",
                  padding: "10px 14px",
                  marginBottom: "16px",
                  fontFamily: "var(--mono)",
                  fontSize: "11px",
                  color: "var(--danger)",
                  lineHeight: "1.6",
                }}
              >
                ⚠ {result.prediction.warning}
              </div>
            )}
            <div className="status-display">
              <div className={`status-icon status-${getStatusInfo().color}`}>
                {getStatusInfo().icon}
              </div>
              <div className={`status-label status-${getStatusInfo().color}`}>
                {getStatusInfo().label}
              </div>
              <div className="confidence">
                {(result.prediction.confidence * 100).toFixed(1)}% Confidence
              </div>
              <div
                style={{
                  fontFamily: "var(--mono)",
                  fontSize: "10px",
                  color: "var(--muted)",
                  marginTop: "6px",
                }}
              >
                Decided by: {result.prediction.model_used}
              </div>
            </div>
          </div>

          {/* TRUST SCORE CARD */}
          <div className="card card-enter" style={{ animationDelay: "0.1s" }}>
            <div className="card-title">
              <span className="card-icon">📊</span>
              Trust Score
            </div>
            <p
              style={{
                fontFamily: "var(--mono)",
                fontSize: "11px",
                color: "var(--muted)",
                marginBottom: "16px",
              }}
            >
              How trustworthy this URL is. High phishing confidence = low trust
              score.
            </p>
            <div className="trust-meter">
              <div
                className="trust-circle-bg"
                style={{
                  background: `conic-gradient(from 0deg, ${result.prediction.trust_score >= 80 ? "var(--accent)" : result.prediction.trust_score >= 50 ? "var(--warn)" : "var(--danger)"} ${result.prediction.trust_score * 3.6}deg, var(--border) ${result.prediction.trust_score * 3.6}deg)`,
                }}
              >
                <div className="trust-circle-fill">
                  <div className="trust-score">
                    {result.prediction.trust_score}
                  </div>
                </div>
              </div>
            </div>
            <div className="trust-label">
              {getRiskLevel(result.prediction.trust_score)}
            </div>
          </div>

          {/* ── WHOIS Analysis ── */}
          <div className="card card-enter" style={{ animationDelay: "0.2s" }}>
            <div className="card-title">
              <span className="card-icon">🌐</span>
              WHOIS Analysis
            </div>
            <p
              style={{
                fontFamily: "var(--mono)",
                fontSize: "11px",
                color: "var(--muted)",
                marginBottom: "16px",
              }}
            >
              Key facts about this domain's registration.
            </p>

            {[
              {
                key: "domain_age_days",
                label: "Domain Age",
                value:
                  result.whois?.domain_age_days != null
                    ? `${result.whois.domain_age_days} days`
                    : null,
                hint:
                  result.whois?.domain_age_days > 365
                    ? "Old domain. Trustworthy sites are usually well-established."
                    : result.whois?.domain_age_days >= 0
                      ? "Very new domain. Phishing sites are often freshly registered."
                      : "Age unknown.",
              },
              {
                key: "days_until_expiry",
                label: "Expires In",
                value:
                  result.whois?.days_until_expiry != null
                    ? `${result.whois.days_until_expiry} days`
                    : null,
                hint:
                  result.whois?.days_until_expiry > 180
                    ? "Long registration. Legitimate sites plan ahead."
                    : "Short expiry. Scam sites rarely pay for long-term registration.",
              },
              {
                key: "privacy_protected",
                label: "Owner Identity",
                value:
                  result.whois?.privacy_protected != null
                    ? result.whois.privacy_protected
                      ? "Hidden"
                      : "Visible"
                    : null,
                hint: result.whois?.privacy_protected
                  ? "Owner is masked. Legitimate sites usually don't hide their identity."
                  : "Owner is publicly visible. That's a good sign.",
              },
              {
                key: "registrar",
                label: "Registrar",
                value: result.whois?.registrar || null,
                hint: "The company this domain was registered through.",
              },
              {
                key: "country",
                label: "Hosted In",
                value: result.whois?.country || null,
                hint: "The country where this domain's servers are located.",
              },
            ].map(({ key, label, value, hint }) => {
              const rawVal =
                key === "privacy_protected"
                  ? result.whois?.privacy_protected
                  : result.whois?.[key];
              const { color, icon } = whoisStatusColor(key, rawVal ?? value);
              return (
                <div
                  key={key}
                  style={{
                    display: "flex",
                    alignItems: "flex-start",
                    gap: "12px",
                    padding: "12px 0",
                    borderBottom: "1px solid var(--border)",
                  }}
                >
                  <span
                    style={{
                      marginTop: "2px",
                      fontWeight: "700",
                      fontSize: "13px",
                      color,
                      minWidth: "16px",
                      textAlign: "center",
                    }}
                  >
                    {icon}
                  </span>
                  <div>
                    <div
                      style={{
                        fontFamily: "var(--mono)",
                        fontSize: "11px",
                        color: "var(--muted)",
                        marginBottom: "2px",
                      }}
                    >
                      {label}
                    </div>
                    <div
                      style={{
                        fontSize: "13px",
                        fontWeight: "600",
                        color: value ? "var(--text)" : "var(--muted)",
                      }}
                    >
                      {value || "Not available"}
                    </div>
                    {hint && (
                      <div
                        style={{
                          fontFamily: "var(--mono)",
                          fontSize: "10px",
                          color,
                          marginTop: "3px",
                          lineHeight: "1.5",
                        }}
                      >
                        {hint}
                      </div>
                    )}
                  </div>
                </div>
              );
            })}
          </div>

          {/* ── SHAP Explanation ── */}
          <div className="card card-enter" style={{ animationDelay: "0.3s" }}>
            <div className="card-title">
              <span className="card-icon">🤖</span>
              SHAP Explanation
            </div>
            <p
              style={{
                fontFamily: "var(--mono)",
                fontSize: "11px",
                color: "var(--muted)",
                marginBottom: "16px",
              }}
            >
              Top 5 features that influenced this prediction. Bar length = how
              much it mattered.
            </p>

            {(() => {
              const features = result.explanation?.local_features?.length
                ? result.explanation.local_features
                : result.explanation?.global_features || [];

              if (!features.length) {
                return (
                  <p
                    style={{
                      fontFamily: "var(--mono)",
                      fontSize: "12px",
                      color: "var(--muted)",
                    }}
                  >
                    No explanation data available.
                  </p>
                );
              }

              // Pick top 5 by absolute SHAP value
              const top5 = [...features]
                .sort((a, b) => Math.abs(b.shap_value) - Math.abs(a.shap_value))
                .slice(0, 5);

              const maxAbs = Math.abs(top5[0].shap_value) || 0.001;

              return (
                <div
                  style={{
                    display: "flex",
                    flexDirection: "column",
                    gap: "14px",
                  }}
                >
                  {top5.map((feature, i) => {
                    const isRisk = feature.direction === "risk";
                    const pct = (Math.abs(feature.shap_value) / maxAbs) * 100;
                    const key = feature.label
                      ?.toLowerCase()
                      .replace(/\s+/g, "_");
                    const lookup = SHAP_LABELS[key];
                    const sentence = lookup
                      ? isRisk
                        ? lookup.risk
                        : lookup.safe
                      : feature.label;
                    const color = isRisk ? "var(--danger)" : "var(--accent)";
                    const tag = isRisk ? "pushed phishing" : "pushed safe";

                    return (
                      <div key={i}>
                        {/* Feature name + verdict tag */}
                        <div
                          style={{
                            display: "flex",
                            justifyContent: "space-between",
                            alignItems: "center",
                            marginBottom: "4px",
                          }}
                        >
                          <span
                            style={{
                              fontFamily: "var(--mono)",
                              fontSize: "11px",
                              color: "var(--muted)",
                            }}
                          >
                            {feature.label}
                          </span>
                          <span
                            style={{
                              fontFamily: "var(--mono)",
                              fontSize: "10px",
                              fontWeight: "700",
                              color,
                            }}
                          >
                            {tag}
                          </span>
                        </div>
                        {/* One-line plain English explanation */}
                        <div
                          style={{
                            fontSize: "12px",
                            color: "var(--text)",
                            marginBottom: "6px",
                            lineHeight: "1.4",
                          }}
                        >
                          {sentence}
                        </div>
                        {/* Contribution bar */}
                        <div
                          style={{
                            height: "5px",
                            background: "var(--border)",
                            borderRadius: "4px",
                            overflow: "hidden",
                          }}
                        >
                          <div
                            style={{
                              width: `${pct}%`,
                              height: "100%",
                              borderRadius: "4px",
                              background: color,
                              transition: "width 0.6s ease",
                            }}
                          />
                        </div>
                      </div>
                    );
                  })}
                </div>
              );
            })()}
          </div>

          {/* ── MODEL VOTES: uses model_comparison[] from backend ── */}
          <div className="card card-enter" style={{ animationDelay: "0.4s" }}>
            <div className="card-title">
              <span className="card-icon">🧠</span>
              Model Votes
            </div>
            <p
              style={{
                fontFamily: "var(--mono)",
                fontSize: "11px",
                color: "var(--muted)",
                marginBottom: "20px",
              }}
            >
              Five models voted independently. Here's what each one said.
            </p>

            <div
              style={{ display: "flex", flexDirection: "column", gap: "18px" }}
            >
              {[...(result.model_comparison || [])]
                .sort((a, b) => {
                  const ORDER = [
                    "XGBoost",
                    "Random Forest",
                    "Logistic Regression",
                    "SVM",
                    "KNN",
                  ];
                  const aIsBest = a.name === result.prediction.model_used;
                  const bIsBest = b.name === result.prediction.model_used;
                  if (aIsBest) return -1;
                  if (bIsBest) return 1;
                  return ORDER.indexOf(a.name) - ORDER.indexOf(b.name);
                })
                .map((model) => {
                  const info = MODEL_INFO[model.name] || {
                    icon: "🔬",
                    desc: "",
                  };
                  const pct = Math.round(model.confidence * 100);
                  const isSafe = model.label === "legitimate";
                  const barColor = isSafe
                    ? "var(--accent)"
                    : model.confidence < 0.7
                      ? "var(--warn)"
                      : "var(--danger)";
                  const verdict = isSafe
                    ? "Safe"
                    : model.confidence < 0.7
                      ? "Suspicious"
                      : "Phishing";
                  const verdictColor = isSafe
                    ? "var(--accent)"
                    : model.confidence < 0.7
                      ? "var(--warn)"
                      : "var(--danger)";
                  const isBest = model.name === result.prediction.model_used;

                  return (
                    <div key={model.name}>
                      <div
                        style={{
                          display: "flex",
                          justifyContent: "space-between",
                          alignItems: "center",
                          marginBottom: "4px",
                        }}
                      >
                        <div
                          style={{
                            display: "flex",
                            alignItems: "center",
                            gap: "8px",
                          }}
                        >
                          <span style={{ fontSize: "16px" }}>{info.icon}</span>
                          <span style={{ fontWeight: "700", fontSize: "13px" }}>
                            {model.name}
                          </span>
                          {/* Badge marks which model made the final call */}
                          {isBest && (
                            <span
                              style={{
                                fontFamily: "var(--mono)",
                                fontSize: "9px",
                                color: "var(--bg)",
                                background: "var(--accent)",
                                borderRadius: "4px",
                                padding: "2px 6px",
                                letterSpacing: "1px",
                                textTransform: "uppercase",
                              }}
                            >
                              Best
                            </span>
                          )}
                        </div>
                        <span
                          style={{
                            fontFamily: "var(--mono)",
                            fontSize: "11px",
                            fontWeight: "700",
                            color: verdictColor,
                          }}
                        >
                          {verdict}
                        </span>
                      </div>
                      <p
                        style={{
                          fontFamily: "var(--mono)",
                          fontSize: "10px",
                          color: "var(--muted)",
                          marginBottom: "8px",
                        }}
                      >
                        {info.desc}
                      </p>
                      <div
                        style={{
                          display: "flex",
                          alignItems: "center",
                          gap: "10px",
                        }}
                      >
                        <div
                          style={{
                            flex: 1,
                            height: "6px",
                            background: "var(--border)",
                            borderRadius: "4px",
                            overflow: "hidden",
                          }}
                        >
                          <div
                            style={{
                              width: `${pct}%`,
                              height: "100%",
                              borderRadius: "4px",
                              background: barColor,
                              transition: "width 0.8s ease",
                            }}
                          />
                        </div>
                        <span
                          style={{
                            fontFamily: "var(--mono)",
                            fontSize: "11px",
                            color: "var(--muted)",
                            minWidth: "36px",
                            textAlign: "right",
                          }}
                        >
                          {pct}%
                        </span>
                      </div>
                    </div>
                  );
                })}

              {(!result.model_comparison ||
                result.model_comparison.length === 0) && (
                <p
                  style={{
                    fontFamily: "var(--mono)",
                    fontSize: "12px",
                    color: "var(--muted)",
                  }}
                >
                  Model breakdown not available.
                </p>
              )}
            </div>
          </div>

          {/* FEEDBACK CARD */}
          <div
            className="card card-enter"
            style={{
              animationDelay: "0.5s",
              justifySelf: "center",
              width: "100%",
              maxWidth: "400px",
            }}
          >
            <div className="card-title">
              <span className="card-icon">💬</span>
              User Feedback
            </div>
            <p
              style={{
                fontFamily: "var(--mono)",
                fontSize: "11px",
                color: "var(--muted)",
                marginBottom: "12px",
                lineHeight: "1.6",
              }}
            >
              Result looks wrong? Tell us — it helps the model improve.
            </p>
            <div className="feedback-buttons">
              {result.prediction.is_phishing ? (
                <button
                  className="feedback-btn safe"
                  onClick={() => handleFeedback("legitimate")}
                >
                  ✓ Mark as Safe
                </button>
              ) : (
                <button
                  className="feedback-btn phishing"
                  onClick={() => handleFeedback("phishing")}
                >
                  ⚠ Report as Phishing
                </button>
              )}
            </div>
            {feedback && (
              <div className="feedback-message success">{feedback}</div>
            )}
          </div>

          {/* SCAN SUMMARY + QUICK TIPS CARD */}
          <div className="card card-enter" style={{ animationDelay: "0.6s" }}>
            {/* ── Scan Summary ── */}
            <div
              className="card-title"
              style={{ justifyContent: "space-between" }}
            >
              <span
                style={{ display: "flex", alignItems: "center", gap: "8px" }}
              >
                <span className="card-icon">🔎</span>
                Scan Summary
              </span>
              <button
                onClick={handleCopy}
                style={{
                  fontFamily: "var(--mono)",
                  fontSize: "10px",
                  letterSpacing: "1px",
                  textTransform: "uppercase",
                  padding: "5px 12px",
                  borderRadius: "6px",
                  border: `1px solid ${copied ? "var(--accent)" : "var(--border)"}`,
                  background: copied ? "var(--accent-dim)" : "transparent",
                  color: copied ? "var(--accent)" : "var(--muted)",
                  cursor: "pointer",
                  transition: "all 0.2s",
                }}
              >
                {copied ? "✓ Copied!" : "⎘ Copy"}
              </button>
            </div>
            <div
              style={{
                background: "var(--bg)",
                border: "1px solid var(--border)",
                borderRadius: "10px",
                padding: "12px 16px",
                marginBottom: "20px",
              }}
            >
              <div
                style={{
                  fontFamily: "var(--mono)",
                  fontSize: "10px",
                  color: "var(--muted)",
                  marginBottom: "4px",
                  letterSpacing: "1px",
                  textTransform: "uppercase",
                }}
              >
                URL Scanned
              </div>
              <div
                style={{
                  fontSize: "12px",
                  color: "var(--text)",
                  wordBreak: "break-all",
                  lineHeight: "1.5",
                  marginBottom: "10px",
                }}
              >
                {url}
              </div>
              <div style={{ display: "flex", gap: "16px", flexWrap: "wrap" }}>
                <div>
                  <div
                    style={{
                      fontFamily: "var(--mono)",
                      fontSize: "10px",
                      color: "var(--muted)",
                      marginBottom: "2px",
                    }}
                  >
                    Verdict
                  </div>
                  <div
                    style={{
                      fontFamily: "var(--mono)",
                      fontSize: "11px",
                      fontWeight: "700",
                      color: result.prediction.is_phishing
                        ? "var(--danger)"
                        : "var(--accent)",
                    }}
                  >
                    {result.prediction.is_phishing ? "⚠ Phishing" : "✓ Safe"}
                  </div>
                </div>
                <div>
                  <div
                    style={{
                      fontFamily: "var(--mono)",
                      fontSize: "10px",
                      color: "var(--muted)",
                      marginBottom: "2px",
                    }}
                  >
                    Confidence
                  </div>
                  <div
                    style={{
                      fontFamily: "var(--mono)",
                      fontSize: "11px",
                      fontWeight: "700",
                      color: "var(--text)",
                    }}
                  >
                    {(result.prediction.confidence * 100).toFixed(1)}%
                  </div>
                </div>
                <div>
                  <div
                    style={{
                      fontFamily: "var(--mono)",
                      fontSize: "10px",
                      color: "var(--muted)",
                      marginBottom: "2px",
                    }}
                  >
                    Model Used
                  </div>
                  <div
                    style={{
                      fontFamily: "var(--mono)",
                      fontSize: "11px",
                      fontWeight: "700",
                      color: "var(--text)",
                    }}
                  >
                    {result.prediction.model_used}
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* NEW SCAN BUTTON */}
          <button onClick={resetScan} className="new-scan-btn">
            ✧ New Scan ✧
          </button>
        </main>
      )}
    </>
  );
}
