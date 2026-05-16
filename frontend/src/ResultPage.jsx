import React from "react";

export default function ResultPage({
  result,
  url,
  feedback,
  onFeedback,
  onNewScan,
}) {
  const getStatusInfo = () => {
    if (!result) return { label: "", color: "", icon: "" };
    const { is_phishing, confidence } = result.prediction;
    if (!is_phishing) return { label: "SAFE", color: "safe", icon: "🛡️" };
    else if (confidence < 70)
      return { label: "SUSPICIOUS", color: "suspicious", icon: "⚡" };
    else return { label: "PHISHING", color: "phishing", icon: "⚠️" };
  };

  const getRiskLevel = (score) => {
    if (score >= 80) return { label: "Safe", color: "low" };
    if (score >= 50) return { label: "Medium Risk", color: "medium" };
    return { label: "High Risk", color: "high" };
  };

  const status = getStatusInfo();

  return (
    <main className="dashboard">
      {/* STATUS CARD */}
      <div className="card status-card card-enter">
        <div className="card-title">
          <span className="card-icon">{status.icon}</span>
          Detection Status
        </div>
        <div className="status-display">
          <div className={`status-icon status-${status.color}`}>
            {status.icon}
          </div>
          <div className={`status-label status-${status.color}`}>
            {status.label}
          </div>
          <div className="confidence">
            {(result.prediction.confidence * 100).toFixed(1)}% Confidence
          </div>
        </div>
      </div>

      {/* TRUST SCORE CARD */}
      <div
        className="card trust-card card-enter"
        style={{ animationDelay: "0.1s" }}
      >
        <div className="card-title">
          <span className="card-icon">📊</span>
          Trust Score
        </div>
        <div className="trust-meter">
          <div
            className="trust-circle-bg"
            style={{
              background: `conic-gradient(from 0deg, var(--accent) ${result.prediction.trust_score * 3.6}deg, var(--border) ${result.prediction.trust_score * 3.6}deg)`,
            }}
          >
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
      <div className="card card-enter" style={{ animationDelay: "0.2s" }}>
        <div className="card-title">
          <span className="card-icon">🌐</span>
          WHOIS Domain Analysis
        </div>
        <div className="whois-table">
          <div className="whois-item">
            <span className="whois-label">Domain Age</span>
            <span className="whois-value">
              {result.whois?.domain_age_days || "N/A"} days
            </span>
          </div>
          <div className="whois-item">
            <span className="whois-label">Registrar</span>
            <span className="whois-value">
              {result.whois?.registrar || "N/A"}
            </span>
          </div>
          <div className="whois-item">
            <span className="whois-label">Expiry Date</span>
            <span className="whois-value">
              {result.whois?.expiry_date || "N/A"}
            </span>
          </div>
          <div className="whois-item">
            <span className="whois-label">Hosting Country</span>
            <span className="whois-value">
              {result.whois?.country || "N/A"}
            </span>
          </div>
          <div className="whois-item">
            <span className="whois-label">SSL Status</span>
            <span className="whois-value">Valid</span>
          </div>
        </div>
      </div>

      {/* SHAP CARD */}
      <div className="card card-enter" style={{ animationDelay: "0.3s" }}>
        <div className="card-title">
          <span className="card-icon">🤖</span>
          AI Explainability
        </div>
        {(() => {
          const features = result.explanation?.global_features?.length
            ? result.explanation.global_features
            : result.explanation?.local_features || [];

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

          const maxAbs = Math.max(
            ...features.map((f) => Math.abs(f.shap_value)),
            0.001,
          );
          const isPhishing = result.prediction.is_phishing;

          return (
            <>
              <p
                style={{
                  fontFamily: "var(--mono)",
                  fontSize: "11px",
                  color: "var(--muted)",
                  marginBottom: "16px",
                  lineHeight: "1.6",
                }}
              >
                Features ranked by influence on this prediction. Red bars pushed
                toward phishing, green toward safe.
              </p>
              <div className="shap-list">
                {features.map((feature, index) => {
                  const pct = (Math.abs(feature.shap_value) / maxAbs) * 100;
                  const isRisk = isPhishing
                    ? feature.shap_value > 0
                    : feature.shap_value < 0;
                  return (
                    <div key={index} style={{ marginBottom: "10px" }}>
                      <div
                        style={{
                          display: "flex",
                          justifyContent: "space-between",
                          marginBottom: "4px",
                        }}
                      >
                        <span
                          style={{
                            fontFamily: "var(--mono)",
                            fontSize: "11px",
                            color: "var(--text)",
                          }}
                        >
                          {feature.label}
                        </span>
                        <span
                          style={{
                            fontFamily: "var(--mono)",
                            fontSize: "10px",
                            color: isRisk ? "var(--danger)" : "var(--accent)",
                            fontWeight: "700",
                          }}
                        >
                          {isRisk ? "▲ Risk" : "✓ Safe"}
                        </span>
                      </div>
                      <div
                        style={{
                          height: "8px",
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
                            background: isRisk
                              ? "var(--danger)"
                              : "var(--accent)",
                            transition: "width 0.6s ease",
                          }}
                        />
                      </div>
                      <div
                        style={{
                          display: "flex",
                          justifyContent: "space-between",
                          marginTop: "3px",
                        }}
                      >
                        <span
                          style={{
                            fontFamily: "var(--mono)",
                            fontSize: "10px",
                            color: "var(--muted)",
                          }}
                        >
                          importance:{" "}
                          {(Math.abs(feature.shap_value) * 100).toFixed(1)}%
                        </span>
                      </div>
                    </div>
                  );
                })}
              </div>
              <div
                style={{
                  marginTop: "16px",
                  padding: "10px 12px",
                  borderRadius: "8px",
                  background: "var(--bg)",
                  border: "1px solid var(--border)",
                  fontFamily: "var(--mono)",
                  fontSize: "11px",
                  color: "var(--muted)",
                  lineHeight: "1.6",
                }}
              >
                Base rate: {(result.explanation.base_value * 100).toFixed(0)}%
                phishing probability before features applied. Model:{" "}
                {result.explanation.model_used}
              </div>
            </>
          );
        })()}
      </div>

      {/* FEATURES CARD */}
      <div className="card card-enter" style={{ animationDelay: "0.4s" }}>
        <div className="card-title">
          <span className="card-icon">📋</span>
          Feature Risk Breakdown
        </div>
        <div className="features-grid">
          {Object.entries(result.features || {})
            .slice(0, 8)
            .map(([key, value]) => (
              <div key={key} className="feature-item">
                <span className="feature-name">{key.replace(/_/g, " ")}</span>
                <span
                  className={`feature-risk risk-${
                    value > 0.5 ? "high" : value > 0.3 ? "medium" : "low"
                  }`}
                >
                  {value > 0.5 ? "High" : value > 0.3 ? "Med" : "Low"}
                </span>
              </div>
            ))}
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
          Is this result incorrect? Let us know to improve the model.
        </p>
        <div className="feedback-buttons">
          {result.prediction.is_phishing ? (
            <button
              className="feedback-btn safe"
              onClick={() => onFeedback("legitimate")}
            >
              ✓ Mark as Safe
            </button>
          ) : (
            <button
              className="feedback-btn phishing"
              onClick={() => onFeedback("phishing")}
            >
              ⚠ Report as Phishing
            </button>
          )}
        </div>
        {feedback && <div className="feedback-message success">{feedback}</div>}
      </div>

      <div></div>

      {/* NEW SCAN BUTTON */}
      <button
        onClick={onNewScan}
        style={{
          background: "linear-gradient(135deg, #00ff8818, #00ff8808)",
          border: "1px solid var(--accent-mid)",
          borderRadius: "12px",
          padding: "12px 28px",
          color: "var(--accent)",
          fontFamily: "var(--mono)",
          fontSize: "16px",
          fontWeight: "700",
          cursor: "pointer",
          alignItems: "center",
          gap: "8px",
          transition: "all 0.2s",
          boxShadow: "0 0 20px #00ff8820",
          letterSpacing: "1px",
        }}
        onMouseEnter={(e) => {
          e.currentTarget.style.background =
            "linear-gradient(135deg, #00ff8830, #00ff8815)";
          e.currentTarget.style.borderColor = "var(--accent)";
          e.currentTarget.style.boxShadow = "0 0 40px #00ff8840";
          e.currentTarget.style.transform = "scale(1.03)";
        }}
        onMouseLeave={(e) => {
          e.currentTarget.style.background =
            "linear-gradient(135deg, #00ff8818, #00ff8808)";
          e.currentTarget.style.borderColor = "var(--accent-mid)";
          e.currentTarget.style.boxShadow = "0 0 20px #00ff8820";
          e.currentTarget.style.transform = "scale(1)";
        }}
      >
        ← New Scan
      </button>
    </main>
  );
}
