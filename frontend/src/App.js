import { useState } from "react";
import "./App.css";
import ScanPage from "./pages/ScanPage";

// ─── Landing Page ─────────────────────────────────────────────────────────────
function LandingPage({ onNavigate }) {
  return (
    <div className="landing">
      <section className="landing-hero">
        <div className="hero-badge">
          <span className="badge-dot" />
          AI-Powered Protection
        </div>
        <h1 className="landing-title">
          Stop Phishing
          <br />
          Before It Starts
        </h1>
        <p className="landing-subtitle">
          PhishGuard AI uses explainable machine learning, real-time WHOIS
          lookups, and SHAP analysis to detect phishing threats with 98%+
          confidence — in seconds.
        </p>
        <div className="landing-cta-group">
          <button className="btn-primary" onClick={() => onNavigate("scan")}>
            Analyze a URL →
          </button>
          <button className="btn-secondary" onClick={() => onNavigate("about")}>
            How It Works
          </button>
        </div>
      </section>

      <div className="stats-strip">
        <div className="stat-item">
          <div className="stat-number">98.4%</div>
          <div className="stat-label">Detection Accuracy</div>
        </div>
        <div className="stat-item">
          <div className="stat-number">&lt;2s</div>
          <div className="stat-label">Analysis Time</div>
        </div>
        <div className="stat-item">
          <div className="stat-number">3+</div>
          <div className="stat-label">ML Models</div>
        </div>
        <div className="stat-item">
          <div className="stat-number">SHAP</div>
          <div className="stat-label">Explainability</div>
        </div>
      </div>

      <div className="section">
        <p className="section-label">The Process</p>
        <h2 className="section-title">How PhishGuard Works</h2>
        <div className="steps-grid">
          <div className="step-card">
            <div className="step-number">01 —</div>
            <div className="step-icon-lg">🔗</div>
            <div className="step-title">Submit a URL</div>
            <p className="step-desc">
              Paste any suspicious link. PhishGuard accepts full URLs, shortened
              links, or IP-based addresses.
            </p>
          </div>
          <div className="step-card">
            <div className="step-number">02 —</div>
            <div className="step-icon-lg">🌐</div>
            <div className="step-title">WHOIS Lookup</div>
            <p className="step-desc">
              Real-time domain intelligence — age, registrar, expiry date, and
              hosting country — is fetched live.
            </p>
          </div>
          <div className="step-card">
            <div className="step-number">03 —</div>
            <div className="step-icon-lg">🤖</div>
            <div className="step-title">ML Analysis</div>
            <p className="step-desc">
              Multiple models (Logistic Regression, Random Forest, XGBoost) vote
              on the URL's risk profile simultaneously.
            </p>
          </div>
          <div className="step-card">
            <div className="step-number">04 —</div>
            <div className="step-icon-lg">📊</div>
            <div className="step-title">SHAP Explanations</div>
            <p className="step-desc">
              Every prediction is explained. See exactly which URL features
              drove the decision, ranked by influence.
            </p>
          </div>
        </div>
      </div>

      <div className="impact-section">
        <div className="impact-inner">
          <div className="impact-text">
            <p className="section-label">Why It Matters</p>
            <h2 className="section-title">Real-World Impact</h2>
            <p className="impact-body">
              Phishing attacks account for over 90% of data breaches worldwide.
              Attackers create convincing fake pages in minutes — traditional
              blacklists can't keep up. PhishGuard uses AI to detect threats
              dynamically, even on brand-new domains.
            </p>
            <button className="btn-primary" onClick={() => onNavigate("scan")}>
              Try It Now →
            </button>
          </div>
          <div className="impact-cards">
            <div className="impact-card">
              <div className="impact-icon">🏦</div>
              <div>
                <div className="impact-card-title">Banking & Finance</div>
                <p className="impact-card-desc">
                  Detect fake banking portals and credential-harvesting pages
                  before users are compromised.
                </p>
              </div>
            </div>
            <div className="impact-card">
              <div className="impact-icon">🎓</div>
              <div>
                <div className="impact-card-title">Education & Research</div>
                <p className="impact-card-desc">
                  Built for transparency — SHAP explanations make every result
                  auditable and usable in research.
                </p>
              </div>
            </div>
            <div className="impact-card">
              <div className="impact-icon">🛡️</div>
              <div>
                <div className="impact-card-title">Everyday Protection</div>
                <p className="impact-card-desc">
                  Anyone can paste a link and get an instant, trustworthy
                  verdict — no security expertise required.
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="tech-strip">
        <p className="tech-strip-label">Built With</p>
        <div className="tech-tags">
          {[
            "Python",
            "FastAPI",
            "React.js",
            "Scikit-learn",
            "XGBoost",
            "SHAP",
            "VirusTotal API",
            "WHOIS",
            "Logistic Regression",
            "Random Forest",
          ].map((t) => (
            <span key={t} className="tech-tag">
              {t}
            </span>
          ))}
        </div>
      </div>

      <div className="final-cta">
        <h2 className="final-cta-title">
          Is that link <em>safe</em>?<br />
          Find out in seconds.
        </h2>
        <p className="final-cta-sub">
          Paste a URL and let PhishGuard's AI give you a full breakdown — free,
          fast, and explainable.
        </p>
        <button className="btn-primary" onClick={() => onNavigate("scan")}>
          Start Scanning →
        </button>
      </div>
    </div>
  );
}

// ─── About Page (placeholder) ─────────────────────────────────────────────────
function AboutPage() {
  return (
    <div
      style={{
        flex: 1,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        padding: "80px 24px",
        textAlign: "center",
      }}
    >
      <div>
        <p
          style={{
            fontFamily: "var(--mono)",
            fontSize: "11px",
            color: "var(--accent)",
            letterSpacing: "3px",
            textTransform: "uppercase",
            marginBottom: "12px",
          }}
        >
          Coming Soon
        </p>
        <h1
          style={{
            fontSize: "clamp(32px, 5vw, 56px)",
            fontWeight: "800",
            letterSpacing: "-2px",
            marginBottom: "16px",
          }}
        >
          About PhishGuard
        </h1>
        <p
          style={{
            fontFamily: "var(--mono)",
            fontSize: "13px",
            color: "var(--muted)",
            maxWidth: "400px",
            lineHeight: "1.9",
          }}
        >
          This page will cover the team, the research paper, and the model
          details.
        </p>
      </div>
    </div>
  );
}

// ─── App ──────────────────────────────────────────────────────────────────────
export default function App() {
  const [page, setPage] = useState("home");

  // scanKey forces ScanPage to fully remount when the nav Scan button is clicked
  // while already on the scan page — this resets all internal state cleanly.
  const [scanKey, setScanKey] = useState(0);

  const handleNavigate = (target) => {
    if (target === "scan" && page === "scan") {
      // Already on scan page — reset it instead of re-navigating
      setScanKey((k) => k + 1);
    } else {
      setPage(target);
    }
  };

  return (
    <>
      <div className="noise" />
      <div className="grid-bg" />
      <div className="glow-orb" />
      <div className="glow-orb-bottom" />

      <div className="app">
        {/* ─── NAV ─── */}
        <nav>
          <div className="nav-logo" onClick={() => handleNavigate("home")}>
            <div className="logo">PhishGuard AI</div>
            <div className="tagline">
              Real-Time AI Powered Phishing Detection
            </div>
          </div>
          <div className="nav-links">
            <button
              className="nav-link cta"
              onClick={() => handleNavigate("scan")}
            >
              Scan
            </button>
            <button
              className={`nav-link ${page === "about" ? "active" : ""}`}
              onClick={() => handleNavigate("about")}
            >
              About
            </button>
            <button className="nav-link disabled" title="Coming soon">
              Compare Models
            </button>
          </div>
        </nav>

        {/* ─── PAGE ROUTER ─── */}
        {page === "home" && <LandingPage onNavigate={handleNavigate} />}
        {page === "about" && <AboutPage />}
        {page === "scan" && (
          <ScanPage key={scanKey} onNewScan={() => setScanKey((k) => k + 1)} />
        )}
      </div>
    </>
  );
}
