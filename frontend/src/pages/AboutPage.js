import "./AboutPage.css";

const TEAM = [
  {
    name: "Anshita Sugandhi",
    role: "Backend",
    icon: "⚙️",
    desc: "Designed and built the FastAPI backend pipeline - feature extraction, model prediction, WHOIS analysis, and the adaptive feedback system for continuous model improvement.",
  },
  {
    name: "Farhana Tabassum",
    role: "Frontend",
    icon: "🎨",
    desc: "Developed the complete React.js interface - responsive scan dashboard, real-time result cards, trust-score gauges, and the dark-theme design system across all pages.",
  },
  {
    name: "Gangisetti Himasree",
    role: "Model Training",
    icon: "🧠",
    desc: "Trained, evaluated, and optimized the ensemble of ML classifiers - Logistic Regression, SVM, KNN, Random Forest, and XGBoost - achieving 98.4% detection accuracy.",
  },
];

export default function AboutPage() {
  return (
    <div className="about-page">
      <section className="about-hero">
        <div className="hero-badge">
          <span className="badge-dot" />
          NGIT CSE 2025-26
        </div>
        <h1 className="about-title">
          About <span className="accent">PhishGuard</span>
        </h1>
        <p className="about-subtitle">
          PhishGuard is an adaptive, explainable phishing detection system that
          combines machine learning with SHAP (SHapley Additive exPlanations)
          and real-time WHOIS domain analysis. It extracts 28 features from any
          URL, runs them through an ensemble of ML classifiers, and returns a
          transparent, auditable verdict - showing exactly which factors drove
          the decision.
        </p>
      </section>

      <section className="about-section">
        <p className="section-label">The Mission</p>
        <h2 className="section-title">Why PhishGuard Exists</h2>
        <div className="mission-cards">
          <div className="mission-card">
            <div className="mission-icon">🔬</div>
            <h3>Explainable AI</h3>
            <p>
              Traditional phishing detectors are black boxes. PhishGuard uses
              SHAP to show exactly <em>why</em> a URL was flagged, making every
              result transparent and auditable.
            </p>
          </div>
          <div className="mission-card">
            <div className="mission-icon">🌐</div>
            <h3>WHOIS Intelligence</h3>
            <p>
              Real-time domain registration analysis - age, expiry, privacy
              protection, and name-server reputation - to catch newly-created
              phishing sites before blacklists can.
            </p>
          </div>
          <div className="mission-card">
            <div className="mission-icon">🔄</div>
            <h3>Adaptive Learning</h3>
            <p>
              User feedback is collected and used to retrain models on-the-fly,
              ensuring the system evolves with emerging phishing tactics without
              needing a full re-deployment.
            </p>
          </div>
        </div>
      </section>

      <section className="about-section">
        <p className="section-label">The Team</p>
        <h2 className="section-title">Meet the Developers</h2>
        <div className="team-grid">
          {TEAM.map((member) => (
            <div key={member.name} className="team-card">
              <div className="team-avatar">{member.icon}</div>
              <h3>{member.name}</h3>
              <p className="team-role">{member.role}</p>
              <p className="team-desc">{member.desc}</p>
            </div>
          ))}
        </div>

        <div className="guide-card">
          <div className="guide-icon">🎓</div>
          <div>
            <p className="guide-label">Project Guide</p>
            <h3>Dr. M. Shabana</h3>
            <p className="guide-dept">
              Department of Computer Science & Engineering - NGIT
            </p>
          </div>
        </div>
      </section>
    </div>
  );
}
