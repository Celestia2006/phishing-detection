import "./AboutPage.css";

const GUIDE = {
  name: "Dr. M. Shabana",
  title: "Project Guide",
  dept: "Department of Computer Science & Engineering — NGIT",
};

const TEAM = [
  {
    name: "Anshita Sugandhi",
    role: "Backend Development",
    desc: "Designed and built the FastAPI backend pipeline — feature extraction, model prediction, WHOIS analysis, and the adaptive feedback system for continuous model improvement.",
  },
  {
    name: "Farhana Tabassum",
    role: "Frontend Development",
    desc: "Developed the complete React.js interface — responsive scan dashboard, real-time result cards, trust-score gauges, and the dark-theme design system across all pages.",
  },
  {
    name: "Gangisetti Himasree",
    role: "Model Training & Evaluation",
    desc: "Trained, evaluated, and optimized the ensemble of ML classifiers — Logistic Regression, SVM, KNN, Random Forest, and XGBoost — achieving 98.4% detection accuracy.",
  },
];

const TECH_STACK = [
  { category: "Backend", items: ["FastAPI", "Python", "Scikit-learn", "XGBoost", "SHAP", "python-whois"] },
  { category: "Frontend", items: ["React.js", "CSS3", "Vite"] },
  { category: "ML Models", items: ["Logistic Regression", "SVM", "KNN", "Random Forest", "XGBoost"] },
  { category: "Feature Engineering", items: ["28 URL-based features", "DNS/WHOIS lookups", "HTML page analysis", "External API integration"] },
];

export default function AboutPage() {
  return (
    <div className="about-page">
      {/* ─── HERO ─── */}
      <section className="about-hero">
        <div className="hero-badge">
          <span className="badge-dot" />
          NGIT CSE 2025-26
        </div>
        <h1 className="about-title">
          About <span className="accent">PhishLens</span>
        </h1>
        <p className="about-subtitle">
          PhishLens is an adaptive, explainable phishing detection system that
          combines machine learning with SHAP (SHapley Additive exPlanations)
          and real-time WHOIS domain analysis. It extracts 28 features from any
          URL, runs them through an ensemble of ML classifiers, and returns a
          transparent, auditable verdict — showing exactly which factors drove
          the decision.
        </p>
      </section>

      {/* ─── WHY PhishLens EXISTS ─── */}
      <section className="about-section">
        <p className="section-label">The Mission</p>
        <h2 className="section-title">Why PhishLens Exists</h2>

        <div className="stats-paragraph">
          <p>
            Phishing remains one of the most pervasive and damaging cyber
            threats in the world today. Over{" "}
            <strong>3.4 billion phishing emails</strong> are sent every single
            day, and phishing is cited as the initial attack vector in more than{" "}
            <strong>80% of reported security incidents</strong>
            {" "}globally. It serves as the primary delivery mechanism for
            ransomware campaigns, business email compromise, and credential
            theft — attacks that have crippled hospitals, governments, and
            Fortune 500 companies. Organizations lose an estimated{" "}
            <strong>$10 billion annually</strong> to phishing-related fraud,
            while the average cost of a data breach involving phishing exceeds
            $4.9 million. Traditional blacklist-based defenses cannot keep pace
            with attackers who spin up convincing fake portals in minutes.
            PhishLens was built to close that gap with AI-driven, real-time
            detection that works even on brand-new domains, providing
            explainable verdicts that users can trust and auditors can verify.
          </p>
        </div>

        <div className="mission-cards">
          <div className="mission-card">
            <div className="mission-icon">🔬</div>
            <h3>Explainable AI</h3>
            <p>
              Traditional phishing detectors are black boxes. PhishLens uses
              SHAP to show exactly <em>why</em> a URL was flagged, making every
              result transparent and auditable.
            </p>
          </div>
          <div className="mission-card">
            <div className="mission-icon">🌐</div>
            <h3>WHOIS Intelligence</h3>
            <p>
              Real-time domain registration analysis — age, expiry, privacy
              protection, and name-server reputation — to catch newly-created
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

      {/* ─── DATASET & TECHNOLOGIES ─── */}
      <section className="about-section alt-bg">
        <p className="section-label">Under the Hood</p>
        <h2 className="section-title">Dataset & Technologies</h2>

        <div className="tech-description">
          <p>
            Models are trained on the{" "}
            <strong>UCI Phishing Websites Dataset</strong> (
            <a
              href="https://archive.ics.uci.edu/dataset/327/phishing+websites"
              target="_blank"
              rel="noopener noreferrer"
            >
              source
            </a>
            ). The system extracts <strong>28 distinct features</strong> from
            every URL, grouped into four analytical categories:
          </p>
          <ul className="feature-list">
            <li>
              <strong>URL-Based Features </strong> — Pure string and regex
              parsing: IP address usage, URL length, @ symbols, double-slash
              redirects, hyphenated domains, subdomain depth, HTTPS tokens,
              redirect patterns, and non-standard ports.
            </li>
            <li>
              <strong>DNS / WHOIS Features </strong> — External network
              lookups: DNS A-record existence, domain age from WHOIS creation
              date, and domain registration length from WHOIS expiry date.
            </li>
            <li>
              <strong>HTML / Page Features </strong> — HTTP fetch and DOM
              parsing: favicon origin, external resource ratios, suspicious
              anchor links, external links in meta/script tags, server form
              handlers, mailto form submissions, abnormal URL matching WHOIS,
              mouseover status spoofing, right-click disabling, popup windows,
              and invisible iframes.
            </li>
            <li>
              <strong>API-Based Features </strong> — Third-party
              intelligence: SSL certificate validity (Python ssl library),
              Google Safe Browsing threat check, VirusTotal multi-vendor scan
              aggregation, and OpenPageRank domain authority scoring.
            </li>
          </ul>
        </div>

        <div className="tech-description" style={{ marginTop: "48px" }}>
          <p>
            <strong>Application Architecture:</strong> PhishLens follows a
            decoupled client-server model. The{" "}
            <strong>React.js frontend</strong> provides a responsive,
            dark-themed dashboard for URL submission, real-time result
            visualization, trust-score gauges, SHAP explanation bars, and WHOIS
            analysis panels. The <strong>FastAPI backend</strong> orchestrates
            the full ML pipeline — feature extraction, multi-model inference,
            SHAP explanation generation, and WHOIS risk analysis — returning a
            unified JSON response in a single API call.
          </p>
          <p style={{ marginTop: "16px" }}>
            <strong>ML Stack:</strong> The system runs an ensemble of five
            classifiers — <strong>Logistic Regression</strong> (fast linear
            baseline), <strong>Support Vector Machine</strong> (high-dimensional
            hyperplane separator), <strong>K-Nearest Neighbors</strong>{" "}
            (instance-based distance learning), <strong>Random Forest</strong>{" "}
            (bagging ensemble of decision trees), and <strong>XGBoost</strong>{" "}
            (gradient boosted trees). On startup, all models are evaluated on a
            held-out validation split and the highest F1-scoring model is
            selected as the primary predictor.
          </p>
          <p style={{ marginTop: "16px" }}>
            <strong>Explainability:</strong> Both local (per-prediction) and
            global (model-level) SHAP feature importances are computed and
            returned to the frontend. Local explanations show the top 5 features
            that drove each individual verdict, while global explanations reveal
            which features matter most across the entire dataset — making every
            decision transparent and auditable.
          </p>
        </div>

        <div className="tech-stack-grid">
          {TECH_STACK.map((group) => (
            <div key={group.category} className="tech-stack-card">
              <h3>{group.category}</h3>
              <div className="tech-tags">
                {group.items.map((item) => (
                  <span key={item} className="tech-tag">
                    {item}
                  </span>
                ))}
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* ─── THE TEAM ─── */}
      <section className="about-section">
        <p className="section-label">The Team</p>
        <h2 className="section-title">Meet the Developers</h2>

        {/* Guide — first, prominent */}
        <div className="guide-frame">
          <div className="guide-frame-header">
            <span className="guide-frame-badge">Project Guide</span>
          </div>
          <div className="guide-frame-body">
            <h3>{GUIDE.name}</h3>
            <p>{GUIDE.dept}</p>
          </div>
        </div>

        {/* Student team — below guide */}
        <div className="team-grid">
          {TEAM.map((member) => (
            <div key={member.name} className="team-frame">
              <div className="team-frame-header">
                <span className="team-frame-role">{member.role}</span>
              </div>
              <div className="team-frame-body">
                <h3>{member.name}</h3>
                <p>{member.desc}</p>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* ─── ACKNOWLEDGEMENTS ─── */}
      <section className="about-section alt-bg">
        <p className="section-label">Acknowledgements</p>
        <h2 className="section-title">Acknowledgements</h2>

        <div className="acknowledgements-card">
          <p>
            We sincerely thank our project guide,{" "}
            <strong>Dr. M. Shabana</strong>, Department of Computer Science &
            Engineering, Neil Gogte Institute of Technology (NGIT), for her
            invaluable mentorship, technical guidance, and continuous
            encouragement throughout the conception, development, and evaluation
            of this project. Her expertise in machine learning and cybersecurity
            was instrumental in shaping the architecture and research direction
            of PhishLens.
          </p>
          <p>
            We are grateful to{" "}
            <strong>Neil Gogte Institute of Technology (NGIT)</strong> for
            providing the computing infrastructure, laboratory resources, and
            academic environment that made this work possible. The institution's
            commitment to hands-on, research-driven learning provided the
            foundation for this project's development.
          </p>
          <p>
            We also acknowledge the{" "}
            <strong>UCI Machine Learning Repository</strong> for providing the
            foundational dataset used to train and evaluate our models —{" "}
            <a
              href="https://archive.ics.uci.edu/dataset/327/phishing+websites"
              target="_blank"
              rel="noopener noreferrer"
            >
              UCI Machine Learning Repository, Phishing Websites Dataset.
              https://archive.ics.uci.edu/dataset/327/phishing+websites
            </a>
            .
          </p>
        </div>
      </section>
    </div>
  );
}
