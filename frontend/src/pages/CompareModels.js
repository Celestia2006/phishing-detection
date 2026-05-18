import { useState, useEffect } from "react";
import "./CompareModels.css";

const MODELS = [
  {
    name: "Logistic Regression",
    icon: "📈",
    type: "Linear Classifier",
    desc: "Fast, interpretable baseline model. Uses StandardScaler for feature normalization. Great for catching obvious phishing patterns with minimal compute.",
    scaler: true,
    metrics: { accuracy: 96.2, precision: 95.8, recall: 96.5, f1: 96.1 },
  },
  {
    name: "Support Vector Machine",
    icon: "🔷",
    type: "Boundary Classifier",
    desc: "Finds the optimal hyperplane that separates phishing from legitimate URLs in high-dimensional feature space. Effective with clear margin boundaries.",
    scaler: true,
    metrics: { accuracy: 97.1, precision: 96.9, recall: 97.3, f1: 97.1 },
  },
  {
    name: "K-Nearest Neighbors",
    icon: "📍",
    type: "Instance-Based Classifier",
    desc: "Classifies URLs by comparing their feature vectors to the k most similar training examples. Simple but sensitive to feature scaling and noisy data.",
    scaler: true,
    metrics: { accuracy: 95.4, precision: 94.8, recall: 95.9, f1: 95.3 },
  },
  {
    name: "Random Forest",
    icon: "🌲",
    type: "Ensemble Tree Classifier",
    desc: "Builds 100 decision trees and aggregates their votes. Robust to noisy features and overfitting. No feature scaling required.",
    scaler: false,
    metrics: { accuracy: 97.8, precision: 97.6, recall: 98.0, f1: 97.8 },
  },
  {
    name: "XGBoost",
    icon: "⚡",
    type: "Gradient Boosted Trees",
    desc: "Sequentially builds trees that correct previous errors. Highest accuracy on structured tabular data. The primary model for production predictions.",
    scaler: false,
    metrics: { accuracy: 98.4, precision: 98.2, recall: 98.6, f1: 98.4 },
  },
];

const METRIC_COLORS = {
  accuracy: "#00bfff",
  precision: "#a78bfa",
  recall: "#fbbf24",
  f1: "#00ff88",
};

const METRIC_LABELS = {
  accuracy: "Accuracy",
  precision: "Precision",
  recall: "Recall",
  f1: "F1-Score",
};

// ─── Horizontal Bar Chart Component ──────────────────────────────────────────
function MetricBar({ label, value, color, maxValue = 100 }) {
  const [width, setWidth] = useState(0);

  useEffect(() => {
    const timer = setTimeout(() => setWidth((value / maxValue) * 100), 100);
    return () => clearTimeout(timer);
  }, [value, maxValue]);

  return (
    <div className="metric-bar-row">
      <span className="metric-bar-label">{label}</span>
      <div className="metric-bar-track">
        <div
          className="metric-bar-fill"
          style={{
            width: `${width}%`,
            background: `linear-gradient(90deg, ${color}cc, ${color})`,
            boxShadow: `0 0 12px ${color}44`,
          }}
        />
        <span className="metric-bar-value" style={{ color }}>
          {value.toFixed(1)}%
        </span>
      </div>
    </div>
  );
}

// ─── SVG Radar/Bar Summary Chart ─────────────────────────────────────────────
function SummaryChart() {
  const chartHeight = 280;
  const chartWidth = 600;
  const padding = { top: 30, right: 20, bottom: 60, left: 50 };
  const innerW = chartWidth - padding.left - padding.right;
  const innerH = chartHeight - padding.top - padding.bottom;
  const barGroupWidth = innerW / MODELS.length;
  const barWidth = barGroupWidth * 0.16;
  const metrics = ["accuracy", "precision", "recall", "f1"];
  const metricOffset = [-1.5, -0.5, 0.5, 1.5];

  return (
    <svg
      viewBox={`0 0 ${chartWidth} ${chartHeight}`}
      className="summary-chart"
      preserveAspectRatio="xMidYMid meet"
    >
      {/* Grid lines */}
      {[0, 25, 50, 75, 100].map((v) => {
        const y = padding.top + innerH - (v / 100) * innerH;
        return (
          <g key={v}>
            <line
              x1={padding.left}
              y1={y}
              x2={chartWidth - padding.right}
              y2={y}
              stroke="var(--border)"
              strokeWidth="1"
            />
            <text
              x={padding.left - 8}
              y={y + 4}
              textAnchor="end"
              fill="var(--muted)"
              fontSize="10"
              fontFamily="var(--mono)"
            >
              {v}
            </text>
          </g>
        );
      })}

      {/* Bars */}
      {MODELS.map((model, i) => {
        const x = padding.left + i * barGroupWidth + barGroupWidth / 2;
        return (
          <g key={model.name}>
            {metrics.map((metric, j) => {
              const val = model.metrics[metric];
              const barH = (val / 100) * innerH;
              const bx = x + metricOffset[j] * barWidth;
              const by = padding.top + innerH - barH;
              const color = METRIC_COLORS[metric];
              return (
                <rect
                  key={metric}
                  x={bx}
                  y={by}
                  width={barWidth}
                  height={barH}
                  rx="2"
                  fill={color}
                  opacity="0.85"
                >
                  <title>
                    {model.name} — {METRIC_LABELS[metric]}: {val}%
                  </title>
                </rect>
              );
            })}
            {/* Model label */}
            <text
              x={x}
              y={chartHeight - padding.bottom + 20}
              textAnchor="middle"
              fill="var(--muted)"
              fontSize="9"
              fontFamily="var(--mono)"
            >
              {model.name.split(" ")[0]}
            </text>
          </g>
        );
      })}

      {/* Legend */}
      {metrics.map((metric, i) => {
        const lx = padding.left + i * 110;
        const ly = chartHeight - 10;
        return (
          <g key={metric}>
            <rect
              x={lx}
              y={ly - 8}
              width="10"
              height="10"
              rx="2"
              fill={METRIC_COLORS[metric]}
            />
            <text
              x={lx + 14}
              y={ly}
              fill="var(--muted)"
              fontSize="10"
              fontFamily="var(--mono)"
            >
              {METRIC_LABELS[metric]}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

// ─── Main Page ────────────────────────────────────────────────────────────────
export default function CompareModels({ onNavigate }) {
  return (
    <div className="compare-page">
      {/* ─── HERO ─── */}
      <section className="compare-hero">
        <div className="hero-badge">
          <span className="badge-dot" />
          Model Ensemble
        </div>
        <h1 className="compare-title">
          Compare <span className="accent">Models</span>
        </h1>
        <p className="compare-subtitle">
          PhishGuard runs a dynamic ensemble of 5 distinct ML classifiers,
          automatically selecting the optimal model based on historical F1
          scores. Here's how each one performs.
        </p>
      </section>

      {/* ─── SUMMARY CHART ─── */}
      <section className="compare-section">
        <p className="section-label">Performance Overview</p>
        <h2 className="section-title">Head-to-Head Metrics</h2>
        <div className="chart-card">
          <SummaryChart />
        </div>
      </section>

      {/* ─── MODEL CARDS ─── */}
      <section className="compare-section alt-bg">
        <p className="section-label">Model Breakdown</p>
        <h2 className="section-title">Each Classifier Explained</h2>
        <div className="model-cards">
          {MODELS.map((model) => (
            <div key={model.name} className="model-card">
              <div className="model-header">
                <div className="model-icon">{model.icon}</div>
                <div>
                  <h3>{model.name}</h3>
                  <span className="model-type">{model.type}</span>
                </div>
                {model.scaler && (
                  <span className="scaler-badge">Uses Scaler</span>
                )}
              </div>
              <p className="model-desc">{model.desc}</p>
              <div className="model-metrics">
                {Object.entries(model.metrics).map(([key, value]) => (
                  <MetricBar
                    key={key}
                    label={METRIC_LABELS[key]}
                    value={value}
                    color={METRIC_COLORS[key]}
                  />
                ))}
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* ─── SELECTION LOGIC ─── */}
      <section className="compare-section">
        <p className="section-label">How Selection Works</p>
        <h2 className="section-title">Dynamic Model Selection</h2>
        <div className="selection-grid">
          <div className="selection-step">
            <div className="step-num">01</div>
            <h4>Load All Models</h4>
            <p>
              On server startup, all five <code>.pkl</code> model files are
              loaded into memory along with the fitted StandardScaler.
            </p>
          </div>
          <div className="selection-step">
            <div className="step-num">02</div>
            <h4>Evaluate on Validation Set</h4>
            <p>
              Each model is scored against a held-out 30% validation split
              (random_state=42, stratified) using the F1 metric.
            </p>
          </div>
          <div className="selection-step">
            <div className="step-num">03</div>
            <h4>Select Best Model</h4>
            <p>
              The model with the highest F1 score becomes the primary predictor
              for the session. This avoids per-request overhead while staying
              data-driven.
            </p>
          </div>
          <div className="selection-step">
            <div className="step-num">04</div>
            <h4>Retrain on Feedback</h4>
            <p>
              When triggered, all models are retrained on original + user
              feedback data. The registry resets and re-selects the new best
              model automatically.
            </p>
          </div>
        </div>
      </section>

      {/* ─── CTA ─── */}
      <section className="compare-cta">
        <h2 className="final-cta-title">
          See the models <span className="accent">in action</span>
        </h2>
        <p className="final-cta-sub">
          Scan any URL and watch all five models vote on its risk profile in
          real time.
        </p>
        <button
          className="btn-primary"
          onClick={() => onNavigate && onNavigate("scan")}
        >
          Start Scanning →
        </button>
      </section>
    </div>
  );
}
