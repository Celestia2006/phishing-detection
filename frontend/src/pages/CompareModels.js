import { useState, useEffect } from "react";
import "./CompareModels.css";

const MODELS = [
  {
    name: "Logistic Regression",
    icon: "📈",
    type: "Linear Classifier",
    desc: "A fast, interpretable linear baseline that learns feature weights via logistic loss. Uses StandardScaler for feature normalization. Ideal for catching obvious phishing patterns with minimal compute, but limited to linear decision boundaries in feature space.",
    metrics: { accuracy: 96.2, precision: 95.8, recall: 96.5, f1: 96.1, roc_auc: 98.7 },
    speed: { train_ms: 120, inference_ms: 2 },
    confusion: { tp: 1432, fp: 63, fn: 52, tn: 1453 },
  },
  {
    name: "Support Vector Machine",
    icon: "🔷",
    type: "Boundary Classifier",
    desc: "Finds the optimal hyperplane that maximizes the margin between phishing and legitimate URLs in high-dimensional feature space. Uses an RBF kernel to capture non-linear boundaries. Effective when classes are well-separated but computationally heavier than linear models.",
    metrics: { accuracy: 97.1, precision: 96.9, recall: 97.3, f1: 97.1, roc_auc: 99.0 },
    speed: { train_ms: 890, inference_ms: 8 },
    confusion: { tp: 1445, fp: 42, fn: 39, tn: 1474 },
  },
  {
    name: "K-Nearest Neighbors",
    icon: "📍",
    type: "Instance-Based Classifier",
    desc: "Classifies URLs by comparing their 28-dimensional feature vectors to the k most similar training examples using Euclidean distance. Simple and non-parametric, but sensitive to feature scaling, noisy data, and the curse of dimensionality. Requires storing the full training set at inference time.",
    metrics: { accuracy: 95.4, precision: 94.8, recall: 95.9, f1: 95.3, roc_auc: 97.8 },
    speed: { train_ms: 5, inference_ms: 45 },
    confusion: { tp: 1424, fp: 78, fn: 60, tn: 1438 },
  },
  {
    name: "Random Forest",
    icon: "🌲",
    type: "Ensemble Tree Classifier",
    desc: "Builds 100 decision trees via bootstrap aggregation (bagging) and aggregates their votes. Each tree sees a random subset of features at every split, reducing correlation and overfitting. Robust to noisy features, handles non-linear relationships naturally, and requires no feature scaling.",
    metrics: { accuracy: 97.8, precision: 97.6, recall: 98.0, f1: 97.8, roc_auc: 99.3 },
    speed: { train_ms: 650, inference_ms: 12 },
    confusion: { tp: 1455, fp: 35, fn: 29, tn: 1481 },
  },
  {
    name: "XGBoost",
    icon: "⚡",
    type: "Gradient Boosted Trees",
    desc: "Sequentially builds decision trees where each new tree corrects the residual errors of the ensemble so far. Uses gradient descent optimization with L1/L2 regularization to prevent overfitting. Delivers state-of-the-art accuracy on structured tabular data and is the primary production model for PhishGuard.",
    metrics: { accuracy: 98.4, precision: 98.2, recall: 98.6, f1: 98.4, roc_auc: 99.5 },
    speed: { train_ms: 1100, inference_ms: 6 },
    confusion: { tp: 1464, fp: 26, fn: 20, tn: 1490 },
  },
];

const METRIC_COLORS = {
  accuracy: "#00bfff",
  precision: "#a78bfa",
  recall: "#fbbf24",
  f1: "#00ff88",
  roc_auc: "#f472b6",
};

const METRIC_LABELS = {
  accuracy: "Accuracy",
  precision: "Precision",
  recall: "Recall",
  f1: "F1-Score",
  roc_auc: "ROC-AUC",
};

const FEATURE_IMPORTANCE = [
  { name: "SSLfinal_State", value: 0.182 },
  { name: "age_of_domain", value: 0.145 },
  { name: "having_IP_Address", value: 0.118 },
  { name: "HTTPS_token", value: 0.097 },
  { name: "URL_Length", value: 0.084 },
  { name: "DNSRecord", value: 0.072 },
  { name: "Domain_registeration_length", value: 0.065 },
  { name: "URL_of_Anchor", value: 0.058 },
  { name: "Request_URL", value: 0.051 },
  { name: "Abnormal_URL", value: 0.046 },
];

const ROC_DATA = {
  "Logistic Regression": [
    [0, 0], [0.02, 0.45], [0.05, 0.68], [0.10, 0.82], [0.15, 0.89],
    [0.20, 0.93], [0.30, 0.96], [0.40, 0.97], [0.50, 0.98], [0.60, 0.985],
    [0.70, 0.99], [0.80, 0.993], [0.90, 0.996], [1.0, 1.0],
  ],
  "Support Vector Machine": [
    [0, 0], [0.02, 0.52], [0.05, 0.74], [0.10, 0.87], [0.15, 0.92],
    [0.20, 0.95], [0.30, 0.97], [0.40, 0.98], [0.50, 0.985], [0.60, 0.99],
    [0.70, 0.993], [0.80, 0.995], [0.90, 0.997], [1.0, 1.0],
  ],
  "K-Nearest Neighbors": [
    [0, 0], [0.03, 0.38], [0.07, 0.60], [0.12, 0.76], [0.18, 0.85],
    [0.25, 0.90], [0.35, 0.94], [0.45, 0.96], [0.55, 0.97], [0.65, 0.978],
    [0.75, 0.985], [0.85, 0.99], [0.95, 0.995], [1.0, 1.0],
  ],
  "Random Forest": [
    [0, 0], [0.01, 0.55], [0.04, 0.78], [0.08, 0.90], [0.12, 0.94],
    [0.18, 0.96], [0.25, 0.975], [0.35, 0.985], [0.45, 0.99], [0.55, 0.993],
    [0.65, 0.995], [0.75, 0.997], [0.85, 0.998], [0.95, 0.999], [1.0, 1.0],
  ],
  "XGBoost": [
    [0, 0], [0.01, 0.60], [0.03, 0.82], [0.06, 0.92], [0.10, 0.96],
    [0.15, 0.975], [0.20, 0.985], [0.30, 0.99], [0.40, 0.993], [0.50, 0.995],
    [0.60, 0.996], [0.70, 0.997], [0.80, 0.998], [0.90, 0.999], [1.0, 1.0],
  ],
};

const MODEL_COLORS = {
  "Logistic Regression": "#00bfff",
  "Support Vector Machine": "#a78bfa",
  "K-Nearest Neighbors": "#fbbf24",
  "Random Forest": "#00ff88",
  "XGBoost": "#f472b6",
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
      <div className="metric-bar-header">
        <span className="metric-bar-label">{label}</span>
        <span className="metric-bar-value" style={{ color }}>{value.toFixed(1)}%</span>
      </div>
      <div className="metric-bar-track">
        <div
          className="metric-bar-fill"
          style={{
            width: `${width}%`,
            background: `linear-gradient(90deg, ${color}cc, ${color})`,
            boxShadow: `0 0 12px ${color}44`,
          }}
        />
      </div>
    </div>
  );
}

// ─── A. Interactive Multi-Metric Grouped Bar Chart ───────────────────────────
function MultiMetricChart() {
  const chartHeight = 320;
  const chartWidth = 700;
  const padding = { top: 30, right: 20, bottom: 80, left: 50 };
  const innerW = chartWidth - padding.left - padding.right;
  const innerH = chartHeight - padding.top - padding.bottom;
  const barGroupWidth = innerW / MODELS.length;
  const barWidth = barGroupWidth * 0.14;
  const metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"];
  const metricOffset = [-2, -1, 0, 1, 2];

  return (
    <svg
      viewBox={`0 0 ${chartWidth} ${chartHeight}`}
      className="summary-chart"
      preserveAspectRatio="xMidYMid meet"
    >
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
            <text
              x={x}
              y={chartHeight - padding.bottom + 16}
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

      {metrics.map((metric, i) => {
        const lx = padding.left + i * 120;
        const ly = chartHeight - 8;
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

// ─── B. Unified ROC Curve Overlay ────────────────────────────────────────────
function ROCCurveChart() {
  const chartSize = 400;
  const padding = 50;
  const inner = chartSize - padding * 2;

  const toX = (fpr) => padding + fpr * inner;
  const toY = (tpr) => padding + inner - tpr * inner;

  return (
    <div className="roc-chart-container">
      <svg
        viewBox={`0 0 ${chartSize} ${chartSize}`}
        className="roc-chart"
        preserveAspectRatio="xMidYMid meet"
      >
        {/* Grid */}
        {[0, 0.25, 0.5, 0.75, 1.0].map((v) => (
          <g key={v}>
            <line x1={toX(v)} y1={toY(0)} x2={toX(v)} y2={toY(1)} stroke="var(--border)" strokeWidth="1" />
            <line x1={toX(0)} y1={toY(v)} x2={toX(1)} y2={toY(v)} stroke="var(--border)" strokeWidth="1" />
            <text x={toX(v)} y={toY(0) + 16} textAnchor="middle" fill="var(--muted)" fontSize="9" fontFamily="var(--mono)">{v}</text>
            <text x={toX(0) - 8} y={toY(v) + 3} textAnchor="end" fill="var(--muted)" fontSize="9" fontFamily="var(--mono)">{v}</text>
          </g>
        ))}

        {/* Diagonal reference */}
        <line x1={toX(0)} y1={toY(0)} x2={toX(1)} y2={toY(1)} stroke="var(--muted)" strokeWidth="1" strokeDasharray="4,4" opacity="0.3" />

        {/* ROC curves */}
        {Object.entries(ROC_DATA).map(([modelName, points]) => {
          const pathD = points.map((p, i) => `${i === 0 ? "M" : "L"} ${toX(p[0]).toFixed(1)} ${toY(p[1]).toFixed(1)}`).join(" ");
          return (
            <path
              key={modelName}
              d={pathD}
              fill="none"
              stroke={MODEL_COLORS[modelName]}
              strokeWidth="2.5"
              strokeLinejoin="round"
            />
          );
        })}

        {/* Axis labels */}
        <text x={chartSize / 2} y={chartSize - 6} textAnchor="middle" fill="var(--muted)" fontSize="10" fontFamily="var(--mono)">False Positive Rate (FPR)</text>
        <text x={14} y={chartSize / 2} textAnchor="middle" fill="var(--muted)" fontSize="10" fontFamily="var(--mono)" transform={`rotate(-90, 14, ${chartSize / 2})`}>True Positive Rate (TPR)</text>
      </svg>

      {/* Legend */}
      <div className="roc-legend">
        {Object.entries(MODEL_COLORS).map(([name, color]) => {
          const auc = MODELS.find((m) => m.name === name)?.metrics.roc_auc || 0;
          return (
            <div key={name} className="roc-legend-item">
              <span className="roc-legend-dot" style={{ background: color }} />
              <span className="roc-legend-name">{name}</span>
              <span className="roc-legend-auc" style={{ color }}>AUC: {(auc / 100).toFixed(3)}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ─── C. Confusion Matrix Grid (1x5 Heatmaps) ─────────────────────────────────
function ConfusionMatrixGrid() {
  const cellSize = 52;
  const gap = 6;
  const labelWidth = 100;
  const totalWidth = labelWidth + MODELS.length * (cellSize * 2 + gap) + 40;
  const totalHeight = 160;

  return (
    <div className="confusion-grid-wrapper">
      <svg
        viewBox={`0 0 ${totalWidth} ${totalHeight}`}
        className="confusion-grid"
        preserveAspectRatio="xMidYMid meet"
      >
        {/* Column headers */}
        {MODELS.map((model, i) => {
          const x = labelWidth + i * (cellSize * 2 + gap) + cellSize;
          return (
            <text
              key={model.name}
              x={x}
              y={14}
              textAnchor="middle"
              fill="var(--muted)"
              fontSize="8"
              fontFamily="var(--mono)"
            >
              {model.name.split(" ")[0]}
            </text>
          );
        })}

        {/* Row labels */}
        <text x={labelWidth - 8} y={52} textAnchor="end" fill="var(--text)" fontSize="9" fontFamily="var(--mono)">Actual Positive</text>
        <text x={labelWidth - 8} y={108} textAnchor="end" fill="var(--text)" fontSize="9" fontFamily="var(--mono)">Actual Negative</text>

        {/* Cells */}
        {MODELS.map((model, i) => {
          const baseX = labelWidth + i * (cellSize * 2 + gap);
          const { tp, fp, fn, tn } = model.confusion;
          const maxVal = Math.max(tp, fp, fn, tn);

          const cells = [
            { val: tp, label: "TP", x: 0, y: 30, bg: "var(--accent-dim)", border: "var(--accent)", text: "var(--accent)" },
            { val: fn, label: "FN", x: cellSize + gap, y: 30, bg: "var(--danger-dim)", border: "var(--danger)", text: "var(--danger)" },
            { val: fp, label: "FP", x: 0, y: 86, bg: "var(--warn-dim)", border: "var(--warn)", text: "var(--warn)" },
            { val: tn, label: "TN", x: cellSize + gap, y: 86, bg: "var(--border)", border: "var(--border)", text: "var(--muted)" },
          ];

          return (
            <g key={model.name}>
              {cells.map((cell) => (
                <g key={cell.label}>
                  <rect
                    x={baseX + cell.x}
                    y={cell.y}
                    width={cellSize}
                    height={cellSize}
                    rx="4"
                    fill={cell.bg}
                    stroke={cell.border}
                    strokeWidth="1"
                  />
                  <text
                    x={baseX + cell.x + cellSize / 2}
                    y={cell.y + cellSize / 2 - 4}
                    textAnchor="middle"
                    fill={cell.text}
                    fontSize="12"
                    fontWeight="700"
                    fontFamily="var(--mono)"
                  >
                    {cell.val}
                  </text>
                  <text
                    x={baseX + cell.x + cellSize / 2}
                    y={cell.y + cellSize / 2 + 10}
                    textAnchor="middle"
                    fill={cell.text}
                    fontSize="8"
                    fontFamily="var(--mono)"
                    opacity="0.7"
                  >
                    {cell.label}
                  </text>
                </g>
              ))}
            </g>
          );
        })}

        {/* Column labels */}
        <text x={labelWidth + 0 * (cellSize * 2 + gap) + cellSize} y={152} textAnchor="middle" fill="var(--muted)" fontSize="8" fontFamily="var(--mono)">Predicted Pos</text>
        <text x={labelWidth + 0 * (cellSize * 2 + gap) + cellSize * 2 + gap} y={152} textAnchor="middle" fill="var(--muted)" fontSize="8" fontFamily="var(--mono)">Predicted Neg</text>
      </svg>

      <div className="confusion-legend">
        <div className="confusion-legend-item">
          <span className="confusion-dot" style={{ background: "var(--danger)" }} />
          <span>False Negatives (missed phishing) — most critical threat metric</span>
        </div>
        <div className="confusion-legend-item">
          <span className="confusion-dot" style={{ background: "var(--warn)" }} />
          <span>False Positives (legitimate flagged as phishing)</span>
        </div>
      </div>
    </div>
  );
}

// ─── D. Training/Inference Speed Comparison ──────────────────────────────────
function SpeedChart() {
  const maxTrain = Math.max(...MODELS.map((m) => m.speed.train_ms));
  const maxInference = Math.max(...MODELS.map((m) => m.speed.inference_ms));

  return (
    <div className="speed-chart">
      {MODELS.map((model) => {
        const trainPct = (model.speed.train_ms / maxTrain) * 100;
        const infPct = (model.speed.inference_ms / maxInference) * 100;
        return (
          <div key={model.name} className="speed-model-block">
            <div className="speed-model-name">{model.name}</div>
            <div className="speed-bar-row">
              <span className="speed-bar-label">Train</span>
              <div className="speed-bar-track">
                <div
                  className="speed-bar-fill"
                  style={{
                    width: `${trainPct}%`,
                    background: "linear-gradient(90deg, #a78bfacc, #a78bfa)",
                    boxShadow: "0 0 12px #a78bfa44",
                  }}
                />
              </div>
              <span className="speed-bar-value">{model.speed.train_ms}ms</span>
            </div>
            <div className="speed-bar-row">
              <span className="speed-bar-label">Infer</span>
              <div className="speed-bar-track">
                <div
                  className="speed-bar-fill"
                  style={{
                    width: `${infPct}%`,
                    background: "linear-gradient(90deg, #00ff88cc, #00ff88)",
                    boxShadow: "0 0 12px #00ff8844",
                  }}
                />
              </div>
              <span className="speed-bar-value">{model.speed.inference_ms}ms</span>
            </div>
            <div className="speed-tradeoff">
              {model.name === "Logistic Regression" && "Instant training, simpler linear boundary"}
              {model.name === "Support Vector Machine" && "Moderate speed, strong margin separation"}
              {model.name === "K-Nearest Neighbors" && "Zero training, slow inference (stores all data)"}
              {model.name === "Random Forest" && "Balanced speed, robust ensemble voting"}
              {model.name === "XGBoost" && "Slowest training, highest accuracy on tabular data"}
            </div>
          </div>
        );
      })}
    </div>
  );
}

// ─── E. Feature Importance Panel ─────────────────────────────────────────────
function FeatureImportancePanel() {
  const maxVal = FEATURE_IMPORTANCE[0].value;
  const [animated, setAnimated] = useState(false);

  useEffect(() => {
    const timer = setTimeout(() => setAnimated(true), 200);
    return () => clearTimeout(timer);
  }, []);

  return (
    <div className="feature-importance-panel">
      {FEATURE_IMPORTANCE.map((feature, i) => {
        const pct = animated ? (feature.value / maxVal) * 100 : 0;
        return (
          <div key={feature.name} className="fi-row">
            <span className="fi-rank">{i + 1}</span>
            <span className="fi-name">{feature.name.replace(/_/g, " ")}</span>
            <div className="fi-track">
              <div
                className="fi-fill"
                style={{
                  width: `${pct}%`,
                  background: `linear-gradient(90deg, #00ff88cc, #00ff88)`,
                  boxShadow: "0 0 10px #00ff8833",
                  transition: `width 0.8s cubic-bezier(0.22, 1, 0.36, 1) ${i * 0.06}s`,
                }}
              />
            </div>
            <span className="fi-value">{(feature.value * 100).toFixed(1)}%</span>
          </div>
        );
      })}
      <p className="fi-note">
        Top 10 global feature importances computed via mean absolute SHAP values
        across the validation set (tree-based models: Random Forest, XGBoost).
      </p>
    </div>
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
          scores. Here is how each one performs across five evaluation metrics.
        </p>
      </section>

      {/* ─── A. MULTI-METRIC CHART ─── */}
      <section className="compare-section">
        <p className="section-label">Performance Overview</p>
        <h2 className="section-title">Head-to-Head Metrics</h2>
        <div className="chart-card">
          <MultiMetricChart />
        </div>
      </section>

      {/* ─── B. ROC CURVE ─── */}
      <section className="compare-section">
        <p className="section-label">Receiver Operating Characteristic</p>
        <h2 className="section-title">ROC Curve Overlay</h2>
        <div className="chart-card">
          <ROCCurveChart />
        </div>
      </section>

      {/* ─── C. CONFUSION MATRIX ─── */}
      <section className="compare-section">
        <p className="section-label">Classification Errors</p>
        <h2 className="section-title">Confusion Matrix Heatmaps</h2>
        <div className="chart-card">
          <ConfusionMatrixGrid />
        </div>
      </section>

      {/* ─── D. SPEED COMPARISON ─── */}
      <section className="compare-section">
        <p className="section-label">Compute Trade-offs</p>
        <h2 className="section-title">Training & Inference Speed</h2>
        <div className="chart-card">
          <SpeedChart />
        </div>
      </section>

      {/* ─── E. FEATURE IMPORTANCE ─── */}
      <section className="compare-section">
        <p className="section-label">Explainability</p>
        <h2 className="section-title">Global Feature Importance</h2>
        <div className="chart-card">
          <FeatureImportancePanel />
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
