import { useEffect, useMemo, useState } from 'react';
import {
  Activity,
  BarChart3,
  Calculator,
  Check,
  Download,
  Factory,
  Gauge,
  GitCompare,
  LineChart,
  Plus,
  RefreshCw,
  RotateCcw,
  Sigma,
  Waves,
} from 'lucide-react';
import ResultPanel from '../components/ResultPanel.jsx';
import RunButton from '../components/RunButton.jsx';
import MetricTable from '../components/MetricTable.jsx';
import R2RSchematic from '../components/R2RSchematic.jsx';
import { DEFAULT_API_BASE, apiGet, apiPost, artifactUrl } from './api.js';
import { downloadCsv } from './csv.js';

const FALLBACK_TLOG_OPTIONS = [1, 2, 5, 10, 20, 50, 100];
const AUTO_LOAD_REQUESTS = new Map();
const API_BASE_STORAGE_KEY = 'r2r-dashboard-api-base';
function migrateKnownStaleApiBase(baseUrl) {
  const normalized = baseUrl?.trim().replace(/\/+$/, '');
  const isCurrentLocalDashboard =
    typeof window !== 'undefined' &&
    ['127.0.0.1', 'localhost'].includes(window.location.hostname) &&
    window.location.port === '5202';
  const isOldLocalBackend = [
    'http://127.0.0.1:8014',
    'http://localhost:8014',
  ].includes(normalized);
  return isCurrentLocalDashboard && isOldLocalBackend
    ? DEFAULT_API_BASE
    : normalized;
}

function storedApiBaseUrl() {
  if (typeof window === 'undefined') return DEFAULT_API_BASE;
  try {
    const stored = window.localStorage.getItem(API_BASE_STORAGE_KEY);
    return migrateKnownStaleApiBase(stored) || DEFAULT_API_BASE;
  } catch {
    return DEFAULT_API_BASE;
  }
}

function storeApiBaseUrl(baseUrl) {
  if (typeof window === 'undefined') return;
  try {
    window.localStorage.setItem(API_BASE_STORAGE_KEY, baseUrl);
  } catch {
    // The dashboard remains usable with its default when storage is unavailable.
  }
}

async function loadCompatibleBackend(candidateBaseUrl) {
  const [health, metadata] = await Promise.all([
    apiGet(candidateBaseUrl, '/health'),
    apiGet(candidateBaseUrl, '/metadata'),
  ]);
  if (health.status !== 'ok') {
    throw new Error('The backend did not report a healthy status.');
  }
  return metadata;
}

function autoLoadRequest(key, loader) {
  if (!AUTO_LOAD_REQUESTS.has(key)) {
    AUTO_LOAD_REQUESTS.set(key, loader().catch((error) => {
      AUTO_LOAD_REQUESTS.delete(key);
      throw error;
    }));
  }
  return AUTO_LOAD_REQUESTS.get(key);
}

const PAGES = [
  { id: 'simulation', label: 'Simulation', icon: Waves },
  { id: 'plants', label: 'Plants', icon: Factory },
  { id: 'sysid', label: 'SysID', icon: Sigma },
  { id: 'logging', label: 'Logging rate', icon: LineChart },
  { id: 'excitation', label: 'Excitation', icon: BarChart3 },
  { id: 'noiseLpf', label: 'Noise-aware logging (LPF)', icon: LineChart },
  { id: 'damping', label: 'Closed-loop damping', icon: Gauge },
  { id: 'retuning', label: 'Retuning (§4)', icon: RefreshCw },
  { id: 'equations', label: 'Equation', icon: Calculator },
  { id: 'drift', label: 'Drift', icon: GitCompare },
];

const EQUATION_SECTIONS = [
  {
    title: 'Logging Adequacy',
    rows: [
      {
        label: 'Logging ratio',
        equation: 'r = T_log / tau_min',
        note: 'Normalizes the selected logging period by the fastest plant time constant.',
      },
      {
        label: 'Reference power law',
        equation: 'MARE_ref(r) = MARE_ref(r0) x (r / r0)^alpha',
        note: 'Paper reference curve used for the NF and SN comparison lines.',
      },
      {
        label: 'Dashboard median MARE_theta',
        equation: 'median_MARE(T_log) = median_p(100 x mean_i |(theta_hat_i - theta_i) / theta_i|)',
        note: 'Median across plant runs at each logging period.',
      },
      {
        label: 'Logging difference',
        equation: 'difference = median_MARE - reference_MARE',
        note: 'Shown in the Median MARE rows and graph-point CSV.',
      },
      {
        label: 'Parameter error',
        equation: 'MARE_theta = mean_i |(theta_hat_i - theta_i) / theta_i|',
        note: 'Parameter-identification error used by the logging adequacy study.',
      },
      {
        label: 'Measurement conditions',
        equation: 'noise-free | tension-only | dual-channel',
        note: 'Every result is scoped to one of three conditions. Tension-only puts noise on the load cell alone; dual-channel adds velocity noise, which is what a real line has. The interior logging optimum is produced by the VELOCITY channel; tension noise alone never creates one.',
      },
      {
        label: 'Sensor noise model',
        equation: 'sigma_T = pct_T x T_max; sigma_v = pct_v x v_max, v_max = v0 / 0.30',
        note: 'Additive, zero-mean, Gaussian, on the sensor channels only, never in the ODE right-hand side. sigma_v is a surface-speed error injected as the SAME speed error on every roller, so a roller of radius R sees sigma_v / R rad/s. Static bias is not swept: temporal differencing cancels a constant offset exactly.',
      },
    ],
  },
  {
    title: 'Excitation',
    rows: [
      {
        label: 'Input profiles',
        equation: 'T_ref(t) = T_ref,0 + Delta T_ref(t)',
        note: 'ET1, ET3, ET6, ET3M, and E_Toggle modify the tension setpoint; EV1 applies the v0 to 1.2v0 midrun line-speed step. EVR is excluded.',
      },
      {
        label: 'Tension dynamics',
        equation: 'dT_i/dt = (EA / L_i)(v_i - v_{i-1}) + (T_{i-1}v_{i-1} - T_i v_i) / L_i',
        note: 'R2R web-span tension equation used in the excitation simulation.',
      },
      {
        label: 'Roller velocity dynamics',
        equation: 'dv_i/dt = (R_i^2 / J_i)(T_{i+1} - T_i) - (f_i / J_i)v_i + (R_i / J_i)u_i',
        note: 'Maps tension load, friction, and excitation torque into roller speed change.',
      },
      {
        label: 'RK4 state update',
        equation: 'x_{k+1} = x_k + (dt / 6)(k1 + 2k2 + 2k3 + k4)',
        note: 'Numerical integration used to simulate each plant and excitation strategy.',
      },
      {
        label: 'Prediction-error objective',
        equation: 'J(theta) = sum_k || W (z[k+1] - z_hat[k+1|k; theta]) ||_2^2',
        note: 'Paper Eq. 8 operating-point-weighted one-step PEM. z is the six-channel logged state (three span tensions and three roller angular velocities), not the tension-only controlled output y. EV1 omits the prediction pair that crosses the midrun line-speed step.',
      },
      {
        label: 'Operating-point weighting',
        equation: 'W = diag(1/T_ref,1..3, 1/omega_ss,UW, 1/omega_ss,Nip, 1/omega_ss,RW)',
        note: 'Each tension residual is divided by its set-point tension and each velocity residual by its steady-state angular velocity, so both channels land on a common percent-of-operating-point scale. The weight uses the controller set-point T_ref, not the load-cell full scale T_max that defines the noise model, and the exact equilibrium omega_ss, not v0/R. It is parameter-free and needs no noise model.',
      },
      {
        label: 'Multi-condition cost',
        equation: 'J_multi(theta) = sum_c sum_k || W^(c) (z^(c)[k] - z_hat^(c)[k]) ||_2^2',
        note: 'ET3M and the two-operating-point variant log their operating points as separate experiments and identify them in ONE joint fit, each condition carrying its own W. This is not an average of separate per-operating-point fits.',
      },
      {
        label: 'Fisher information',
        equation: 'F = sum_k J_k^T W^2 J_k, with J_k = d z_hat_k / d theta',
        note: 'The weight enters squared because it multiplies the residual once, so the Jacobian of the weighted residual is W J_k. On this scale the absolute value of kappa(F) depends on the operating-point normalization, so only the ordering across plants or excitations is interpretable: rank, do not quote.',
      },
      {
        label: 'Initialization',
        equation: 'theta_init = 1.01 x theta_true (UW/RW symmetrized); bounds [theta_init/10, 10 theta_init]',
        note: 'The starting point is fixed and is NOT an experimental axis. There is no alpha sweep over {1.5, 2, 5, 10, 50}. The optimizer is scipy least_squares (trust-region reflective) only.',
      },
      {
        label: 'Corrected controller protocol',
        equation: 'T_ctrl = 1 ms; T_I = max(0.1 s, 5 mean(T_ref) / mean(|u_ss|))',
        note: 'Each plant uses its professor-resolved integral time, a nominal-tension/current-line-speed steady baseline, and the unclamped paper velocity correction.',
      },
      {
        label: 'Excitation comparison',
        equation: 'delta_strategy = dashboard_MARE_theta - paper_MARE_theta',
        note: 'Used in the Excitation comparison table and summary CSV.',
      },
    ],
  },
  {
    title: 'Noise-Aware Logging (LPF)',
    rows: [
      {
        label: 'Sensor-noise measurement',
        equation: 'y_meas[k] = y[k] + eta[k]',
        note: 'The representative dual-channel dose is 0.3%/0.3% of full scale. The measured campaign grid is 0.02-0.5%.',
      },
      {
        label: 'LPF coefficient',
        equation: 'alpha = 1 - exp(-2*pi*f_c*dt)',
        note: 'Impulse-invariant first-order anti-alias coefficient evaluated at the 1 ms plant step before logging subsampling.',
      },
      {
        label: 'Filtered signal',
        equation: 'y_f[k] = alpha*y_meas[k] + (1 - alpha)*y_f[k-1]',
        note: 'Filtered tension samples are used for the LPF logging comparison.',
      },
      {
        label: 'Paper-dashboard error',
        equation: 'Error(%) = |MARE_paper - MARE_dashboard| / MARE_paper * 100',
        note: 'Pass/fail check used in the Noise-aware logging comparison table.',
      },
      {
        label: 'Recommendation rule',
        equation: 'Dual-channel: T_log = 5-20 ms and f_c >= 50 Hz. Noise-free: tau_min / T_log >= 5.',
        note: 'The pooled interior optimum sits at 5 ms, but its position is not universal - ET1 bottoms out at 20 ms while the others settle at 5-10 ms. Under tension-only noise there is no interior optimum at all: the finest 1 ms setting is best.',
      },
      {
        label: 'Anti-alias filter as a feasibility gate',
        equation: 'f_c <= 20 Hz: 67-100% convergence failure. f_c >= 50 Hz: converges.',
        note: '50 Hz is the WORKING cutoff, not a lower bound to be exceeded - raising it to 100 Hz makes things worse (25.9% vs 30.8% at E_Toggle, 20 ms, tension-only 0.3%). The gate is an over-aggressive low cutoff, not the absence of a filter: unfiltered logging also converges. Among converged runs the main-effect spread is T_log 74 pp > noise amplitude 30 pp ~ excitation 29 pp > cutoff 9 pp, and the factors interact too strongly for any single additive ranking.',
      },
      {
        label: 'Safe no-filter logging boundary',
        equation: 'unfiltered: 20 ms sits at 2.5-3.0x noise-free; 50 ms at 1.2x',
        note: 'The safe no-filter boundary is 50 ms. Applying a >= 50 Hz filter brings 20 ms back inside the 2x tolerance (1.6-1.8x).',
      },
    ],
  },
  {
    title: 'Closed-Loop Damping',
    rows: [
      {
        label: 'Normalized gain',
        equation: 'Kp* = normalized proportional gain',
        note: 'Kp* is the practical tuning knob used in the Closed-loop damping gain sweep.',
      },
      {
        label: 'Closed-loop damping ratio',
        equation: 'zeta_i = -Re(lambda_i) / |lambda_i|',
        note: 'Damping is calculated from the continuous nine-state Jacobian of the deployed outer tension PI, inner velocity P, feedforward, and plant dynamics.',
      },
      {
        label: 'Minimum damping',
        equation: 'zeta_CL,min = min_i(zeta_i)',
        note: 'Closed-loop damping shows this indicator is compressed under cascade PI plus feedforward.',
      },
      {
        label: 'Parameter error',
        equation: 'MARE_theta = mean_i |(theta_hat_i - theta_i) / theta_i| x 100',
        note: 'Paper-vs-dashboard comparisons use MARE_theta for NF and SN cases.',
      },
      {
        label: 'Validation error',
        equation: 'Error(%) = |MARE_theta,paper - MARE_theta,dashboard| / MARE_theta,paper x 100',
        note: 'Acceptance checks compare dashboard rows to the Closed-loop damping reference values.',
      },
    ],
  },
  {
    title: 'Drift Equation',
    rows: [
      {
        label: 'Fixed drift scenario',
        equation: 'p_drift = s_d p_0 for the full identification run',
        note: 'Each Section 3.3 point is one fixed perturbed plant; controller settings remain at their pre-drift values.',
      },
      {
        label: 'Drifted parameters',
        equation: 'EA_d = s_EA EA0, f_i,d = s_f f_i0, J_i,d = s_J J_i0',
        note: 'The three scenarios isolate elastic stiffness drift, friction drift, and inertia drift.',
      },
      {
        label: 'Seven-parameter drift error',
        equation: 'MARE_theta = (1/7) sum_{i=1}^7 |(theta_hat_i,d - theta_i,0) / theta_i,0|',
        note: 'Each independently simulated post-drift estimate is scored against the pre-drift baseline vector using the arithmetic mean of all seven parameters.',
      },
      {
        label: 'Median SysID drift error',
        equation: 'median_MARE_theta(d, c) = median_{plant,seed}(100 x MARE_theta(d, c))',
        note: 'NF contains one fresh simulation per plant and SN contains three freshly simulated noise seeds per plant.',
      },
      {
        label: 'Fresh simulation and Eq. 7 PEM',
        equation: 'plant simulation -> filtered logs -> nonlinear one-step PEM -> median',
        note: 'Dashboard values are recalculated from the model with a 1 ms controller and the weighted PEM. Paper values are comparison-only and never supply a dashboard result. The paper publishes the EA family as a band across its five legs rather than per leg, so those reference cells are intentionally empty.',
      },
      {
        label: 'Drift controller and logging protocol',
        equation: 'T_s = 1 ms; T_log = 20 ms; LPF 100 Hz; tension-only; T_I = auto_Ti; velocity clamp = none',
        note: 'The drift campaign keeps its tension-only acquisition at 100 Hz. That is deliberately NOT the representative dual-channel condition used for the excitation table, so the two must not be compared cell by cell. The per-roller diagnostic reads a change from the plant own pre-drift profile, which is not flat: under this acquisition the nip already carries the largest stiffness error with no drift.',
      },
      {
        label: 'Paper comparison difference',
        equation: 'difference_c = dashboard_MARE_theta,c - paper_MARE_theta,c',
        note: 'The comparison table and CSV report this difference for both NF and SN.',
      },
      {
        label: 'Asymmetric J drift',
        equation: 'J_UW -> 0.7J_UW or 0.5J_UW; J_RW -> 1.5J_RW or 2.0J_RW',
        note: 'The Drift tab uses the two paper-domain asymmetric J cases: UW -30%, RW +50% and UW -50%, RW +100%.',
      },
    ],
  },
];

function Field({ label, value, onChange, type = 'number', step = '0.01' }) {
  return (
    <label className="field">
      <span>{label}</span>
      <input type={type} value={value} step={step} onChange={(event) => onChange(event.target.value)} />
    </label>
  );
}

function SelectField({ label, value, onChange, options }) {
  const normalizedOptions = options.map((option) =>
    typeof option === 'string' ? { value: option, label: option } : option,
  );
  return (
    <label className="field">
      <span>{label}</span>
      <select value={value} onChange={(event) => onChange(event.target.value)}>
        {normalizedOptions.map((option) => (
          <option value={option.value} key={option.value}>
            {option.label}
          </option>
        ))}
      </select>
    </label>
  );
}

function PlantPage({ plants, selectedPlantId, onSelect }) {
  const selectedPlant = plants.find((plant) => plant.plant_id === selectedPlantId) ?? plants[0];
  const options = plants.map((plant) => ({ value: plant.plant_id, label: plant.label }));
  if (!selectedPlant) {
    return <section className="panel empty-panel">No plant data loaded.</section>;
  }

  const selectedSummary = {
    plant_id: selectedPlant.plant_id,
    material: selectedPlant.material,
    scale: selectedPlant.scale,
    regime: selectedPlant.regime,
    EA_N: selectedPlant.EA_N,
    v_ref_m_s: selectedPlant.v_ref_m_s,
    T_ref_N: selectedPlant.T_ref_N,
    T_max_N: selectedPlant.T_max_N,
    sensor_noise_sigma_N: selectedPlant.sensor_noise_sigma_N,
    zeta_cl_min: selectedPlant.zeta_cl_min,
    overshoot_percent: selectedPlant.overshoot_percent,
  };

  const selectedParameters = {
    EA_N: selectedPlant.EA_N,
    v_ref_m_s: selectedPlant.v_ref_m_s,
    T_ref_N: selectedPlant.T_ref_N,
    T_max_N: selectedPlant.T_max_N,
    sensor_noise_sigma_N: selectedPlant.sensor_noise_sigma_N,
    recommended_excitation_amplitude_V: selectedPlant.recommended_excitation_amplitude_V,
    baseline_range_compatible: selectedPlant.baseline_range_compatible,
    roller_radius_m: selectedPlant.roller_radius_m?.join(', '),
    span_length_m: selectedPlant.span_length_m?.join(', '),
    inertia_kg_m2: selectedPlant.inertia_kg_m2?.join(', '),
    viscous_friction: selectedPlant.viscous_friction?.join(', '),
    process_noise_b: selectedPlant.process_noise_b,
  };

  return (
    <section className="plant-layout">
      <div className="panel controls-panel compact-controls">
        <SelectField label="Selected plant" value={selectedPlant.plant_id} options={options} onChange={onSelect} />
        <p className="plant-note">
          The selected plant is sent to Simulation, SysID, Logging rate, Excitation, Noise-aware logging, Closed-loop damping, and Drift.
        </p>
      </div>
      <div className="panel plant-panel">
        <div className="plant-heading">
          <div>
            <h2>{selectedPlant.label}</h2>
            <span>Supplement Table S12 plant preset</span>
          </div>
        </div>
        <div className="plant-detail-grid">
          <MetricTable rows={selectedSummary} />
          <MetricTable rows={selectedParameters} />
        </div>
        <p className="plant-note">
          Professor-resolved plant values are now used directly: scalar `EA_N`, `v_ref`, `T_ref`, `T_max`, per-roller
          `R`, `J`, `f`, and span lengths `L`. The supplement `b_i` typo is treated as viscous friction `f_i`.
        </p>
        <h2>All Plants</h2>
        <MetricTable rows={plants} />
      </div>
    </section>
  );
}

function ExcitationChart({ rows }) {
  if (!rows.length) {
    return <div className="empty-panel compact-empty">Run the excitation study to calculate the chart.</div>;
  }
  const chartWidth = 920;
  const chartHeight = 430;
  const margin = { top: 54, right: 24, bottom: 74, left: 72 };
  const plotWidth = chartWidth - margin.left - margin.right;
  const plotHeight = chartHeight - margin.top - margin.bottom;
  const rowMax = Math.max(
    ...rows.flatMap((row) => [
      Number(row.dashboard_NF_percent ?? 0),
      Number(row.paper_NF_percent ?? 0),
      Number(row.dashboard_SN_percent ?? 0),
      Number(row.paper_SN_percent ?? 0),
    ]),
    50,
  );
  const maxValue = Math.ceil((rowMax * 1.12) / 10) * 10;
  const ticks = Array.from({ length: Math.floor(maxValue / 10) + 1 }, (_, index) => index * 10);
  const groupWidth = plotWidth / rows.length;
  const barWidth = Math.min(22, groupWidth * 0.16);
  const y = (value) => margin.top + plotHeight - (value / maxValue) * plotHeight;

  return (
    <div className="chart-frame" role="img" aria-label="Dashboard and paper median MARE theta by excitation strategy under NF and SN">
      <svg viewBox={`0 0 ${chartWidth} ${chartHeight}`} className="excitation-chart">
        {ticks.map((tick) => (
          <g key={tick}>
            <line
              x1={margin.left}
              x2={chartWidth - margin.right}
              y1={y(tick)}
              y2={y(tick)}
              className={tick === 0 ? 'chart-axis' : 'chart-grid'}
            />
            <text x={margin.left - 14} y={y(tick) + 5} textAnchor="end" className="chart-tick">
              {tick}
            </text>
          </g>
        ))}
        <line x1={margin.left} x2={margin.left} y1={margin.top} y2={margin.top + plotHeight} className="chart-axis" />
        <text x={24} y={margin.top + plotHeight / 2} className="chart-axis-label" transform={`rotate(-90 24 ${margin.top + plotHeight / 2})`}>
          Median MARE_theta (%)
        </text>
        {rows.map((row, index) => {
          const center = margin.left + groupWidth * index + groupWidth / 2;
          const nfValue = Number(row.dashboard_NF_percent ?? 0);
          const paperNfValue = Number(row.paper_NF_percent ?? 0);
          const snValue = Number(row.dashboard_SN_percent ?? 0);
          const paperSnValue = Number(row.paper_SN_percent ?? 0);
          const nfHeight = margin.top + plotHeight - y(nfValue);
          const paperNfHeight = margin.top + plotHeight - y(paperNfValue);
          const snHeight = margin.top + plotHeight - y(snValue);
          const paperSnHeight = margin.top + plotHeight - y(paperSnValue);
          return (
            <g key={row.strategy}>
              <rect x={center - 2 * barWidth - 5} y={y(nfValue)} width={barWidth} height={nfHeight} className="bar-nf" />
              <rect x={center - barWidth - 2} y={y(paperNfValue)} width={barWidth} height={paperNfHeight} className="bar-nf-paper" />
              <rect x={center + 2} y={y(snValue)} width={barWidth} height={snHeight} className="bar-sn" />
              <rect x={center + barWidth + 5} y={y(paperSnValue)} width={barWidth} height={paperSnHeight} className="bar-sn-paper" />
              {[
                { x: center - 1.5 * barWidth - 5, value: nfValue, label: "Dashboard NF" },
                { x: center - barWidth / 2 - 2, value: paperNfValue, label: "Paper NF" },
                { x: center + barWidth / 2 + 2, value: snValue, label: "Dashboard SN" },
                { x: center + 1.5 * barWidth + 5, value: paperSnValue, label: "Paper SN" },
              ].map((item) => {
                const labelY = Math.max(margin.top + 12, y(item.value) - 7);
                return (
                  <text
                    key={item.label}
                    x={item.x}
                    y={labelY}
                    textAnchor="start"
                    transform={`rotate(-62 ${item.x} ${labelY})`}
                    className="bar-label excitation-bar-label"
                  >
                    {item.value.toFixed(1)}%
                  </text>
                );
              })}
              <text x={center} y={margin.top + plotHeight + 28} textAnchor="middle" className="chart-category">
                {row.strategy}
              </text>
            </g>
          );
        })}
        <text x={margin.left + plotWidth / 2} y={chartHeight - 18} textAnchor="middle" className="chart-axis-label">
          Excitation strategy
        </text>
        <g transform={`translate(${margin.left + 12} ${margin.top + 8})`}>
          <rect width="510" height="30" className="chart-legend-box" />
          <rect x="10" y="9" width="14" height="12" className="bar-nf" />
          <text x="30" y="20" className="chart-legend-text">Dashboard NF</text>
          <rect x="130" y="9" width="14" height="12" className="bar-nf-paper" />
          <text x="150" y="20" className="chart-legend-text">Paper NF (MARE)</text>
          <rect x="275" y="9" width="14" height="12" className="bar-sn" />
          <text x="295" y="20" className="chart-legend-text">Dashboard SN</text>
          <rect x="405" y="9" width="14" height="12" className="bar-sn-paper" />
          <text x="425" y="20" className="chart-legend-text">Paper SN (MARE)</text>
        </g>
      </svg>
    </div>
  );
}

function ExcitationPage({
  baseUrl,
  plantOptions,
  excitationOptions,
}) {
  const [autoLoaded, setAutoLoaded] = useState(false);
  const [showComparison, setShowComparison] = useState(false);
  const [excitationPlantId, setExcitationPlantId] = useState('ALL');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const summaryRows = Array.isArray(result?.metrics?.comparison_rows) ? result.metrics.comparison_rows : [];
  const plantRows = Array.isArray(result?.metrics?.raw_rows) ? result.metrics.raw_rows : [];
  const expectedPlantCount = Number(result?.metrics?.plant_count ?? (excitationPlantId === 'ALL' ? 10 : 1));
  const bestNF = summaryRows.length
    ? summaryRows.reduce((best, row) => (Number(row.dashboard_NF_percent) < Number(best.dashboard_NF_percent) ? row : best), summaryRows[0])
    : null;
  const bestSN = summaryRows.length
    ? summaryRows.reduce((best, row) => (Number(row.dashboard_SN_percent) < Number(best.dashboard_SN_percent) ? row : best), summaryRows[0])
    : null;
  const allPlantsValid = summaryRows.length > 0 && summaryRows.every(
    (row) => Number(row.valid_plant_count_NF) === expectedPlantCount && Number(row.valid_plant_count_SN) === expectedPlantCount,
  );
  const csvUrl = artifactUrl(baseUrl, result?.csv_url);
  const plantCsvUrl = artifactUrl(baseUrl, result?.raw_csv_url);
  const resultRows = summaryRows.map((row) => ({
    strategy: row.strategy,
    channels: row.channels,
    NF_dashboard_percent: Number(Number(row.displayed_dashboard_NF_percent).toFixed(2)),
    NF_paper_percent: row.paper_NF_percent,
    NF_difference_percent: Number(Number(row.difference_NF_percent).toFixed(2)),
    NF_valid_plants: `${Number(row.valid_plant_count_NF)}/${expectedPlantCount}`,
    SN_dashboard_percent: Number(Number(row.displayed_dashboard_SN_percent).toFixed(2)),
    SN_paper_percent: row.paper_SN_percent,
    SN_difference_percent: Number(Number(row.difference_SN_percent).toFixed(2)),
    SN_valid_plants: `${Number(row.valid_plant_count_SN)}/${expectedPlantCount}`,
    __provenance: row.__provenance,
  }));
  const comparisonRows = summaryRows.map((row) => {
    const nfDelta = Number(Number(row.difference_NF_percent).toFixed(2));
    const snDelta = Number(Number(row.difference_SN_percent).toFixed(2));
    return {
      strategy: row.strategy,
      NF_dashboard_percent: Number(Number(row.displayed_dashboard_NF_percent).toFixed(2)),
      NF_paper_percent: row.paper_NF_percent,
      NF_difference_percent: nfDelta,
      NF_match: Math.abs(nfDelta) <= 0.5 ? 'similar' : 'check',
      NF_valid_plants: `${Number(row.valid_plant_count_NF)}/${expectedPlantCount}`,
      SN_dashboard_percent: Number(Number(row.displayed_dashboard_SN_percent).toFixed(2)),
      SN_paper_percent: row.paper_SN_percent,
      SN_difference_percent: snDelta,
      SN_match: Math.abs(snDelta) <= 0.5 ? 'similar' : 'check',
      SN_valid_plants: `${Number(row.valid_plant_count_SN)}/${expectedPlantCount}`,
      __provenance: row.__provenance,
    };
  });
  const activePlant = plantOptions.find((plant) => plant.value === excitationPlantId);

  async function runExcitationStudy() {
    setLoading(true);
    setError('');
    try {
      const payload = await apiPost(baseUrl, '/validate/excitation', { plant_id: excitationPlantId, force_rerun: true });
      setResult(payload);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    if (autoLoaded) return;
    setAutoLoaded(true);
    setLoading(true);
    autoLoadRequest(
      `excitation:${baseUrl}`,
      () => apiPost(baseUrl, '/validate/excitation', { plant_id: 'ALL' }),
    )
      .then(setResult)
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }, [autoLoaded, baseUrl]);

  return (
    <section className="excitation-layout">
      {error && <ErrorBanner message={error} />}
      <div className="panel controls-panel compact-controls">
        <div className="section-heading">
          <BarChart3 size={17} />
          <h2>Excitation Controls</h2>
        </div>
        <div className="field-grid">
          {plantOptions.length > 0 && (
            <SelectField label="Plant scope" value={excitationPlantId} options={plantOptions} onChange={setExcitationPlantId} />
          )}
          <p className="plant-note">
            {excitationPlantId === 'ALL'
              ? 'Reruns all professor-resolved plants with the 1 ms paper controller and the Eq. 8 weighted one-step PEM, then reports the median MARE.'
              : `Reruns ${activePlant?.label ?? 'the selected plant'} with the corrected controller and one-step PEM protocol.`}
          </p>
          <RunButton loading={loading} onClick={runExcitationStudy}>Run Excitation Study</RunButton>
        </div>
      </div>
      <div className="panel excitation-panel">
        <div className="excitation-summary">
          <div>
            <h2>Backend Equation Validation</h2>
            <p className="plant-note">
              The bars are fresh backend results: RK4 simulation with a 1 ms controller, logged z = [T1, T2, T3, omega_UW, omega_Nip, omega_RW],
              paper Eq. 8 operating-point-weighted one-step PEM on the six-channel logged state z, then MARE_theta = 100*mean(abs(relative error)).
              Sensor noise is filtered at the 1 ms plant rate before both control and logging. EV1 is estimated on segments split at its midrun speed step,
              so the corrected ET1 and EV1 SN values come from their actual excitation protocols rather than a cross-step residual or reference copy.
            </p>
          </div>
          <div className="summary-metrics">
            <span>NF best: {bestNF ? `${bestNF.strategy} (${Number(bestNF.dashboard_NF_percent).toFixed(1)}%)` : 'n/a'}</span>
            <span>SN best: {bestSN ? `${bestSN.strategy} (${Number(bestSN.dashboard_SN_percent).toFixed(1)}%)` : 'n/a'}</span>
            <span>
              Coverage: {allPlantsValid
                ? `${expectedPlantCount}/${expectedPlantCount} ${expectedPlantCount === 1 ? 'plant' : 'plants'} for every NF and SN row`
                : 'incomplete plant coverage'}
            </span>
          </div>
        </div>
        <div className="action-row">
          <button
            className={showComparison ? 'primary-button' : 'icon-text-button'}
            type="button"
            onClick={() => setShowComparison((current) => !current)}
          >
            <BarChart3 size={16} />
            Comparison
          </button>
          {resultRows.length > 0 && (
            <button className="primary-button" type="button" onClick={() => downloadCsv('excitation_summary_dashboard_vs_paper.csv', resultRows)}>
              <Download size={16} />
              Download summary CSV
            </button>
          )}
          {plantRows.length > 0 && (
            <button className="icon-text-button" type="button" onClick={() => downloadCsv('excitation_plant_runs.csv', plantRows)}>
              <Download size={16} />
              Plant-run CSV
            </button>
          )}
          {csvUrl && (
            <a className="icon-link" href={csvUrl} target="_blank" rel="noreferrer">
              <Download size={16} />
              Backend CSV
            </a>
          )}
          {plantCsvUrl && (
            <a className="icon-link" href={plantCsvUrl} target="_blank" rel="noreferrer">
              <Download size={16} />
              Backend plant CSV
            </a>
          )}
        </div>
        <ExcitationChart rows={summaryRows} />
      </div>
      {showComparison && (
        <div className="panel excitation-table-panel comparison-panel">
          <div>
            <h2>Comparison</h2>
            <p className="plant-note">Dashboard result is compared with the v5 paper MARE. Every v5 number was recomputed under the weighted estimator, so these are not the v4.1 values under a new label. Difference = dashboard - paper.</p>
          </div>
          <MetricTable rows={comparisonRows} />
        </div>
      )}
      <div className="panel excitation-table-panel">
        <h2>Calculated Data</h2>
        <MetricTable rows={resultRows} />
      </div>
    </section>
  );
}

function chartScale(value, inMin, inMax, outMin, outMax) {
  if (Math.abs(inMax - inMin) < 1e-12) return 0.5 * (outMin + outMax);
  return outMin + ((value - inMin) * (outMax - outMin)) / (inMax - inMin);
}

function NoiseLpfResult({ baseUrl, result, artifactVersion }) {
  if (!result || result.study !== 'noise-aware-logging-lpf') {
    return <section className="panel empty-panel">No noise-aware logging run selected.</section>;
  }

  const plots = Object.entries(result.plots ?? {});
  const status = result.metrics?.status ?? 'UNKNOWN';
  const summaryUrl = artifactUrl(baseUrl, result.summary_url);

  return (
    <section className="noise-lpf-page">
      <div className="panel noise-lpf-summary">
        <div>
          <h2>Noise-Aware Logging (LPF)</h2>
          <span className={`validation-badge ${String(status).toLowerCase()}`}>{status}</span>
        </div>
        <MetricTable rows={result.metrics} />
        {summaryUrl && (
          <a className="icon-link" href={summaryUrl} target="_blank" rel="noreferrer">
            Summary
          </a>
        )}
      </div>

      <div className="noise-lpf-plot-grid">
        {plots.map(([key, plot]) => {
          const url = cacheBustUrl(artifactUrl(baseUrl, plot.url), artifactVersion);
          if (!url) return null;
          return (
            <article className="panel noise-lpf-plot-panel" key={key}>
              <h2>{plot.title}</h2>
              <div className="noise-lpf-plot-frame">
                <img src={url} alt={plot.title} />
              </div>
            </article>
          );
        })}
      </div>

      <div className="noise-lpf-table-grid">
        <section className="panel">
          <h2>Comparison Table</h2>
          <MetricTable rows={result.comparison_table} />
        </section>
        <section className="panel">
          <h2>Acceptance Criteria</h2>
          <MetricTable rows={result.acceptance_criteria} />
        </section>
        <section className="panel">
          <h2>Filter Configurations</h2>
          <MetricTable rows={result.filter_configurations} />
        </section>
        <section className="panel">
          <h2>Tlog Sweep Data</h2>
          <MetricTable rows={result.tlog_sweep} />
        </section>
        <section className="panel noise-lpf-wide">
          <h2>Heatmap Comparison Data</h2>
          <MetricTable rows={result.heatmap_comparison} />
        </section>
      </div>
    </section>
  );
}

function DampingStepChart({ rows, metrics = [] }) {
  if (!rows.length) return <div className="empty-panel compact-empty">Run closed-loop damping validation to calculate the step response.</div>;
  const chartWidth = 920;
  const chartHeight = 430;
  const margin = { top: 28, right: 28, bottom: 64, left: 66 };
  const plotWidth = chartWidth - margin.left - margin.right;
  const plotHeight = chartHeight - margin.top - margin.bottom;
  const numericValues = rows
    .map((row) => Number(row.normalized_tension))
    .filter((value) => Number.isFinite(value));
  const timeValues = rows
    .map((row) => Number(row.time_s))
    .filter((value) => Number.isFinite(value));
  const yMin = Math.min(0.95, Math.floor((Math.min(...numericValues) - 0.03) * 20) / 20);
  const yMax = Math.max(1.25, Math.ceil((Math.max(...numericValues) + 0.03) * 20) / 20);
  const xMax = Math.max(...timeValues, 4);
  const x = (value) => chartScale(value, 0, xMax, margin.left, chartWidth - margin.right);
  const y = (value) => chartScale(value, yMin, yMax, margin.top + plotHeight, margin.top);
  const metricByGain = Object.fromEntries(metrics.map((row) => [String(row.kp_star), row]));
  const series = [
    { key: 'Tref', label: 'Tref/Tref,0', color: '#20262a', dash: '8 7' },
    { key: 50, label: `Kp*=50 (${formatNumber(Number(metricByGain['50']?.t90_s) * 1000, 1)} ms)`, color: '#2f6fbb' },
    { key: 100, label: `Kp*=100 (${formatNumber(Number(metricByGain['100']?.t90_s) * 1000, 1)} ms)`, color: '#198d6b' },
    { key: 200, label: `Kp*=200 (${formatNumber(Number(metricByGain['200']?.t90_s) * 1000, 1)} ms)`, color: '#d7372f' },
  ];
  const yTicks = Array.from({ length: 5 }, (_, index) => yMin + (index * (yMax - yMin)) / 4);
  const xTicks = Array.from({ length: Math.floor(xMax) + 1 }, (_, index) => index);
  const kp200Metric = metricByGain['200'];
  const overshootX = x(Number(kp200Metric?.peak_time_s ?? 1.03));
  const overshootY = y(Number(kp200Metric?.peak_normalized_tension ?? 1.30));

  return (
    <div className="chart-frame" role="img" aria-label="Normalized tension step response by Kp star">
      <svg viewBox={`0 0 ${chartWidth} ${chartHeight}`} className="damping-chart">
        {yTicks.map((tick) => (
          <g key={tick}>
            <line x1={margin.left} x2={chartWidth - margin.right} y1={y(tick)} y2={y(tick)} className="chart-grid" />
            <text x={margin.left - 12} y={y(tick) + 4} textAnchor="end" className="chart-tick">
              {tick.toFixed(2)}
            </text>
          </g>
        ))}
        {xTicks.map((tick) => (
          <g key={tick}>
            <line x1={x(tick)} x2={x(tick)} y1={margin.top} y2={margin.top + plotHeight} className="chart-grid subtle-grid" />
            <text x={x(tick)} y={margin.top + plotHeight + 28} textAnchor="middle" className="chart-tick">
              {tick}
            </text>
          </g>
        ))}
        <line x1={margin.left} x2={chartWidth - margin.right} y1={margin.top + plotHeight} y2={margin.top + plotHeight} className="chart-axis" />
        <line x1={margin.left} x2={margin.left} y1={margin.top} y2={margin.top + plotHeight} className="chart-axis" />
        <text x={chartWidth / 2} y={chartHeight - 20} textAnchor="middle" className="chart-axis-label">Time (s)</text>
        <text x="20" y={margin.top + plotHeight / 2} textAnchor="middle" className="chart-axis-label" transform={`rotate(-90 20 ${margin.top + plotHeight / 2})`}>
          Normalized tension
        </text>
        {series.map((item) => {
          const values = rows.filter((row) => String(row.kp_star) === String(item.key));
          const points = values.map((row) => `${x(Number(row.time_s)).toFixed(1)},${y(Number(row.normalized_tension)).toFixed(1)}`).join(' ');
          return <polyline key={item.key} fill="none" stroke={item.color} strokeWidth={item.key === 'Tref' ? 2 : 3} strokeDasharray={item.dash} points={points} />;
        })}
        <g transform={`translate(${chartWidth - 240} ${margin.top + 190})`}>
          <rect width="202" height="94" className="chart-legend-box" />
          {series.map((item, index) => (
            <g key={item.key} transform={`translate(12 ${17 + index * 20})`}>
              <line x1="0" x2="24" y1="0" y2="0" stroke={item.color} strokeWidth="3" strokeDasharray={item.dash} />
              <text x="32" y="5" className="chart-legend-text">{item.label}</text>
            </g>
          ))}
        </g>
        <g transform={`translate(${margin.left + 14} ${margin.top + 12})`}>
          <rect width="276" height="84" className="chart-inset-box" />
          <text x="10" y="22" className="rise-label rise-red">
            Kp*=200: t90 {formatNumber(Number(metricByGain['200']?.t90_s) * 1000, 2)} ms
          </text>
          <text x="10" y="42" className="rise-label rise-green">
            Kp*=100: t90 {formatNumber(Number(metricByGain['100']?.t90_s) * 1000, 2)} ms
          </text>
          <text x="10" y="62" className="rise-label rise-blue">
            Kp*=50: t90 {formatNumber(Number(metricByGain['50']?.t90_s) * 1000, 2)} ms
          </text>
        </g>
        <text x={overshootX + 12} y={overshootY - 12} className="overshoot-label">
          OS = {formatNumber(kp200Metric?.overshoot_percent, 1)}%
        </text>
      </svg>
    </div>
  );
}

function DampingPaperDashboardChart({ rows }) {
  if (!rows.length) return <div className="empty-panel compact-empty">No paper-vs-dashboard rows yet.</div>;
  const chartWidth = 980;
  const chartHeight = 430;
  const margin = { top: 30, right: 24, bottom: 86, left: 72 };
  const plotWidth = chartWidth - margin.left - margin.right;
  const plotHeight = chartHeight - margin.top - margin.bottom;
  const maxValue = Math.ceil((Math.max(...rows.flatMap((row) => [Number(row.paper_MARE_theta_percent), Number(row.dashboard_MARE_theta_percent)])) * 1.2) / 10) * 10;
  const y = (value) => chartScale(value, 0, maxValue, margin.top + plotHeight, margin.top);
  const groupWidth = plotWidth / rows.length;
  const barWidth = Math.min(28, groupWidth * 0.28);
  const ticks = Array.from({ length: Math.floor(maxValue / 10) + 1 }, (_, index) => index * 10);

  return (
    <div className="chart-frame" role="img" aria-label="Paper versus dashboard MARE theta comparison">
      <svg viewBox={`0 0 ${chartWidth} ${chartHeight}`} className="damping-chart">
        {ticks.map((tick) => (
          <g key={tick}>
            <line x1={margin.left} x2={chartWidth - margin.right} y1={y(tick)} y2={y(tick)} className={tick === 0 ? 'chart-axis' : 'chart-grid'} />
            <text x={margin.left - 12} y={y(tick) + 4} textAnchor="end" className="chart-tick">{tick}</text>
          </g>
        ))}
        <line x1={margin.left} x2={margin.left} y1={margin.top} y2={margin.top + plotHeight} className="chart-axis" />
        <text x="22" y={margin.top + plotHeight / 2} className="chart-axis-label" textAnchor="middle" transform={`rotate(-90 22 ${margin.top + plotHeight / 2})`}>
          MARE_theta (%)
        </text>
        {rows.map((row, index) => {
          const center = margin.left + groupWidth * index + groupWidth / 2;
          const paper = Number(row.paper_MARE_theta_percent);
          const dashboard = Number(row.dashboard_MARE_theta_percent);
          return (
            <g key={`${row.condition}-${row.kp_star}`}>
              <rect x={center - barWidth - 3} y={y(paper)} width={barWidth} height={margin.top + plotHeight - y(paper)} className="bar-paper" />
              <rect x={center + 3} y={y(dashboard)} width={barWidth} height={margin.top + plotHeight - y(dashboard)} className="bar-dashboard" />
              <text x={center} y={margin.top + plotHeight + 24} textAnchor="middle" className="chart-category">{row.condition}</text>
              <text x={center} y={margin.top + plotHeight + 44} textAnchor="middle" className="chart-category">Kp*={row.kp_star}</text>
              <text x={center - barWidth / 2 - 3} y={y(paper) - 8} textAnchor="middle" className="bar-label">{formatNumber(paper, 1)}</text>
              <text x={center + barWidth / 2 + 3} y={y(dashboard) - 8} textAnchor="middle" className="bar-label">{formatNumber(dashboard, 1)}</text>
            </g>
          );
        })}
        <g transform={`translate(${margin.left + 10} ${margin.top + 8})`}>
          <rect width="184" height="30" className="chart-legend-box" />
          <rect x="10" y="9" width="14" height="12" className="bar-paper" />
          <text x="30" y="20" className="chart-legend-text">Paper</text>
          <rect x="98" y="9" width="14" height="12" className="bar-dashboard" />
          <text x="118" y="20" className="chart-legend-text">Dashboard</text>
        </g>
      </svg>
    </div>
  );
}

function DampingSysIdChart({ rows }) {
  if (!rows.length) return <div className="empty-panel compact-empty">No SysID gain sweep rows yet.</div>;
  const chartWidth = 920;
  const chartHeight = 390;
  const margin = { top: 30, right: 28, bottom: 62, left: 70 };
  const plotWidth = chartWidth - margin.left - margin.right;
  const plotHeight = chartHeight - margin.top - margin.bottom;
  const maxValue = Math.ceil((Math.max(...rows.flatMap((row) => [Number(row.NF_MARE_theta_percent), Number(row.SN_MARE_theta_percent)])) * 1.18) / 5) * 5;
  const x = (kp) => chartScale(kp, 50, 200, margin.left, chartWidth - margin.right);
  const y = (value) => chartScale(value, 0, maxValue, margin.top + plotHeight, margin.top);
  const ticks = Array.from({ length: Math.floor(maxValue / 5) + 1 }, (_, index) => index * 5);
  const nfPoints = rows.map((row) => `${x(Number(row.kp_star)).toFixed(1)},${y(Number(row.NF_MARE_theta_percent)).toFixed(1)}`).join(' ');
  const snPoints = rows.map((row) => `${x(Number(row.kp_star)).toFixed(1)},${y(Number(row.SN_MARE_theta_percent)).toFixed(1)}`).join(' ');

  return (
    <div className="chart-frame" role="img" aria-label="SysID error versus Kp star">
      <svg viewBox={`0 0 ${chartWidth} ${chartHeight}`} className="damping-chart">
        {ticks.map((tick) => (
          <g key={tick}>
            <line x1={margin.left} x2={chartWidth - margin.right} y1={y(tick)} y2={y(tick)} className={tick === 0 ? 'chart-axis' : 'chart-grid'} />
            <text x={margin.left - 12} y={y(tick) + 4} textAnchor="end" className="chart-tick">{tick}</text>
          </g>
        ))}
        {[50, 100, 200].map((kp) => (
          <text key={kp} x={x(kp)} y={margin.top + plotHeight + 28} textAnchor="middle" className="chart-tick">{kp}</text>
        ))}
        <line x1={margin.left} x2={chartWidth - margin.right} y1={margin.top + plotHeight} y2={margin.top + plotHeight} className="chart-axis" />
        <line x1={margin.left} x2={margin.left} y1={margin.top} y2={margin.top + plotHeight} className="chart-axis" />
        <text x={chartWidth / 2} y={chartHeight - 18} textAnchor="middle" className="chart-axis-label">Kp*</text>
        <text x="22" y={margin.top + plotHeight / 2} textAnchor="middle" className="chart-axis-label" transform={`rotate(-90 22 ${margin.top + plotHeight / 2})`}>
          MARE_theta (%)
        </text>
        <polyline fill="none" stroke="#2f6f73" strokeWidth="3" points={nfPoints} />
        <polyline fill="none" stroke="#c8682f" strokeWidth="3" points={snPoints} />
        {rows.map((row) => (
          <g key={row.kp_star}>
            <circle cx={x(Number(row.kp_star))} cy={y(Number(row.NF_MARE_theta_percent))} r="5" fill="#2f6f73" />
            <circle cx={x(Number(row.kp_star))} cy={y(Number(row.SN_MARE_theta_percent))} r="5" fill="#c8682f" />
          </g>
        ))}
        <g transform={`translate(${margin.left + 12} ${margin.top + 8})`}>
          <rect width="122" height="30" className="chart-legend-box" />
          <line x1="10" x2="26" y1="15" y2="15" stroke="#2f6f73" strokeWidth="4" />
          <text x="34" y="20" className="chart-legend-text">NF</text>
          <line x1="70" x2="86" y1="15" y2="15" stroke="#c8682f" strokeWidth="4" />
          <text x="94" y="20" className="chart-legend-text">SN</text>
        </g>
      </svg>
    </div>
  );
}

function DampingRegimeChart({ rows }) {
  if (!rows.length) return <div className="empty-panel compact-empty">No regime comparison rows yet.</div>;
  const chartWidth = 980;
  const chartHeight = 450;
  const margin = { top: 38, right: 30, bottom: 72, left: 72 };
  const plotWidth = chartWidth - margin.left - margin.right;
  const plotHeight = chartHeight - margin.top - margin.bottom;
  const maxValue = 60;
  const y = (value) => chartScale(value, 0, maxValue, margin.top + plotHeight, margin.top);
  const groupWidth = plotWidth / rows.length;
  const barWidth = Math.min(24, groupWidth * 0.12);
  const ticks = [0, 20, 40, 60];

  return (
    <div className="chart-frame" role="img" aria-label="O-UD versus H-Damp paper and dashboard comparison">
      <svg viewBox={`0 0 ${chartWidth} ${chartHeight}`} className="damping-chart">
        {ticks.map((tick) => (
          <g key={tick}>
            <line x1={margin.left} x2={chartWidth - margin.right} y1={y(tick)} y2={y(tick)} className={tick === 0 ? 'chart-axis' : 'chart-grid'} />
            <text x={margin.left - 12} y={y(tick) + 4} textAnchor="end" className="chart-tick">{tick}</text>
          </g>
        ))}
        <line x1={margin.left} x2={margin.left} y1={margin.top} y2={margin.top + plotHeight} className="chart-axis" />
        <text x="22" y={margin.top + plotHeight / 2} textAnchor="middle" className="chart-axis-label" transform={`rotate(-90 22 ${margin.top + plotHeight / 2})`}>
          Median MARE_theta (%)
        </text>
        {rows.map((row, index) => {
          const center = margin.left + groupWidth * index + groupWidth / 2;
          const values = [
            { value: Number(row.paper_O_UD_MARE_theta_percent), className: 'bar-oud-paper', offset: -1.8, label: 'P O-UD' },
            { value: Number(row.dashboard_O_UD_MARE_theta_percent), className: 'bar-oud-dashboard', offset: -0.6, label: 'D O-UD' },
            { value: Number(row.paper_H_Damp_MARE_theta_percent), className: 'bar-hdamp-paper', offset: 0.6, label: 'P H-Damp' },
            { value: Number(row.dashboard_H_Damp_MARE_theta_percent), className: 'bar-hdamp-dashboard', offset: 1.8, label: 'D H-Damp' },
          ];
          const bracketY = y(Math.max(Number(row.paper_H_Damp_MARE_theta_percent), Number(row.dashboard_H_Damp_MARE_theta_percent)) + 7);
          return (
            <g key={row.kp_star}>
              {values.map((item) => (
                <g key={item.label}>
                  <rect
                    x={center + item.offset * (barWidth + 3) - barWidth / 2}
                    y={y(item.value)}
                    width={barWidth}
                    height={margin.top + plotHeight - y(item.value)}
                    className={item.className}
                  />
                  <text x={center + item.offset * (barWidth + 3)} y={y(item.value) - 7} textAnchor="middle" className="bar-label">
                    {formatNumber(item.value, 1)}
                  </text>
                </g>
              ))}
              <text x={center} y={margin.top + plotHeight + 28} textAnchor="middle" className="chart-category">Kp*={row.kp_star}</text>
              <path d={`M ${center - 46} ${bracketY} h 92 m -92 0 v 9 m 92 -9 v 9`} className="ratio-bracket" />
              <text x={center} y={bracketY - 5} textAnchor="middle" className="ratio-label">
                P {formatNumber(row.paper_H_Damp_to_O_UD_ratio, 1)}x · D {formatNumber(row.dashboard_H_Damp_to_O_UD_ratio, 1)}x
              </text>
              <path d={`M ${center} ${margin.top + 10} l 6 10 h -12 z`} className="median-marker" />
              <text x={center} y={margin.top + 5} textAnchor="middle" className="median-label">{row.plant_median_marker}</text>
            </g>
          );
        })}
        <g transform={`translate(${margin.left + 12} ${margin.top - 26})`}>
          <rect width="552" height="30" className="chart-legend-box" />
          <rect x="10" y="9" width="14" height="12" className="bar-oud-paper" />
          <text x="30" y="20" className="chart-legend-text">Paper O-UD</text>
          <rect x="116" y="9" width="14" height="12" className="bar-oud-dashboard" />
          <text x="136" y="20" className="chart-legend-text">Dashboard O-UD</text>
          <rect x="262" y="9" width="14" height="12" className="bar-hdamp-paper" />
          <text x="282" y="20" className="chart-legend-text">Paper H-Damp</text>
          <rect x="398" y="9" width="14" height="12" className="bar-hdamp-dashboard" />
          <text x="418" y="20" className="chart-legend-text">Dashboard H-Damp</text>
        </g>
      </svg>
    </div>
  );
}

function RetuningConvergenceChart({ conv }) {
  if (!conv || !conv.ours_cs?.length) {
    return <div className="empty-panel compact-empty">No convergence data in the campaign checkpoints.</div>;
  }
  const chartWidth = 920;
  const chartHeight = 400;
  const margin = { top: 24, right: 30, bottom: 58, left: 70 };
  const plotWidth = chartWidth - margin.left - margin.right;
  const plotHeight = chartHeight - margin.top - margin.bottom;
  const series = [
    { key: 'ours_cs', label: 'CS-BO ours', color: '#2f6fbb', dash: null },
    { key: 'paper_cs', label: 'CS-BO paper', color: '#2f6fbb', dash: '7 6' },
    { key: 'ours_ws', label: 'WS-BO ours', color: '#d7372f', dash: null },
    { key: 'paper_ws', label: 'WS-BO paper', color: '#d7372f', dash: '7 6' },
  ].filter((s) => (conv[s.key] ?? []).length > 0);
  const all = series.flatMap((s) => conv[s.key]).filter((v) => Number.isFinite(v) && v > 0);
  const logMin = Math.floor(Math.log10(Math.min(...all)));
  const logMax = Math.ceil(Math.log10(Math.max(...all)));
  const x = (evalIdx) => chartScale(evalIdx, 1, 30, margin.left, chartWidth - margin.right);
  const y = (v) => chartScale(Math.log10(Math.max(v, 1e-9)), logMin, logMax, margin.top + plotHeight, margin.top);
  const yTicks = [];
  for (let d = logMin; d <= logMax; d += 1) yTicks.push(10 ** d);
  const xTicks = [1, 5, 10, 15, 20, 25, 30];
  return (
    <div className="chart-frame" role="img" aria-label="Median best retuning cost against real-plant evaluations, dashboard against paper">
      <svg viewBox={`0 0 ${chartWidth} ${chartHeight}`} className="damping-chart">
        {yTicks.map((tick) => (
          <g key={tick}>
            <line x1={margin.left} x2={chartWidth - margin.right} y1={y(tick)} y2={y(tick)} className="chart-grid" />
            <text x={margin.left - 12} y={y(tick) + 4} textAnchor="end" className="chart-tick">{tick}</text>
          </g>
        ))}
        {xTicks.map((tick) => (
          <text key={tick} x={x(tick)} y={margin.top + plotHeight + 26} textAnchor="middle" className="chart-tick">{tick}</text>
        ))}
        <line x1={margin.left} x2={chartWidth - margin.right} y1={margin.top + plotHeight} y2={margin.top + plotHeight} className="chart-axis" />
        <line x1={margin.left} x2={margin.left} y1={margin.top} y2={margin.top + plotHeight} className="chart-axis" />
        <text x={chartWidth / 2} y={chartHeight - 16} textAnchor="middle" className="chart-axis-label">Real-plant evaluations</text>
        <text x="20" y={margin.top + plotHeight / 2} textAnchor="middle" className="chart-axis-label" transform={`rotate(-90 20 ${margin.top + plotHeight / 2})`}>
          Median best cost S (log)
        </text>
        {series.map((item) => {
          const points = conv[item.key]
            .map((v, i) => `${x(i + 1).toFixed(1)},${y(v).toFixed(1)}`)
            .join(' ');
          return <polyline key={item.key} fill="none" stroke={item.color} strokeWidth={2.5} strokeDasharray={item.dash ?? undefined} points={points} />;
        })}
        <g transform={`translate(${chartWidth - 236} ${margin.top + 10})`}>
          <rect width="206" height={16 + series.length * 20} className="chart-legend-box" />
          {series.map((item, index) => (
            <g key={item.key} transform={`translate(12 ${17 + index * 20})`}>
              <line x1="0" x2="24" y1="0" y2="0" stroke={item.color} strokeWidth="3" strokeDasharray={item.dash ?? undefined} />
              <text x="32" y="5" className="chart-legend-text">{item.label}</text>
            </g>
          ))}
        </g>
      </svg>
    </div>
  );
}

function RetuningFinalCostChart({ methods }) {
  if (!methods?.length) {
    return <div className="empty-panel compact-empty">No campaign results loaded.</div>;
  }
  const chartWidth = 920;
  const rowHeight = 54;
  const margin = { top: 18, right: 120, bottom: 52, left: 118 };
  const chartHeight = margin.top + methods.length * rowHeight + margin.bottom;
  const values = methods.flatMap((m) => [m.median, m.paper_median]).filter((v) => Number.isFinite(v) && v > 0);
  const logMin = Math.log10(Math.min(...values)) - 0.08;
  const logMax = Math.log10(Math.max(...values)) + 0.08;
  const x = (v) => chartScale(Math.log10(v), logMin, logMax, margin.left, chartWidth - margin.right);
  const ticks = [0.1, 0.2, 0.4, 0.6, 1.0].filter((t) => Math.log10(t) >= logMin && Math.log10(t) <= logMax);
  return (
    <div className="chart-frame" role="img" aria-label="Median final retuning cost per method, dashboard against paper">
      <svg viewBox={`0 0 ${chartWidth} ${chartHeight}`} className="damping-chart">
        {ticks.map((tick) => (
          <g key={tick}>
            <line x1={x(tick)} x2={x(tick)} y1={margin.top} y2={chartHeight - margin.bottom} className="chart-grid" />
            <text x={x(tick)} y={chartHeight - margin.bottom + 22} textAnchor="middle" className="chart-tick">{tick}</text>
          </g>
        ))}
        {methods.map((m, i) => {
          const y = margin.top + rowHeight * (i + 0.5);
          const xd = x(m.median);
          const xp = Number.isFinite(m.paper_median) ? x(m.paper_median) : null;
          return (
            <g key={m.method}>
              <text x={margin.left - 12} y={y + 4} textAnchor="end" className="chart-legend-text" fontWeight="600">{m.method}</text>
              {xp != null && <line x1={Math.min(xd, xp)} x2={Math.max(xd, xp)} y1={y} y2={y} stroke="#c9c2b6" strokeWidth="2.5" />}
              {xp != null && <circle cx={xp} cy={y} r="7" fill="#8a8578" />}
              {xp != null && (
                <text x={xp} y={y + 24} textAnchor="middle" className="chart-tick" fill="#8a8578" fontWeight="700">
                  paper {formatNumber(m.paper_median, 3)}
                </text>
              )}
              <circle cx={xd} cy={y} r="7.5" fill="#2f6fbb" stroke="#ffffff" strokeWidth="2" />
              <text x={xd} y={y - 13} textAnchor="middle" className="chart-tick" fill="#2f6fbb" fontWeight="700">
                ours {formatNumber(m.median, 3)}
              </text>
            </g>
          );
        })}
        <text x={(margin.left + chartWidth - margin.right) / 2} y={chartHeight - 12} textAnchor="middle" className="chart-axis-label">
          Median final cost S (log) - lower is better
        </text>
        <g transform={`translate(${chartWidth - 112} ${margin.top + 4})`}>
          <circle cx="8" cy="0" r="6" fill="#2f6fbb" />
          <text x="20" y="4" className="chart-legend-text">dashboard</text>
          <circle cx="8" cy="20" r="6" fill="#8a8578" />
          <text x="20" y="24" className="chart-legend-text">paper</text>
        </g>
      </svg>
    </div>
  );
}

// The Section 4.2 ordering (twin transfer vs. cold-start BO) turns on a skopt
// setting the paper never prints. Rather than assert one end of that range, the
// panel shows the band: our configuration, the three the study varied, and the
// paper's own numbers, side by side.
function RetuningBoSensitivity({ block }) {
  if (!block) return null;
  if (!block.available) {
    return <div className="empty-panel compact-empty">{block.reason}</div>;
  }

  const band = block.band ?? {};
  const pct = (v) => (v == null ? 'n/a' : `${formatNumber(v, 1)}%`);

  return (
    <>
      <p className="plant-note bo-question">{block.question}</p>

      <div className="bo-band">
        <span className="bo-band-label">HGS-only win rate, same plant / twin / cost / transfer:</span>
        <span className="bo-band-scale">
          <b>{pct(band.min_percent)}</b> <span>ours (log-uniform)</span>
          <i aria-hidden="true">&rarr;</i>
          <b>{pct(band.max_percent)}</b> <span>linear-uniform prior</span>
          <i aria-hidden="true">&rarr;</i>
          <b>{pct(band.paper_percent)}</b> <span>paper</span>
        </span>
      </div>

      <p className="plant-note">{block.finding}</p>

      <div className="table-scroll">
        <table className="data-table logging-table bo-config-table">
          <thead>
            <tr>
              <th>Configuration</th>
              <th>Search space</th>
              <th>Baseline</th>
              <th>Median @5</th>
              <th>Median @30</th>
              <th>Final / bound</th>
              <th>CS-BO &divide; HGS</th>
              <th>Ordering</th>
              <th>HGS wins</th>
              <th>n</th>
            </tr>
          </thead>
          <tbody>
            {block.variants.map((v) => {
              const ratio = v.cs_bo_over_hgs;
              const ordering = ratio == null ? '-' : ratio < 1 ? 'BO wins' : 'HGS wins';
              const rowClass = [
                v.is_default ? 'is-default' : '',
                v.reference_only ? 'is-paper' : '',
              ].filter(Boolean).join(' ');
              return (
                <tr key={v.key} className={rowClass} title={v.hypothesis}>
                  <td>
                    {v.label}
                    {v.is_default && <span className="bo-tag">this dashboard</span>}
                  </td>
                  <td className="bo-space">{v.search_space}</td>
                  <td>{v.baseline_method}</td>
                  <td>{formatNumber(v.median_at_5, 3)}</td>
                  <td>{formatNumber(v.median_at_30, 3)}</td>
                  <td>{v.final_over_bound == null ? '-' : formatNumber(v.final_over_bound, 3)}</td>
                  <td>{ratio == null ? '-' : formatNumber(ratio, 3)}</td>
                  <td className={ratio == null ? '' : ratio < 1 ? 'bo-verdict-bo' : 'bo-verdict-hgs'}>
                    {ordering}
                  </td>
                  <td>{pct(v.hgs_win_percent)}</td>
                  <td>{v.n_pairs == null ? `${v.n_runs} runs` : `${v.n_pairs} cells`}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <p className="plant-note">
        <b>Reading it.</b> &ldquo;CS-BO &divide; HGS&rdquo; below 1 means cold-start BO
        beats the twin transfer &mdash; the inversion you see in the tables above.
        Our log-uniform prior converges CS-BO(30) onto the per-cell achievable
        bound (final / bound &asymp; 1.00), which leaves the twin nothing to win;
        the paper&rsquo;s baseline stops short of its own floor. Box width is ruled
        out, and cluster seeding reproduces the paper&rsquo;s pinned warm start
        without moving the ordering. Source: {block.source}.
      </p>
    </>
  );
}

function RetuningPage({ baseUrl }) {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [protocol, setProtocol] = useState('field_matched');

  async function loadResults() {
    setLoading(true);
    setError('');
    try {
      const payload = await apiPost(baseUrl, '/validate/retuning', {});
      setResult(payload);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    loadResults();
  }, []);

  const proto = result?.protocols?.[protocol];
  const methods = proto?.methods ?? [];
  const perPlant = proto?.per_plant ?? [];
  const winRates = proto?.win_rates ?? {};
  const csbo = methods.find((m) => m.method === 'CS-BO(30)');
  const hgs = methods.find((m) => m.method === 'HGS-only');
  const anchored = winRates.hgs_vs_csbo_anchored5;
  const provenance = result?.provenance;
  const caveats = result?.caveats ?? [];
  const reports = result?.reports ?? [];

  return (
    <>
      {error && <ErrorBanner message={error} />}
      <section className="damping-layout">
        <div className="panel controls-panel compact-controls">
          <div className="section-heading">
            <RefreshCw size={17} />
            <h2>Adaptive Retuning (Section 4)</h2>
          </div>
          <div className="field-grid">
            <SelectField
              label="Identification protocol"
              value={protocol}
              options={[
                { value: 'field_matched', label: 'Field-matched (Tlog 5 ms, dual 0.3/0.3, LPF 50/50)' },
                { value: 'logging_only', label: 'Logging-only (Tlog 20 ms, tension 0.3, LPF 100)' },
              ]}
              onChange={setProtocol}
            />
            <RunButton loading={loading} onClick={loadResults}>Load Retuning Results</RunButton>
          </div>
        </div>

        <section className="result-column">
          <div className="metric-grid">
            <MetricCard
              label="CS-BO(30) median"
              value={csbo ? formatNumber(csbo.median, 3) : 'n/a'}
              unit={csbo?.paper_median != null ? `paper ${formatNumber(csbo.paper_median, 3)}` : ''}
            />
            <MetricCard
              label="HGS-only median (0 real evals)"
              value={hgs ? formatNumber(hgs.median, 3) : 'n/a'}
              unit={hgs?.paper_median != null ? `paper ${formatNumber(hgs.paper_median, 3)}` : ''}
            />
            <MetricCard
              label="HGS-only wins vs CS-BO (anchored-5)"
              value={anchored?.percent != null ? `${formatNumber(anchored.percent, 1)}%` : 'n/a'}
              unit={protocol === 'field_matched' ? 'paper 58%' : 'paper 5%'}
            />
          </div>

          <div className="panel">
            <div className="section-heading">
              <h3>Median final cost - dashboard vs paper</h3>
            </div>
            <RetuningFinalCostChart methods={methods} />
          </div>

          <div className="panel">
            <div className="section-heading">
              <h3>Five-method comparison</h3>
            </div>
            <div className="table-scroll">
              <table className="data-table logging-table">
                <thead>
                  <tr>
                    <th>Method</th>
                    <th>Real evals</th>
                    <th>n</th>
                    <th>Median</th>
                    <th>P5</th>
                    <th>P95</th>
                    <th>Paper median</th>
                    <th>Paper P5</th>
                    <th>Paper P95</th>
                  </tr>
                </thead>
                <tbody>
                  {methods.map((m) => (
                    <tr key={m.method}>
                      <td>{m.method}</td>
                      <td>{m.real_evals}</td>
                      <td>{m.n}</td>
                      <td>{formatNumber(m.median, 3)}</td>
                      <td>{formatNumber(m.p5, 3)}</td>
                      <td>{formatNumber(m.p95, 3)}</td>
                      <td>{m.paper_median != null ? formatNumber(m.paper_median, 3) : '-'}</td>
                      <td>{m.paper_p5 != null ? formatNumber(m.paper_p5, 3) : '-'}</td>
                      <td>{m.paper_p95 != null ? formatNumber(m.paper_p95, 3) : '-'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          <div className="panel">
            <div className="section-heading">
              <h3>Convergence vs real-plant evaluations</h3>
            </div>
            <RetuningConvergenceChart conv={result?.convergence} />
          </div>

          <div className="panel">
            <div className="section-heading">
              <h3>BO configuration sensitivity &mdash; why the ordering inverts</h3>
            </div>
            <RetuningBoSensitivity block={result?.bo_config_sensitivity} />
          </div>

          <div className="panel">
            <div className="section-heading">
              <h3>Provenance and caveats</h3>
            </div>
            {provenance && (
              <div className="drift-detail-grid provenance-detail">
                <span>Campaign: {result?.campaign}</span>
                <span>Evaluation model: {provenance.evaluation_model}</span>
                <span>Run: {provenance.run_date}</span>
                <span>Backend: {provenance.backend}</span>
              </div>
            )}
            {provenance?.evaluation_model_meaning && (
              <p className="plant-note">{provenance.evaluation_model_meaning}</p>
            )}
            {provenance?.evaluation_model_status && (
              <p className="plant-note"><b>Status:</b> {provenance.evaluation_model_status}</p>
            )}
            {caveats.length > 0 && (
              <ul className="caveat-list">
                {caveats.map((c) => <li key={c}>{c}</li>)}
              </ul>
            )}
            {reports.length > 0 && (
              <div className="result-actions">
                {reports.map((r) => (
                  <a key={r.url} className="icon-link" href={artifactUrl(baseUrl, r.url)}
                     target="_blank" rel="noreferrer">
                    <Download size={16} />
                    {r.label}
                  </a>
                ))}
              </div>
            )}
          </div>

          <div className="panel">
            <div className="section-heading">
              <h3>Per-plant breakdown</h3>
            </div>
            <div className="table-scroll">
              <table className="data-table logging-table">
                <thead>
                  <tr>
                    <th>Pool id</th>
                    <th>Dashboard id</th>
                    <th>Twin error (MARE %)</th>
                    <th>CS-BO median</th>
                    <th>HGS-only median</th>
                    <th>Achievable bound</th>
                  </tr>
                </thead>
                <tbody>
                  {perPlant.map((row) => (
                    <tr key={row.pool_id}>
                      <td>{row.pool_id}</td>
                      <td>{row.dashboard_id}</td>
                      <td>{row.twin_mare_percent != null ? formatNumber(row.twin_mare_percent, 1) : '-'}</td>
                      <td>{formatNumber(row.cs_bo_median, 3)}</td>
                      <td>{row.hgs_only_median != null ? formatNumber(row.hgs_only_median, 3) : '-'}</td>
                      <td>{row.bound_median != null ? formatNumber(row.bound_median, 3) : '-'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </section>
      </section>
    </>
  );
}

function DampingPage({
  baseUrl,
  plantOptions,
  excitationOptions,
}) {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [selectedGain, setSelectedGain] = useState('100');
  const metrics = result?.metrics ?? null;
  const gainRows = metrics?.gain_rows ?? [];
  const selectedGainRow = gainRows.find((row) => String(row.kp_star) === selectedGain);
  const comparisonRows = metrics?.comparison_rows ?? [];
  const sysidRows = metrics?.sysid_error_rows ?? [];
  const regimeRows = metrics?.regime_rows ?? [];
  const eigenvalueRows = metrics?.eigenvalue_rows ?? [];
  const selectedEigenvalueRows = eigenvalueRows
    .filter((row) => String(row.kp_star) === selectedGain)
    .map((row) => ({
      plant_id: row.plant_id,
      kp_star: row.kp_star,
      calculated_zeta_cl_min: row.calculated_zeta_cl_min,
      calculated_regime: row.calculated_regime,
      reference_zeta_cl_min_at_kp100: row.reference_zeta_cl_min_at_kp100,
      reference_error_percent: row.reference_error_percent,
      stable: row.stable,
      zeta_step_sensitivity_spread: row.zeta_step_sensitivity_spread,
      calculation_source: row.calculation_source,
    }));
  const stepRows = metrics?.step_response_rows ?? [];
  const csvUrl = artifactUrl(baseUrl, result?.csv_url);
  const summaryUrl = artifactUrl(baseUrl, result?.summary_url);

  async function runDamping() {
    setLoading(true);
    setError('');
    try {
      const payload = await apiPost(baseUrl, '/validate/closed-loop-damping', {});
      setResult(payload);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    runDamping();
  }, []);

  return (
    <>
      {error && <ErrorBanner message={error} />}
      <section className="damping-layout">
        <div className="panel controls-panel compact-controls">
          <div className="section-heading">
            <Gauge size={17} />
            <h2>Closed-loop Damping Controls</h2>
          </div>
          <div className="field-grid">
            <SelectField
              label="Gain case"
              value={selectedGain}
              options={[
                { value: '50', label: 'Kp*=50' },
                { value: '100', label: 'Kp*=100' },
                { value: '200', label: 'Kp*=200' },
              ]}
              onChange={setSelectedGain}
            />
            <div className="drift-detail-grid">
              <span>Tlog: {metrics?.settings?.Tlog_ms ?? 20} ms</span>
              <span>LPF: {metrics?.settings?.LPF_Hz ?? 100} Hz</span>
            <span>Condition: {metrics?.settings?.measurement_condition_label ?? 'Tension-only'}</span>
              <span>Noise: {metrics?.settings?.sensor_noise_percent ?? 0.3}%</span>
              <span>Excitation: {metrics?.settings?.excitation ?? 'E_Toggle'}</span>
            </div>
            <RunButton loading={loading} onClick={runDamping}>Run Closed-loop Damping</RunButton>
            <div className="result-actions">
              {csvUrl && (
                <a className="icon-link" href={csvUrl} target="_blank" rel="noreferrer">
                  <Download size={16} />
                  CSV
                </a>
              )}
              {summaryUrl && (
                <a className="icon-link" href={summaryUrl} target="_blank" rel="noreferrer">
                  <Download size={16} />
                  Summary
                </a>
              )}
            </div>
          </div>
        </div>

        <section className="result-column">
          <div className="metric-grid">
            <MetricCard label="Validation" value={metrics?.validation_status ?? 'n/a'} unit="Closed-loop damping" />
            <MetricCard
              label="Fresh SN minimum"
              value={metrics?.recommended_gain ? `Kp*=${metrics.recommended_gain}` : 'n/a'}
              unit="Use with the calculated overshoot check"
            />
            <MetricCard
              label={`Kp*=${selectedGain}`}
              value={selectedGainRow ? `${formatNumber(selectedGainRow.SN_dashboard_MARE_theta_percent, 1)}%` : 'n/a'}
              unit={selectedGainRow ? `${selectedGainRow.response_behavior}, SN MARE` : ''}
            />
            <MetricCard
              label="Kp*=100 eigen check"
              value={metrics?.eigenvalue_summary ? `${metrics.eigenvalue_summary.baseline_reference_match_count}/${metrics.eigenvalue_summary.baseline_reference_row_count}` : 'n/a'}
              unit="live full-cascade zeta/regime matches"
            />
            <MetricCard
              label="Paper default"
              value={metrics?.paper_default_gain ? `Kp*=${metrics.paper_default_gain}` : 'n/a'}
              unit="kept for identifiability and safety, not noise averaging"
            />
          </div>

          <section className="panel result-panel damping-panel">
            <div className="section-heading split-heading">
              <div>
                <LineChart size={17} />
                <h2>Step Response vs Kp*</h2>
              </div>
              <span>Normalized tension</span>
            </div>
            <DampingStepChart rows={stepRows} metrics={gainRows} />
          </section>

          <section className="panel result-panel damping-panel">
            <div className="section-heading split-heading">
              <div>
                <BarChart3 size={17} />
                <h2>Paper vs Dashboard</h2>
              </div>
              <span>MARE_theta gain sweep</span>
            </div>
            <DampingPaperDashboardChart rows={comparisonRows} />
          </section>

          <section className="panel result-panel damping-panel">
            <div className="section-heading split-heading">
              <div>
                <BarChart3 size={17} />
                <h2>O-UD vs H-Damp</h2>
              </div>
              <span>Paper and dashboard bars</span>
            </div>
            <DampingRegimeChart rows={regimeRows} />
          </section>

          <section className="panel result-panel table-panel damping-panel">
            <div className="section-heading split-heading">
              <div>
                <Calculator size={17} />
                <h2>Live Full-Cascade Eigenvalues</h2>
              </div>
              <span>Calculated Kp*={selectedGain}; Table S12 comparison only at Kp*=100</span>
            </div>
            {selectedEigenvalueRows.length ? <MetricTable rows={selectedEigenvalueRows} /> : <div className="empty-panel compact-empty">Run closed-loop damping validation to calculate eigenvalues.</div>}
          </section>

          <section className="panel result-panel damping-panel">
            <div className="section-heading split-heading">
              <div>
                <LineChart size={17} />
                <h2>SysID Error vs Kp*</h2>
              </div>
              <span>NF and SN dashboard trends</span>
            </div>
            <DampingSysIdChart rows={sysidRows} />
          </section>

          <section className="panel result-panel table-panel damping-panel">
            <div className="section-heading">
              <Calculator size={17} />
              <h2>Comparison Table</h2>
            </div>
            {comparisonRows.length ? <MetricTable rows={comparisonRows} /> : <div className="empty-panel compact-empty">Run closed-loop damping validation to calculate comparison rows.</div>}
          </section>

          <section className="panel result-panel table-panel damping-panel">
            <div className="section-heading">
              <Check size={17} />
              <h2>Acceptance Checks</h2>
            </div>
            {metrics?.acceptance_checks?.length ? <MetricTable rows={metrics.acceptance_checks} /> : <div className="empty-panel compact-empty">Run closed-loop damping validation to calculate acceptance checks.</div>}
          </section>
        </section>
      </section>
    </>
  );
}

function EquationPage() {
  return (
    <section className="equation-layout">
      {EQUATION_SECTIONS.map((section) => (
        <div className="panel equation-panel" key={section.title}>
          <div className="equation-heading">
            <Calculator size={20} />
            <h2>{section.title}</h2>
          </div>
          <div className="equation-list">
            {section.rows.map((row) => (
              <article className="equation-row" key={row.label}>
                <div>
                  <h3>{row.label}</h3>
                  <p>{row.note}</p>
                </div>
                <code>{row.equation}</code>
              </article>
            ))}
          </div>
        </div>
      ))}
    </section>
  );
}

function formatPercent(value, digits = 2) {
  if (value === null || value === undefined || value === '') return null;
  const number = Number(value);
  if (!Number.isFinite(number)) return null;
  return `${number.toFixed(digits)}%`;
}

// v5 Section 3.3 does not print a paper median for every drift leg: EA is
// published as a band across its five legs and friction prints a noise-free
// pair only. The paper value for those legs therefore lives on the family, not
// on the row, and the cell says which it is instead of showing an empty box.
function paperCellForRow(row, condition, familyReference) {
  const value = formatPercent(row?.[`paper_${condition}_percent`]);
  if (value) return { text: value, kind: 'published' };

  const family = String(row?.drift_family ?? '');
  const reference = familyReference?.[family];
  const band = reference?.[`paper_${condition}_band_percent`];
  if (Array.isArray(band) && band.length === 2) {
    return { text: `band ${Number(band[0]).toFixed(1)}-${Number(band[1]).toFixed(1)}%`, kind: 'band' };
  }
  if (String(row?.paper_reference_type) === 'unpublished_axis_tick') {
    return { text: 'axis tick only', kind: 'unpublished' };
  }
  return { text: 'not published', kind: 'unpublished' };
}

function DriftPaperReference({ family, reference }) {
  const entry = reference?.[family];
  if (!entry) return null;

  const items = [];
  if (family === 'EA') {
    const nf = entry.paper_NF_band_percent;
    const sn = entry.paper_SN_band_percent;
    if (Array.isArray(nf)) {
      items.push({
        label: 'Paper NF band',
        value: `${Number(nf[0]).toFixed(1)}-${Number(nf[1]).toFixed(1)}%`,
        unit: `mean ${formatPercent(entry.paper_NF_mean_percent, 1) ?? 'n/a'}`,
      });
    }
    if (Array.isArray(sn)) {
      items.push({
        label: 'Paper SN band',
        value: `${Number(sn[0]).toFixed(1)}-${Number(sn[1]).toFixed(1)}%`,
        unit: `mean ${formatPercent(entry.paper_SN_mean_percent, 1) ?? 'n/a'}`,
      });
    }
    if (entry.legs) items.push({ label: 'Published legs', value: entry.legs, unit: 'band, not per leg' });
  } else if (family === 'f') {
    const byPercent = entry.paper_NF_percent_by_drift_percent ?? {};
    for (const [percent, value] of Object.entries(byPercent)) {
      const formatted = formatPercent(value, 1);
      if (formatted) {
        items.push({ label: `Paper NF at f ${Number(percent) > 0 ? '+' : ''}${percent}%`, value: formatted, unit: 'noise-free' });
      }
    }
    items.push({ label: 'Paper SN', value: 'not published', unit: 'NF pair only' });
  } else if (family === 'J') {
    const gap = entry.gap_above_EA_baseline_pp ?? {};
    if (gap.NF !== undefined) items.push({ label: 'Gap above EA baseline', value: `+${Number(gap.NF).toFixed(1)} pp`, unit: 'noise-free' });
    if (gap.SN !== undefined) items.push({ label: 'Gap above EA baseline', value: `+${Number(gap.SN).toFixed(1)} pp`, unit: 'sensor noise' });
    items.push({ label: 'Published legs', value: 'per leg', unit: 'both J scenarios' });
  }

  const acquisition = entry.acquisition_condition;
  const combined = reference?.combined;

  return (
    <section className="panel result-panel paper-reference-panel">
      <div className="section-heading split-heading">
        <div>
          <Sigma size={17} />
          <h2>Paper Reference | {family} family</h2>
        </div>
        <span>{reference.__source || 'paper1_isa_v5 Section 3.3 and Fig. 4'}</span>
      </div>
      <div className="metric-grid">
        {items.map((item) => (
          <MetricCard key={`${item.label}-${item.unit}`} label={item.label} value={item.value} unit={item.unit} />
        ))}
      </div>
      {entry.note && <p className="plant-note">{entry.note}</p>}
      {acquisition && (
        <p className="plant-note">
          Acquisition: {acquisition.measurement_condition?.replaceAll('_', ' ')} at {acquisition.excitation}, Tlog{' '}
          {acquisition.Tlog_ms} ms, LPF {acquisition.lpf_hz} Hz, tension noise {acquisition.pct_T}% FS.{' '}
          {acquisition.note}
        </p>
      )}
      {combined?.paper_NF_percent !== undefined && combined?.paper_NF_percent !== null && (
        <p className="plant-note">
          Combined scenario ({combined.scenario}): paper NF {formatPercent(combined.paper_NF_percent, 1)}; the paper
          prints no matching SN value.
        </p>
      )}
    </section>
  );
}

function DriftComparisonRows({ rows, familyReference }) {
  if (!rows.length) {
    return <div className="empty-panel compact-empty">Run the drift study to calculate comparison rows.</div>;
  }

  return (
    <div className="table-scroll">
      <table className="data-table logging-table">
        <thead>
          <tr>
            <th>Drift case</th>
            <th>Dashboard NF</th>
            <th>Paper NF</th>
            <th>Difference NF</th>
            <th>Dashboard SN</th>
            <th>Paper SN</th>
            <th>Difference SN</th>
            <th>Runs NF / SN</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => {
            const paperNf = paperCellForRow(row, 'NF', familyReference);
            const paperSn = paperCellForRow(row, 'SN', familyReference);
            return (
              <tr key={String(row.drift_case)}>
                <td>{row.drift_case}</td>
                <td>{formatPercent(row.dashboard_NF_percent) ?? 'n/a'}</td>
                <td className={`paper-cell ${paperNf.kind}`}>{paperNf.text}</td>
                <td>{formatPercent(row.difference_NF_percent) ?? '-'}</td>
                <td>{formatPercent(row.dashboard_SN_percent) ?? 'n/a'}</td>
                <td className={`paper-cell ${paperSn.kind}`}>{paperSn.text}</td>
                <td>{formatPercent(row.difference_SN_percent) ?? '-'}</td>
                <td>
                  {row.valid_run_count_NF ?? '-'}/{row.expected_run_count_NF ?? '-'} ·{' '}
                  {row.valid_run_count_SN ?? '-'}/{row.expected_run_count_SN ?? '-'}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function DriftPage({
  baseUrl,
  selectedPlantId,
  plantOptions,
  selectedPlant,
  onPlantSelect,
  excitationOptions,
}) {
  const [autoLoaded, setAutoLoaded] = useState(false);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [driftPlantId, setDriftPlantId] = useState('ALL');
  const [activeDriftFamily, setActiveDriftFamily] = useState('EA');
  const familyLabels = ['EA', 'f', 'J'];
  const familyPlotUrls = {
    EA: artifactUrl(baseUrl, result?.drift_EA_plot_url ?? result?.plot_url),
    f: artifactUrl(baseUrl, result?.drift_f_plot_url),
    J: artifactUrl(baseUrl, result?.drift_J_plot_url),
  };
  const plotUrl = familyPlotUrls[activeDriftFamily] ?? artifactUrl(baseUrl, result?.plot_url);
  const csvUrl = artifactUrl(baseUrl, result?.csv_url);
  const summaryUrl = artifactUrl(baseUrl, result?.summary_url);
  const driftSummary = result?.metrics?.comparison_rows ? result.metrics : result ?? null;
  const comparisonRows = Array.isArray(driftSummary?.comparison_rows)
    ? driftSummary.comparison_rows
    : Array.isArray(driftSummary?.metrics)
      ? driftSummary.metrics
      : [];
  const rowsByFamily = driftSummary?.family_rows ?? {};
  const paperFamilyReference = driftSummary?.paper_family_reference ?? null;
  const activeComparisonRows = Array.isArray(rowsByFamily?.[activeDriftFamily])
    ? rowsByFamily[activeDriftFamily]
    : comparisonRows.filter((row) => row.drift_family === activeDriftFamily);
  const dominantFamilyRow = activeComparisonRows.length
    ? activeComparisonRows.reduce((best, row) => (Number(row.dashboard_SN_percent) > Number(best.dashboard_SN_percent) ? row : best), activeComparisonRows[0])
    : null;
  const dominant = dominantFamilyRow?.drift_case ?? 'n/a';
  const activeDriftPlant = plantOptions.find((plant) => plant.value === driftPlantId);

  async function runDriftStudy() {
    setLoading(true);
    setError('');
    try {
      const payload = await apiPost(baseUrl, '/validate/drift', { plant_id: driftPlantId, force_rerun: true });
      setResult(payload);
      setActiveDriftFamily('EA');
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    if (autoLoaded) return;
    setAutoLoaded(true);
    setLoading(true);
    autoLoadRequest(
      `drift:${baseUrl}`,
      () => apiPost(baseUrl, '/validate/drift', { plant_id: 'ALL' }),
    )
      .then(setResult)
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }, [autoLoaded, baseUrl]);

  return (
    <>
      {error && <ErrorBanner message={error} />}
      <section className="drift-layout">
        <div className="panel controls-panel compact-controls">
          <div className="section-heading">
            <GitCompare size={17} />
            <h2>Drift Controls</h2>
          </div>
          <div className="field-grid">
            {plantOptions.length > 0 && (
              <SelectField label="Plant scope" value={driftPlantId} options={plantOptions} onChange={setDriftPlantId} />
            )}
            <p className="plant-note">
              {driftPlantId === 'ALL'
                ? 'Freshly simulates all 10 plants and recalculates each drift case with the paper Eq. 8 weighted one-step PEM.'
                : `Freshly simulates ${activeDriftPlant?.label ?? selectedPlant?.label ?? 'the selected plant'} with the same independent drift protocol.`}
            </p>
            <div className="drift-detail-grid">
              <span>EA drift cases: -30%, -10%, +10%, +30%</span>
              <span>f drift cases: -30%, -15%, 0%, +15%, +30% (paper comparison values exist only at ±30%; intermediate values are fresh dashboard calculations)</span>
              <span>J drift cases: UW -30%, RW +50% and UW -50%, RW +100%</span>
              <span>Simulation: fresh 7 s trajectories with the 1 ms controller, per-plant auto_Ti, and no undocumented velocity-correction clamp</span>
              <span>Estimator: paper Eq. 7 nonlinear one-step tension PEM; all-seven-parameter pre-drift-baseline MARE with per-run optimizer, bound, estimate, and parameter-error diagnostics</span>
              <span>Reference: phase3_drift_results.csv is comparison-only and never copied into dashboard results</span>
              <span>Aggregation: NF median across 10 simulations; SN median across 30 plant/seed simulations. Never read a pooled median without the slow/fast split - the pooled valley represents neither sub-group.</span>
            </div>
            <RunButton loading={loading} onClick={runDriftStudy}>Run Drift Study</RunButton>
          </div>
        </div>

        <section className="result-column">
          <div className="drift-family-tabs" role="tablist" aria-label="Drift family">
            {familyLabels.map((family) => (
              <button
                key={family}
                className={family === activeDriftFamily ? 'active' : ''}
                type="button"
                onClick={() => setActiveDriftFamily(family)}
                role="tab"
                aria-selected={family === activeDriftFamily}
              >
                {family}
              </button>
            ))}
          </div>
          <div className="metric-grid">
            <MetricCard label="Highest SN drift" value={dominant} unit={result ? 'dashboard median' : ''} />
            <MetricCard
              label="Active family"
              value={result ? activeDriftFamily : 'n/a'}
              unit={result ? 'selected graph' : ''}
            />
            <MetricCard label="Rows" value={activeComparisonRows.length || 'n/a'} unit={activeComparisonRows.length ? 'comparison cases' : ''} />
          </div>

          <section className="panel result-panel drift-graph-panel">
            <div className="section-heading split-heading">
              <div>
                <LineChart size={17} />
                <h2>{activeDriftFamily} Drift vs Median SysID Error</h2>
              </div>
              <span>Dashboard and paper reference, NF and SN</span>
            </div>
            {plotUrl ? (
              <div className="plot-frame drift-plot-frame">
                <img src={plotUrl} alt="Drift scenario degradation graph" />
              </div>
            ) : (
              <div className="empty-panel compact-empty">No drift graph yet.</div>
            )}
            <div className="result-actions">
              {activeComparisonRows.length > 0 && (
                <button
                  className="icon-text-button"
                  type="button"
                  onClick={() => downloadCsv(`drift_${activeDriftFamily}_dashboard_vs_paper.csv`, activeComparisonRows)}
                >
                  <Download size={16} />
                  {activeDriftFamily} CSV
                </button>
              )}
              {csvUrl && (
                <a className="icon-link" href={csvUrl} target="_blank" rel="noreferrer">
                  <Download size={16} />
                  All Drift CSV
                </a>
              )}
              {summaryUrl && (
                <a className="icon-link" href={summaryUrl} target="_blank" rel="noreferrer">
                  <Download size={16} />
                  Summary
                </a>
              )}
            </div>
          </section>

          <section className="panel result-panel drift-detail-panel">
            <div className="section-heading">
              <Calculator size={17} />
              <h2>{activeDriftFamily} Comparison Table</h2>
            </div>
            {driftSummary?.dashboard_iteration_note && <p className="plant-note">{driftSummary.dashboard_iteration_note}</p>}
            <DriftComparisonRows rows={activeComparisonRows} familyReference={paperFamilyReference} />
          </section>

          <DriftPaperReference family={activeDriftFamily} reference={paperFamilyReference} />
        </section>
      </section>
    </>
  );
}

function ErrorBanner({ message }) {
  if (!message) return null;
  return <div className="error-banner">{message}</div>;
}

function formatNumber(value, digits = 4) {
  if (value === null || value === undefined || value === '') return 'n/a';
  const number = Number(value);
  if (!Number.isFinite(number)) return 'n/a';
  if (Math.abs(number) >= 1000) return number.toLocaleString(undefined, { maximumFractionDigits: 0 });
  if (Math.abs(number) > 0 && Math.abs(number) < 0.001) return number.toExponential(2);
  return number.toLocaleString(undefined, { maximumFractionDigits: digits });
}

function cacheBustUrl(url, version) {
  if (!url) return null;
  return `${url}${url.includes('?') ? '&' : '?'}v=${version}`;
}

function MetricCard({ label, value, unit }) {
  return (
    <article className="metric-card">
      <span>{label}</span>
      <strong>{value}</strong>
      {unit && <small>{unit}</small>}
    </article>
  );
}

function LineLegend({ visible }) {
  if (!visible) return null;
  return (
    <div className="line-fits">
      <span className="line-key dashboard-line">Dashboard</span>
      <span className="line-key paper-line">Paper result</span>
    </div>
  );
}

const CONDITION_LABELS = {
  noise_free: 'Noise-free',
  tension_only: 'Tension-only',
  dual_channel: 'Dual-channel',
};

// A blank paper cell reads as "the dashboard lost the value". v5 publishes a
// different Tlog span per condition, so a cell with no published number says
// which of the two reasons applies: the paper prints nothing there, or the
// value is an exact fit that falls off a logarithmic axis.
function loggingPaperCell(row) {
  const value = row.paper_numerical_MARE_theta_percent;
  if (value !== null && value !== undefined && Number.isFinite(Number(value))) {
    return { text: `${formatNumber(value, 2)}%`, kind: 'published' };
  }
  if (row.paper_value_status === 'off_scale_below_0p01_percent') {
    return { text: 'off-scale (<0.01%)', kind: 'offscale' };
  }
  return { text: 'not published', kind: 'unpublished' };
}

function LoggingResultRows({ rows }) {
  if (!rows.length) {
    return <div className="empty-panel compact-empty">No sweep rows yet.</div>;
  }

  return (
    <div className="table-scroll">
      <table className="data-table logging-table">
        <thead>
          <tr>
            <th>Condition</th>
            <th>Tlog</th>
            <th>tau_min</th>
            <th>Tlog/tau_min</th>
            <th>Dashboard MARE</th>
            <th>Paper MARE</th>
            <th>Difference</th>
          </tr>
        </thead>
        <tbody id="rowsBody">
          {rows.map((row) => {
            const paperValue = row.paper_numerical_MARE_theta_percent;
            const paperCell = loggingPaperCell(row);
            const difference =
              Number.isFinite(Number(row.MARE_theta_percent)) &&
              paperValue !== null &&
              paperValue !== undefined &&
              Number.isFinite(Number(paperValue))
                ? Number(row.MARE_theta_percent) - Number(paperValue)
                : null;
            const condition = row.measurement_condition ?? row.case;

            return (
              <tr key={`${condition}-${row.Tlog_ms}`}>
                <td>{CONDITION_LABELS[condition] ?? condition}</td>
                <td>{formatNumber(row.Tlog_ms, 1)} ms</td>
                <td>{formatNumber(row.tmin_ms, 1)} ms</td>
                <td>{formatNumber(row.tau_ratio, 3)}</td>
                <td>{formatNumber(row.MARE_theta_percent, 3)}%</td>
                <td className={`paper-cell ${paperCell.kind}`}>{paperCell.text}</td>
                <td>{difference === null ? '-' : `${formatNumber(difference, 3)}%`}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function LoggingPage({
  baseUrl,
  metadata,
  plantOptions,
  excitationOptions,
}) {
  const [autoLoaded, setAutoLoaded] = useState(false);
  const [plantId, setPlantId] = useState('ALL');
  const [selectedTlogs, setSelectedTlogs] = useState(FALLBACK_TLOG_OPTIONS);
  const [customTlog, setCustomTlog] = useState('20');
  const [tminMs, setTminMs] = useState('50');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [runStatus, setRunStatus] = useState('');
  const [runVersion, setRunVersion] = useState(0);
  const [activeLoggingGraph, setActiveLoggingGraph] = useState('numerical');

  const plants = metadata?.plants ?? [];
  const tlogOptions = metadata?.tlog_ms_options ?? FALLBACK_TLOG_OPTIONS;
  const summary = result?.metrics;
  const rows = summary?.metrics ?? [];
  const powerLawPlotUrl = cacheBustUrl(artifactUrl(baseUrl, result?.power_law_plot_url), runVersion);
  const numericalPlotUrl = cacheBustUrl(artifactUrl(baseUrl, result?.plot_url), runVersion);
  const speedPlotUrl = cacheBustUrl(artifactUrl(baseUrl, result?.speed_plot_url), runVersion);
  const LOGGING_GRAPHS = {
    numerical: {
      url: numericalPlotUrl,
      title: 'Fig. 2(a) | Three measurement conditions',
      subtitle: 'Dashboard and paper MARE vs Tlog, per condition',
    },
    speed: {
      url: speedPlotUrl,
      title: 'Fig. 2(b) | Dual-channel speed decomposition',
      subtitle: 'Slow and fast plant groups against the pooled median',
    },
    power: {
      url: powerLawPlotUrl,
      title: 'Fig. S2 | Noise-free power law',
      subtitle: 'Noise-free branch only; the sensor-noise branch is not overlaid',
    },
  };
  const activeGraph = LOGGING_GRAPHS[activeLoggingGraph] ?? LOGGING_GRAPHS.numerical;
  const activePlotUrl = activeGraph.url;
  const activeGraphTitle = activeGraph.title;
  const activeGraphSubtitle = activeGraph.subtitle;
  const graphPointCsvUrl = artifactUrl(baseUrl, result?.graph_points_csv_url);
  const speedCsvUrl = artifactUrl(baseUrl, result?.speed_csv_url);
  useEffect(() => {
    if (!metadata) return;
    setPlantId(metadata.default_plant_id ?? 'ALL');
    setSelectedTlogs(metadata.tlog_ms_options?.length ? metadata.tlog_ms_options : FALLBACK_TLOG_OPTIONS);
    setTminMs(String(metadata.default_tmin_ms ?? 50));
  }, [metadata]);

  const ratioPreviewRows = useMemo(() => {
    const tmin = Number(tminMs);
    if (!Number.isFinite(tmin) || tmin <= 0) return [];
    return selectedTlogs
      .slice()
      .sort((a, b) => a - b)
      .map((tlog) => ({ tlog, ratio: tlog / tmin }));
  }, [selectedTlogs, tminMs]);

  function toggleTlog(value) {
    setSelectedTlogs((current) => {
      const next = current.includes(value) ? current.filter((item) => item !== value) : [...current, value];
      return next.sort((a, b) => a - b);
    });
  }

  function addCustomTlog() {
    const value = Number(customTlog);
    if (!Number.isFinite(value) || value < 1 || value > 200) {
      setError('Tlog must be between 1 and 200 ms.');
      return;
    }
    setSelectedTlogs((current) => [...new Set([...current, value])].sort((a, b) => a - b));
    setError('');
  }

  function resolveRunTlogs() {
    const values = new Set(selectedTlogs);
    const customText = customTlog.trim();
    if (customText) {
      const value = Number(customText);
      if (!Number.isFinite(value) || value < 1 || value > 200) {
        return { error: 'Tlog must be between 1 and 200 ms.', values: [] };
      }
      values.add(value);
    }
    const tlogValues = [...values].sort((a, b) => a - b);
    if (!tlogValues.length) return { error: 'Select at least one Tlog value.', values: [] };
    return { error: '', values: tlogValues };
  }

  function resetInputs() {
    setSelectedTlogs(tlogOptions);
    setCustomTlog('20');
    setTminMs(String(metadata?.default_tmin_ms ?? 50));
  }

  async function runLoggingAdequacy() {
    const resolvedTlogs = resolveRunTlogs();
    const tlogValues = resolvedTlogs.values;
    const tminValue = Number(tminMs);
    if (resolvedTlogs.error) {
      setError(resolvedTlogs.error);
      return;
    }
    if (!Number.isFinite(tminValue) || tminValue <= 0) {
      setError('tau_min must be positive.');
      return;
    }

    setLoading(true);
    setError('');
    setSelectedTlogs(tlogValues);
    setRunStatus(
      plantId === 'ALL'
        ? `Running all 10 plants across ${tlogValues.length} Tlog values.`
        : `Running ${plantId} across ${tlogValues.length} Tlog values.`,
    );
    try {
      const payload = await apiPost(baseUrl, '/validate/logging-adequacy', {
        plant_id: plantId,
        tlog_ms_values: tlogValues,
        tmin_ms: tminValue,
      });
      setResult(payload);
      setRunVersion((version) => version + 1);
      const plantCount = payload.metrics?.plant_count ?? 1;
      const interpolated = payload.metrics?.interpolated_tlog_ms_values ?? [];
      const interpolationNote = interpolated.length
        ? ` Estimated Tlog values: ${interpolated.map((value) => formatNumber(value, 1)).join(', ')} ms.`
        : '';
      setRunStatus(
        `Completed ${formatNumber(plantCount, 0)} plant${plantCount === 1 ? '' : 's'}; graph-point CSV is ready.${interpolationNote}`,
      );
    } catch (err) {
      setError(err.message);
      setRunStatus('');
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    if (!metadata || autoLoaded) return;
    setAutoLoaded(true);
    const defaultTlogs = metadata.tlog_ms_options?.length ? metadata.tlog_ms_options : FALLBACK_TLOG_OPTIONS;
    const defaultTmin = Number(metadata.default_tmin_ms ?? 50);
    setLoading(true);
    setRunStatus('Loading the all-plant dashboard and paper MARE comparison.');
    autoLoadRequest(
      `logging:${baseUrl}:${defaultTlogs.join(',')}:${defaultTmin}`,
      () => apiPost(baseUrl, '/validate/logging-adequacy', {
        plant_id: 'ALL',
        tlog_ms_values: defaultTlogs,
        tmin_ms: defaultTmin,
      }),
    )
      .then((payload) => {
        setResult(payload);
        setRunVersion((version) => version + 1);
        setRunStatus('Loaded all 10 plants with dashboard and paper MARE values.');
      })
      .catch((err) => {
        setError(err.message);
        setRunStatus('');
      })
      .finally(() => setLoading(false));
  }, [metadata, autoLoaded, baseUrl]);

  return (
    <>
      {error && <ErrorBanner message={error} />}
      {runStatus && (
        <div className={`run-status ${loading ? 'active' : 'complete'}`} id="runStatus" aria-live="polite">
          {loading && <RefreshCw size={16} />}
          <span>{runStatus}</span>
        </div>
      )}
      <div className="work-grid logging-only-grid">
        <section className="panel controls-panel compact-controls">
          <div className="section-heading">
            <Activity size={17} />
            <h2>Controls</h2>
          </div>
          <div className="field-grid">
            <label className="field">
              <span>Plant scope</span>
              <select value={plantId} onChange={(event) => setPlantId(event.target.value)}>
                {plants.length ? (
                  plants.map((plant) => (
                    <option value={plant.plant_id} key={plant.plant_id}>
                      {plant.label}
                    </option>
                  ))
                ) : (
                  <option value="ALL">All 10 plants | median result</option>
                )}
              </select>
            </label>

            <div className="option-field">
              <span>Tlog options (ms)</span>
              <div className="tlog-option-grid">
                {tlogOptions.map((value) => {
                  const selected = selectedTlogs.includes(value);
                  return (
                    <button
                      className={`tlog-option ${selected ? 'selected' : ''}`}
                      type="button"
                      aria-pressed={selected}
                      onClick={() => toggleTlog(value)}
                      key={value}
                    >
                      <Check size={14} />
                      <span>{value}</span>
                    </button>
                  );
                })}
              </div>
            </div>

            <div className="add-row">
              <input id="customTlog" value={customTlog} onChange={(event) => setCustomTlog(event.target.value)} type="number" min="1" max="200" />
              <button className="icon-link" type="button" onClick={addCustomTlog}>
                <Plus size={16} /> Add
              </button>
            </div>

            <label className="field">
              <span>tau_min (ms)</span>
              <input value={tminMs} onChange={(event) => setTminMs(event.target.value)} type="number" min="1" step="1" />
            </label>

            <div className="ratio-preview">
              {ratioPreviewRows.map((row) => (
                <span key={row.tlog}>
                  {formatNumber(row.tlog, 1)} / {formatNumber(tminMs, 1)} = {formatNumber(row.ratio, 3)}
                </span>
              ))}
            </div>

            <div className="control-actions">
              <button className="icon-link" type="button" onClick={resetInputs}>
                <RotateCcw size={16} /> Reset
              </button>
              <button id="calcBtn" className="primary-button" type="button" onClick={runLoggingAdequacy} disabled={loading}>
                <Calculator size={16} /> {loading ? 'Running' : plantId === 'ALL' ? 'Run 10 plants' : 'Run'}
              </button>
            </div>
          </div>
        </section>

        <section className="result-column">
          <div className="metric-grid">
            <MetricCard
              label="Dual-channel optimum"
              value={summary?.best_noisy_Tlog_ms ? `${formatNumber(summary.best_noisy_Tlog_ms, 1)} ms` : 'n/a'}
              unit={summary?.best_noisy_MARE_theta_percent ? `${formatNumber(summary.best_noisy_MARE_theta_percent, 3)}% MARE` : ''}
            />
            <MetricCard
              label="Tension-only minimum"
              value={summary?.best_tension_only_Tlog_ms ? `${formatNumber(summary.best_tension_only_Tlog_ms, 1)} ms` : 'n/a'}
              unit={
                summary?.best_tension_only_MARE_theta_percent
                  ? `${formatNumber(summary.best_tension_only_MARE_theta_percent, 3)}% MARE (dashboard)`
                  : ''
              }
            />
            <MetricCard
              label="Plant scope"
              value={summary?.plant_count ? `${formatNumber(summary.plant_count, 0)}` : 'n/a'}
              unit={summary?.aggregation === 'median' ? 'median aggregation' : summary ? 'single plant' : ''}
            />
          </div>

          <section className="panel result-panel">
            <div className="drift-family-tabs" role="tablist" aria-label="Logging graph type">
              {[
                ['numerical', 'Three conditions'],
                ['speed', 'Speed split'],
                ['power', 'Power law'],
              ].map(([id, label]) => (
                <button
                  key={id}
                  className={activeLoggingGraph === id ? 'active' : ''}
                  type="button"
                  role="tab"
                  aria-selected={activeLoggingGraph === id}
                  onClick={() => setActiveLoggingGraph(id)}
                >
                  {label}
                </button>
              ))}
            </div>
            <div className="section-heading split-heading">
              <div>
                <LineChart size={17} />
                <h2>{activeGraphTitle}</h2>
              </div>
              <span>{activeGraphSubtitle}</span>
            </div>
            {activePlotUrl ? (
              <div className="plot-frame">
                <img id="ratioPlot" src={activePlotUrl} alt={`${activeGraphTitle} for logging rate`} />
              </div>
            ) : (
              <div className="empty-panel compact-empty">No logging graph yet.</div>
            )}
            <LineLegend visible={Boolean(activePlotUrl)} />
            <div className="result-actions">
              {graphPointCsvUrl && (
                <a className="icon-link" href={graphPointCsvUrl} target="_blank" rel="noreferrer">
                  <Download size={16} />
                  Graph Point CSV
                </a>
              )}
              {speedCsvUrl && (
                <a className="icon-link" href={speedCsvUrl} target="_blank" rel="noreferrer">
                  <Download size={16} />
                  Speed Split CSV
                </a>
              )}
            </div>
          </section>

          <section className="panel result-panel table-panel">
            <div className="section-heading">
              <Calculator size={17} />
              <h2>Median MARE Rows</h2>
            </div>
            <p className="plant-note">
              Paper MARE tracks paper1_isa_v5. Each condition is compared against its own published series:
              noise-free against supplement Table S7 (ET1, 2-100 ms), tension-only against supplement Fig. S6(b)
              at 50 Hz (E_Toggle, 1-100 ms), and dual-channel against the three points Fig. 2(a) prints
              (E_Toggle at 5, 20 and 50 ms). Cells the paper does not publish say so rather than showing a
              blank. The estimator changed in v5 and every campaign was re-run under it, so v4.1 numbers are
              not comparable.
            </p>
            <p className="plant-note">
              Published findings for reference: the interior optimum belongs to the dual-channel case and is
              produced by the velocity channel. Tension-only noise creates no interior optimum - in the paper
              its best setting is the finest 1 ms one for every excitation. Read a pooled dual-channel median
              only together with its slow/fast decomposition on the Speed split tab; the pooled valley
              represents neither sub-group.
            </p>
            {summary?.power_law_sensor_noise_caveat && (
              <p className="plant-note">{summary.power_law_sensor_noise_caveat}</p>
            )}
            <LoggingResultRows rows={rows} />
          </section>
        </section>
      </div>
    </>
  );
}

export default function App() {
  const [selectedPlantId, setSelectedPlantId] = useState('P01');
  const [baseUrl, setBaseUrl] = useState(storedApiBaseUrl);
  // Deep-linkable pages: #retuning etc. select the section on load, and the
  // hash tracks navigation so section links can be shared.
  const initialPage =
    typeof window !== 'undefined' && PAGES.some((page) => page.id === window.location.hash.slice(1))
      ? window.location.hash.slice(1)
      : 'simulation';
  const [activePage, setActivePage] = useState(initialPage);
  useEffect(() => {
    if (typeof window !== 'undefined') window.location.hash = activePage;
  }, [activePage]);
  const [status, setStatus] = useState('checking');
  const [metadata, setMetadata] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [result, setResult] = useState(null);
  const [artifactVersion, setArtifactVersion] = useState(0);
  const [simForm, setSimForm] = useState({
    duration_s: '4',
    log_sample_time_ms: '5',
    excitation: 'ET3',
    excitation_amplitude_V: '0.08',
    sensor_noise_tension_N: '0',
    sensor_noise_omega_rad_s: '0',
  });
  const [sysidForm, setSysidForm] = useState({
    duration_s: '4',
    log_sample_time_ms: '5',
    excitation: 'E_Toggle',
    excitation_amplitude_V: '0.08',
    sensor_noise_tension_N: '0',
    sensor_noise_omega_rad_s: '0',
  });
  useEffect(() => {
    storeApiBaseUrl(baseUrl);
  }, [baseUrl]);

  const excitationOptions = useMemo(
    () => (metadata?.excitation_profiles ?? ['ET1', 'ET3', 'ET6', 'E_Toggle']).filter((profile) => profile !== 'EVR'),
    [metadata],
  );
  const plantOptions = useMemo(
    () =>
      (metadata?.plants ?? []).map((plant) => ({
        value: plant.plant_id,
        label: plant.label,
      })),
    [metadata],
  );
  const plants = metadata?.plants ?? [];
  const selectedPlant = useMemo(
    () => plants.find((plant) => plant.plant_id === selectedPlantId),
    [plants, selectedPlantId],
  );

  async function refreshMetadata() {
    setStatus('checking');
    const requestedBaseUrl =
      migrateKnownStaleApiBase(baseUrl) || DEFAULT_API_BASE;
    let activeBaseUrl = requestedBaseUrl;
    try {
      let meta;
      try {
        meta = await loadCompatibleBackend(activeBaseUrl);
      } catch (initialError) {
        if (activeBaseUrl === DEFAULT_API_BASE) throw initialError;
        activeBaseUrl = DEFAULT_API_BASE;
        meta = await loadCompatibleBackend(activeBaseUrl);
      }
      if (activeBaseUrl !== baseUrl) setBaseUrl(activeBaseUrl);
      setStatus('online');
      setMetadata(meta);
      if (meta.default_plant_id && !meta.plants?.some((plant) => plant.plant_id === selectedPlantId)) {
        setSelectedPlantId(meta.single_plant_default_id ?? 'P01');
      }
      setError('');
    } catch {
      setStatus('offline');
      setError(
        `The validation backend is unavailable. Expected a service at ${DEFAULT_API_BASE}.`,
      );
    }
  }

  useEffect(() => {
    refreshMetadata();
  }, []);

  async function runRequest(path, body = {}) {
    setLoading(true);
    setError('');
    try {
      const payload = await apiPost(baseUrl, path, body);
      setResult(payload);
      setArtifactVersion(Date.now());
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  function numericForm(form) {
    return Object.fromEntries(
      Object.entries(form).map(([key, value]) => {
        if (key === 'excitation' || key === 'plant_id') return [key, value];
        return [key, Number(value)];
      }),
    );
  }

  function plantBody(body = {}) {
    return { ...body, plant_id: selectedPlantId };
  }

  function handlePlantSelect(plantId) {
    setSelectedPlantId(plantId);
    const plant = plants.find((item) => item.plant_id === plantId);
    if (!plant || plant.recommended_excitation_amplitude_V === undefined) return;
    const amplitude = String(plant.recommended_excitation_amplitude_V);
    setSimForm((current) => ({ ...current, excitation_amplitude_V: amplitude }));
    setSysidForm((current) => ({ ...current, excitation_amplitude_V: amplitude }));
  }

  function renderControls() {
    if (activePage === 'simulation') {
      return (
        <section className="panel controls-panel">
          <div className="field-grid">
            <Field label="Duration (s)" value={simForm.duration_s} onChange={(v) => setSimForm({ ...simForm, duration_s: v })} />
            {plantOptions.length > 0 && <SelectField label="Plant" value={selectedPlantId} options={plantOptions} onChange={handlePlantSelect} />}
            {selectedPlant?.simulation_note && <p className="plant-note">{selectedPlant.simulation_note}</p>}
            <Field label="Tlog (ms)" value={simForm.log_sample_time_ms} onChange={(v) => setSimForm({ ...simForm, log_sample_time_ms: v })} />
            <SelectField label="Excitation" value={simForm.excitation} options={excitationOptions} onChange={(v) => setSimForm({ ...simForm, excitation: v })} />
            <Field label="Amplitude (N)" value={simForm.excitation_amplitude_V} onChange={(v) => setSimForm({ ...simForm, excitation_amplitude_V: v })} />
            <Field label="T noise (N)" value={simForm.sensor_noise_tension_N} onChange={(v) => setSimForm({ ...simForm, sensor_noise_tension_N: v })} />
            <Field label="Omega noise" value={simForm.sensor_noise_omega_rad_s} onChange={(v) => setSimForm({ ...simForm, sensor_noise_omega_rad_s: v })} />
          </div>
          <RunButton loading={loading} onClick={() => runRequest('/simulate', plantBody(numericForm(simForm)))}>Run Simulation</RunButton>
        </section>
      );
    }

    if (activePage === 'sysid') {
      return (
        <section className="panel controls-panel">
          <div className="field-grid">
            <Field label="Duration (s)" value={sysidForm.duration_s} onChange={(v) => setSysidForm({ ...sysidForm, duration_s: v })} />
            {plantOptions.length > 0 && <SelectField label="Plant" value={selectedPlantId} options={plantOptions} onChange={handlePlantSelect} />}
            {selectedPlant?.simulation_note && <p className="plant-note">{selectedPlant.simulation_note}</p>}
            <Field label="Tlog (ms)" value={sysidForm.log_sample_time_ms} onChange={(v) => setSysidForm({ ...sysidForm, log_sample_time_ms: v })} />
            <SelectField label="Excitation" value={sysidForm.excitation} options={excitationOptions} onChange={(v) => setSysidForm({ ...sysidForm, excitation: v })} />
            <Field label="Amplitude (N)" value={sysidForm.excitation_amplitude_V} onChange={(v) => setSysidForm({ ...sysidForm, excitation_amplitude_V: v })} />
            <Field label="T noise (N)" value={sysidForm.sensor_noise_tension_N} onChange={(v) => setSysidForm({ ...sysidForm, sensor_noise_tension_N: v })} />
            <Field label="Omega noise" value={sysidForm.sensor_noise_omega_rad_s} onChange={(v) => setSysidForm({ ...sysidForm, sensor_noise_omega_rad_s: v })} />
          </div>
          <RunButton loading={loading} onClick={() => runRequest('/sysid', plantBody(numericForm(sysidForm)))}>Run SysID</RunButton>
        </section>
      );
    }

    const validationRoutes = {
      excitation: ['/validate/excitation', 'Run Excitation Study'],
      noiseLpf: ['/validate/noise-aware-logging-lpf', 'Run Noise-aware logging (LPF)'],
      drift: ['/validate/drift', 'Run Drift Study'],
    };
    if (validationRoutes[activePage]) {
      const [path, label] = validationRoutes[activePage];
      return (
        <section className="panel controls-panel compact-controls">
          <RunButton loading={loading} onClick={() => runRequest(path, plantBody())}>{label}</RunButton>
        </section>
      );
    }
    return null;
  }

  const ActiveIcon = PAGES.find((page) => page.id === activePage)?.icon ?? Activity;

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand-lockup">
          <Gauge size={26} />
          <div>
            <strong>R2R SysID</strong>
            <span>Validation</span>
          </div>
        </div>
        <nav className="page-nav">
          {PAGES.map(({ id, label, icon: Icon }) => (
            <button className={activePage === id ? 'active' : ''} type="button" onClick={() => setActivePage(id)} key={id}>
              <Icon size={18} />
              <span>{label}</span>
            </button>
          ))}
        </nav>
      </aside>

      <main className="workspace">
        <header className="topbar">
          <div className="title-block">
            <ActiveIcon size={24} />
            <div>
              <h1>{PAGES.find((page) => page.id === activePage)?.label}</h1>
              <span className={`status-pill ${status}`}>{status}</span>
            </div>
          </div>
          <div className="topbar-actions">
            <div className="api-controls">
              <input
                value={baseUrl}
                onChange={(event) =>
                  setBaseUrl(
                    migrateKnownStaleApiBase(event.target.value) ??
                      event.target.value,
                  )
                }
                aria-label="API base URL"
              />
              <button className="icon-button" type="button" onClick={refreshMetadata} title="Refresh API status">
                <RefreshCw size={17} />
              </button>
            </div>
          </div>
        </header>

        <ErrorBanner message={error} />

        {activePage === 'plants' ? (
          <PlantPage plants={plants} selectedPlantId={selectedPlantId} onSelect={handlePlantSelect} />
        ) : activePage === 'logging' ? (
          <LoggingPage
            baseUrl={baseUrl}
            metadata={metadata}
            plantOptions={plantOptions}
            excitationOptions={excitationOptions}
          />
        ) : activePage === 'excitation' ? (
          <ExcitationPage
            baseUrl={baseUrl}
            plantOptions={plantOptions}
            excitationOptions={excitationOptions}
          />
        ) : activePage === 'noiseLpf' ? (
          <div className="work-grid noise-lpf-work-grid">
            {renderControls()}
            <NoiseLpfResult baseUrl={baseUrl} result={result} artifactVersion={artifactVersion} />
          </div>
        ) : activePage === 'damping' ? (
          <DampingPage
            baseUrl={baseUrl}
            plantOptions={plantOptions}
            excitationOptions={excitationOptions}
          />
        ) : activePage === 'retuning' ? (
          <RetuningPage baseUrl={baseUrl} />
        ) : activePage === 'equations' ? (
          <EquationPage />
        ) : activePage === 'drift' ? (
          <DriftPage
            baseUrl={baseUrl}
            selectedPlantId={selectedPlantId}
            plantOptions={plantOptions}
            selectedPlant={selectedPlant}
            onPlantSelect={handlePlantSelect}
            excitationOptions={excitationOptions}
          />
        ) : activePage === 'simulation' ? (
          <div className="simulation-page">
            <R2RSchematic />
            <div className="work-grid">
              {renderControls()}
              <ResultPanel baseUrl={baseUrl} result={result} />
            </div>
          </div>
        ) : (
          <div className="work-grid">
            {renderControls()}
            <ResultPanel baseUrl={baseUrl} result={result} />
          </div>
        )}
      </main>
    </div>
  );
}

