function formatValue(value) {
  if (value === null || value === undefined) return '';
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) return String(value);
    if (value === 0) return '0';
    if (Math.abs(value) >= 1000 || Math.abs(value) < 0.001) return value.toExponential(3);
    return value.toFixed(4).replace(/0+$/, '').replace(/\.$/, '');
  }
  if (typeof value === 'boolean') return value ? 'true' : 'false';
  return String(value);
}

const HIDDEN_KEYS = new Set([
  '__provenance',
  'source',
  'path',
  'url',
  'file',
  'file_name',
  'filename',
  'reference_tau_ratio',
  'tmin_adjusted_model_note',
  'local_path',
  'value_status',
  'run_timestamp_utc',
  'code_version',
  'plant_scope',
  'run_settings_json',
  'tau_min_source',
  'data_source',
  'display_adjustment_type',
  'paper_reference_type',
  'logging_fit_display_note',
  'dashboard_metric_equation',
  'paper_note',
  'dashboard_iteration_note',
]);

const PROVENANCE_LABELS = {
  computed_raw: 'calc',
  dashboard_simulation: 'sim',
  dashboard_simulation_median: 'sim median',
  computed_dashboard_minus_reference: 'calc diff',
  computed_adjusted: 'adjusted',
  paper_reference: 'paper',
  configured_value: 'setting',
  method_setting: 'method',
  interpolated_reference: 'interp',
  mixed: 'mixed',
  mixed_raw_and_adjusted: 'mixed',
};

function displayLabel(key) {
  return String(key)
    .replace(/^raw_/, '')
    .replace(/raw_dashboard_/g, 'dashboard_')
    .replace(/displayed_dashboard_/g, 'dashboard_')
    .replace(/reference_power_law_/g, 'paper_model_')
    .replace(/paper_reference_/g, 'paper_');
}

function isDisplayKey(key) {
  const normalized = String(key).toLowerCase();
  if (HIDDEN_KEYS.has(normalized)) return false;
  if (normalized.startsWith('raw_dashboard_') || normalized.startsWith('displayed_dashboard_')) return false;
  if (normalized.endsWith('_path') || normalized.endsWith('_url') || normalized.endsWith('_file')) return false;
  if (normalized.includes('reference_file') || normalized.includes('local_path')) return false;
  return true;
}

function inferProvenance(key, row) {
  const explicit = row?.__provenance?.[key];
  if (explicit) return explicit;
  const normalized = String(key).toLowerCase();
  if (normalized.includes('paper') || normalized.includes('reference_power_law')) return 'paper_reference';
  if (normalized.includes('interpolated')) return 'interpolated_reference';
  if (normalized.includes('assumed') || normalized.includes('tau_min_source')) {
    return 'configured_value';
  }
  if (normalized.includes('display_adjustment')) {
    return 'method_setting';
  }
  if (normalized.includes('raw_dashboard') || normalized.includes('dashboard') || normalized.includes('rmse') || normalized.includes('difference')) {
    return row?.value_status ?? 'computed_raw';
  }
  return row?.value_status ?? 'computed_raw';
}

function provenanceForColumn(column, rows) {
  const values = [...new Set(rows.map((row) => inferProvenance(column, row)).filter(Boolean))];
  if (values.length === 0) return null;
  if (values.length === 1) return values[0];
  return 'mixed';
}

function HeaderWithBadge({ label, provenance }) {
  const badgeLabel = PROVENANCE_LABELS[provenance] ?? provenance;
  return (
    <span className="provenance-header">
      <span>{label}</span>
      {badgeLabel && <span className={`provenance-badge ${provenance}`}>{badgeLabel}</span>}
    </span>
  );
}

export default function MetricTable({ rows }) {
  if (!rows) return null;

  if (!Array.isArray(rows)) {
    const entries = Object.entries(rows).filter(([key]) => isDisplayKey(key));
    if (entries.length === 0) return null;
    return (
      <table className="data-table">
        <tbody>
          {entries.map(([key, value]) => (
            <tr key={key}>
              <th><HeaderWithBadge label={displayLabel(key)} provenance={inferProvenance(key, rows)} /></th>
              <td>{formatValue(value)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    );
  }

  if (rows.length === 0) return null;
  const columns = Object.keys(rows[0]).filter(
    (column) => isDisplayKey(column) && rows.some((row) => row[column] !== null && row[column] !== undefined),
  );
  if (columns.length === 0) return null;
  return (
    <div className="table-scroll">
      <table className="data-table">
        <thead>
          <tr>
            {columns.map((column) => (
              <th key={column}><HeaderWithBadge label={displayLabel(column)} provenance={provenanceForColumn(column, rows)} /></th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, index) => (
            <tr key={`${index}-${JSON.stringify(row).slice(0, 12)}`}>
              {columns.map((column) => (
                <td key={column}>{formatValue(row[column])}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
