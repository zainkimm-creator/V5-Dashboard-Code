import { useState } from 'react';
import { Calculator, Download, FileSpreadsheet, FileText, Image as ImageIcon } from 'lucide-react';
import { artifactUrl } from '../src/api.js';
import MetricTable from './MetricTable.jsx';

export default function ResultPanel({ baseUrl, result }) {
  const [showCalculations, setShowCalculations] = useState(false);

  if (!result) {
    return <section className="panel empty-panel">No run selected.</section>;
  }

  const csvUrl = artifactUrl(baseUrl, result.csv_url);
  const xlsxUrl = artifactUrl(baseUrl, result.xlsx_url);
  const plotUrl = artifactUrl(baseUrl, result.plot_url);
  const summaryUrl = artifactUrl(baseUrl, result.summary_url);
  const markdownUrl = artifactUrl(baseUrl, result.markdown_url);
  const resultMetrics = result.metrics;
  const summaryMetrics =
    resultMetrics && resultMetrics.metrics
      ? Object.fromEntries(Object.entries(resultMetrics).filter(([key]) => key !== 'metrics'))
      : null;
  const metrics = resultMetrics?.metrics ?? resultMetrics;
  const calculations = result.calculations ?? [];
  const hasCalculations = calculations.length > 0;

  return (
    <section className="panel result-panel">
      <div className="result-actions">
        {csvUrl && (
          <a className="icon-link" href={csvUrl} target="_blank" rel="noreferrer">
            <Download size={16} /> CSV
          </a>
        )}
        {xlsxUrl && (
          <a className="icon-link" href={xlsxUrl} target="_blank" rel="noreferrer">
            <FileSpreadsheet size={16} /> XLSX
          </a>
        )}
        {plotUrl && (
          <a className="icon-link" href={plotUrl} target="_blank" rel="noreferrer">
            <ImageIcon size={16} /> Plot
          </a>
        )}
        {summaryUrl && (
          <a className="icon-link" href={summaryUrl} target="_blank" rel="noreferrer">
            <FileText size={16} /> Summary
          </a>
        )}
        {markdownUrl && (
          <a className="icon-link" href={markdownUrl} target="_blank" rel="noreferrer">
            <FileText size={16} /> Report
          </a>
        )}
        {hasCalculations && (
          <button
            className={`icon-link ${showCalculations ? 'active-action' : ''}`}
            type="button"
            onClick={() => setShowCalculations((value) => !value)}
          >
            <Calculator size={16} /> Calculation
          </button>
        )}
      </div>

      {showCalculations && hasCalculations && (
        <div className="calculation-panel">
          <div className="calculation-heading">
            <h2>Calculation Examples</h2>
            {result.calculation_summary && <p>{result.calculation_summary}</p>}
          </div>
          <div className="calculation-list">
            {calculations.map((calculation, index) => (
              <article className="calculation-card" key={`${calculation.parameter}-${index}`}>
                <div className="calculation-card-header">
                  <h3>{calculation.title}</h3>
                  <span>{calculation.parameter}</span>
                </div>
                {calculation.summary && <p>{calculation.summary}</p>}
                <p className="formula">{calculation.formula}</p>
                {calculation.steps?.length > 0 && (
                  <ol className="calculation-steps">
                    {calculation.steps.map((step) => (
                      <li key={step}>{step}</li>
                    ))}
                  </ol>
                )}
                {calculation.values && <MetricTable rows={calculation.values} />}
                {calculation.substitution && <p className="formula">{calculation.substitution}</p>}
                {calculation.result && <p className="calculation-result">{calculation.result}</p>}
              </article>
            ))}
          </div>
        </div>
      )}

      {plotUrl && (
        <div className="plot-frame">
          <img src={plotUrl} alt="Validation plot" />
        </div>
      )}

      <div className="result-grid">
        {summaryMetrics && <MetricTable rows={summaryMetrics} />}
        {metrics && !Array.isArray(metrics) && <MetricTable rows={metrics} />}
        {Array.isArray(metrics) && <MetricTable rows={metrics} />}
        {result.estimates && <MetricTable rows={result.estimates} />}
        {result.error_table && <MetricTable rows={result.error_table} />}
        {result.comparison_table && <MetricTable rows={result.comparison_table} />}
        {result.preview_rows && <MetricTable rows={result.preview_rows} />}
      </div>
    </section>
  );
}
