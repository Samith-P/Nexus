import { useState, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import './LiteratureReview.css';

/* ─── helpers ─────────────────────────────────────── */
function cleanTitle(raw = '') {
  return raw.split('\n')[0].trim();
}

function formatAuthors(authors = []) {
  if (!authors.length) return null;
  if (authors.length <= 5) return authors.join(', ');
  return authors.slice(0, 5).join(', ') + ` +${authors.length - 5} more`;
}

// Collapse excessive whitespace/newlines that come from raw PDF extraction
function cleanText(raw = '') {
  if (!raw) return '';
  return raw
    .replace(/\r\n/g, '\n')
    // Remove hyphenation across lines
    .replace(/(\w)-\n(\w)/g, '$1$2')
    // Collapse horizontal whitespace
    .replace(/[ \t]+/g, ' ')
    // Replace single newlines with space (preserves double newlines)
    .replace(/([^\n])\n(?=[^\n])/g, '$1 ')
    // Collapse multiple newlines into paragraph breaks (max 2)
    .replace(/\n{3,}/g, '\n\n')
    // Strip leading spaces on new lines
    .replace(/\n +/g, '\n')
    .trim();
}

/* ─── Accordion for a single paper section ─────────── */
function SectionAccordion({ label, summary, fullText }) {
  const [open, setOpen] = useState(false);
  if (!summary && !fullText) return null;

  const cleanedExcerpt = cleanText(fullText || '');

  return (
    <div className="lr-accordion-item">
      <button
        className={`lr-accordion-btn${open ? ' open' : ''}`}
        onClick={() => setOpen(o => !o)}
        aria-expanded={open}
      >
        <span className="lr-accordion-label">{label}</span>
        <span className={`lr-accordion-chevron${open ? ' open' : ''}`}>▼</span>
      </button>

      {open && (
        <div className="lr-accordion-body">

          {/* AI Summary block */}
          {summary && (
            <div className="lr-acc-summary">
              <div className="lr-acc-summary-header">
                <span className="lr-acc-tag">✦ AI Summary</span>
              </div>
              <p className="lr-acc-summary-text">{cleanText(summary)}</p>
            </div>
          )}

          {/* Collapsible raw excerpt */}
          {cleanedExcerpt && (
            <details className="lr-acc-excerpt-details">
              <summary className="lr-acc-excerpt-toggle">Show source excerpt</summary>
              <blockquote className="lr-acc-excerpt">
                {cleanedExcerpt}
              </blockquote>
            </details>
          )}

        </div>
      )}
    </div>
  );
}

/* ─── Single paper card ──────────────────────────── */
function PaperCard({ paper, index }) {
  const title = cleanTitle(paper.title);
  const { sections = {}, section_summaries = {}, insights = {}, gaps = [], evidence_spans = [], summary } = paper;

  const sectionKeys = Object.keys(section_summaries).filter(k => k !== 'title');

  return (
    <div className="lr-paper-card">
      {/* Header */}
      <div className="lr-paper-header">
        <h3 className="lr-paper-title">{title}</h3>
        <span className="lr-paper-idx">Paper {index + 1}</span>
      </div>

      {/* Overall summary */}
      {summary && (
        <div className="lr-summary">
          <div className="lr-summary-label">Overall Summary</div>
          <p className="lr-summary-text">{cleanText(summary)}</p>
        </div>
      )}

      {/* Section accordions */}
      {sectionKeys.length > 0 && (
        <div className="lr-section-block">
          <div className="lr-sub-heading">📄 Section Breakdowns</div>
          <div className="lr-accordion">
            {sectionKeys.map(key => (
              <SectionAccordion
                key={key}
                label={key.charAt(0).toUpperCase() + key.slice(1)}
                summary={section_summaries[key]}
                fullText={sections[key]}
              />
            ))}
          </div>
        </div>
      )}

      {/* Insights */}
      {insights && (insights.contributions?.length > 0 || insights.methods?.length > 0 || insights.results?.length > 0) && (
        <div className="lr-section-block">
          <div className="lr-sub-heading">💡 Key Insights</div>
          <div className="lr-insights">
            {insights.contributions?.length > 0 && (
              <div className="lr-insight-group">
                <div className="lr-insight-label">🎯 Contributions</div>
                <ul className="lr-insight-list">
                  {insights.contributions.map((c, i) => <li key={i}>{c}</li>)}
                </ul>
              </div>
            )}
            {insights.methods?.length > 0 && (
              <div className="lr-insight-group">
                <div className="lr-insight-label">🔬 Methods</div>
                <ul className="lr-insight-list">
                  {insights.methods.map((m, i) => <li key={i}>{m}</li>)}
                </ul>
              </div>
            )}
            {insights.results?.length > 0 && (
              <div className="lr-insight-group">
                <div className="lr-insight-label">📊 Results</div>
                <ul className="lr-insight-list">
                  {insights.results.map((r, i) => <li key={i}>{r}</li>)}
                </ul>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Evidence spans */}
      {evidence_spans?.length > 0 && (
        <div className="lr-section-block">
          <div className="lr-sub-heading">🔍 Evidence Spans</div>
          <div className="lr-evidence-list">
            {evidence_spans.map((ev, i) => (
              <div key={i} className="lr-evidence-item">
                <span className="lr-evidence-section-tag">{ev.section}</span>
                <div className="lr-evidence-text">"{cleanText(ev.text || '').slice(0, 300)}…"</div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

/* ─── Related Works card ─────────────────────────── */
function RelatedCard({ work }) {
  const title = cleanTitle(work.title);
  const authors = formatAuthors(work.authors);

  return (
    <div className="lr-related-card">
      <div className="lr-related-title">{title}</div>
      {authors && <div className="lr-related-authors">👤 {authors}</div>}
      {work.abstract && <div className="lr-related-abstract">{work.abstract}</div>}

      <div className="lr-related-meta">
        {work.year && <span className="lr-related-badge">📅 {work.year}</span>}
        {work.citation_count != null && (
          <span className="lr-related-badge">📚 {work.citation_count.toLocaleString()} citations</span>
        )}
        {work.source && <span className="lr-related-badge">{work.source}</span>}
      </div>

      {work.url && (
        <a href={work.url} target="_blank" rel="noreferrer" className="lr-related-link">
          View Paper ↗
        </a>
      )}
    </div>
  );
}

/* ─── Main page ──────────────────────────────────── */
export default function LiteratureReview() {
  const navigate = useNavigate();
  const fileInputRef = useRef(null);

  const [file, setFile] = useState(null);
  const [dragOver, setDragOver] = useState(false);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);

  /* drag & drop */
  const handleDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    const dropped = e.dataTransfer.files[0];
    if (dropped && dropped.type === 'application/pdf') {
      setFile(dropped);
      setResults(null);
      setError(null);
    } else {
      setError('Please drop a valid PDF file.');
    }
  };

  const handleFileChange = (e) => {
    const picked = e.target.files[0];
    if (picked) {
      setFile(picked);
      setResults(null);
      setError(null);
    }
  };

  const removeFile = () => {
    setFile(null);
    setResults(null);
    setError(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  /* submit */
  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) return;

    setLoading(true);
    setError(null);
    setResults(null);

    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch('http://localhost:8000/api/literature/review/sync', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail || `Server error: ${response.status}`);
      }

      const data = await response.json();
      setResults(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const papers = results?.papers || [];
  const relatedWorks = results?.related_works || [];
  const researchGaps = results?.research_gaps || [];
  const processingTime = results?.processing_time_seconds;

  return (
    <div className="lr-container">
      <div className="lr-content-wrapper">

        {/* Back button */}
        <button className="lr-back-btn" onClick={() => navigate('/')}>
          ← Back to Home
        </button>

        {/* Header */}
        <div className="lr-header">
          <h1 className="lr-title">Literature Review</h1>
          <p className="lr-subtitle">
            Upload a research PDF and get an automated structured review — summaries,
            key insights, section breakdowns, related works, and research gaps.
          </p>
        </div>

        {/* Upload card */}
        <div className="lr-upload-card">
          <form onSubmit={handleSubmit}>
            <div
              className={`lr-dropzone${dragOver ? ' drag-over' : ''}`}
              onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
              onDragLeave={() => setDragOver(false)}
              onDrop={handleDrop}
            >
              <span className="lr-dropzone-icon">📄</span>
              <div className="lr-dropzone-text">
                {file ? file.name : 'Drop your PDF here'}
              </div>
              <div className="lr-dropzone-sub">
                {file
                  ? `${(file.size / 1024 / 1024).toFixed(2)} MB`
                  : <>or <span>click to browse</span> — PDF only</>
                }
              </div>
              <input
                ref={fileInputRef}
                type="file"
                accept="application/pdf"
                className="lr-file-input"
                onChange={handleFileChange}
                disabled={loading}
              />
            </div>

            {file && (
              <div style={{ textAlign: 'center' }}>
                <span className="lr-file-chip">
                  📎 {file.name}
                  <button type="button" onClick={removeFile} title="Remove file">✕</button>
                </span>
              </div>
            )}

            <button
              type="submit"
              className="lr-submit-btn"
              disabled={!file || loading}
            >
              {loading ? 'Analyzing…' : '🔍 Run Literature Review'}
            </button>
          </form>
        </div>

        {/* Error */}
        {error && !loading && (
          <div className="lr-error">⚠️ {error}</div>
        )}

        {/* Loading */}
        {loading && (
          <div className="lr-loading">
            <div className="lr-spinner" />
            Running literature review…
            <div className="lr-loading-hint">
              This can take up to a minute. We're extracting sections,
              generating summaries, and fetching related works.
            </div>
          </div>
        )}

        {/* Results */}
        {!loading && results && (
          <div className="lr-results-wrapper">

            {/* Stats bar */}
            <div className="lr-stats-bar">
              <div className="lr-stat-chip">
                📄 Papers analysed: <strong>{papers.length}</strong>
              </div>
              <div className="lr-stat-chip">
                🔗 Related works: <strong>{relatedWorks.length}</strong>
              </div>
              <div className="lr-stat-chip">
                ⚠️ Research gaps: <strong>{researchGaps.length}</strong>
              </div>
              {processingTime != null && (
                <div className="lr-stat-chip">
                  ⏱ Processed in: <strong>{processingTime.toFixed(1)}s</strong>
                </div>
              )}
            </div>

            {/* Papers */}
            {papers.length > 0 && (
              <section>
                <div className="lr-section-title">📚 Paper Analysis</div>
                {papers.map((paper, idx) => (
                  <PaperCard key={idx} paper={paper} index={idx} />
                ))}
              </section>
            )}

            {/* Global Research Gaps */}
            {researchGaps.length > 0 && (
              <section>
                <div className="lr-section-title">🔭 Research Gaps</div>
                <div className="lr-global-gaps">
                  {researchGaps.map((gap, i) => (
                    <div key={i} className="lr-global-gap-item">
                      <span className="lr-global-gap-icon">⚠️</span>
                      <span>{gap}</span>
                    </div>
                  ))}
                </div>
              </section>
            )}

            {/* Related Works */}
            {relatedWorks.length > 0 && (
              <section>
                <div className="lr-section-title">🌐 Related Works</div>
                <div className="lr-related-grid">
                  {relatedWorks.map((work, idx) => (
                    <RelatedCard key={idx} work={work} />
                  ))}
                </div>
              </section>
            )}

          </div>
        )}
      </div>
    </div>
  );
}
