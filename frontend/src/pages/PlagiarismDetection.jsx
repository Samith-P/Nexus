import { useState, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import './PlagiarismDetection.css';

/* ─── helpers ─────────────────────────────────────── */
function cleanText(raw = '') {
  if (!raw) return '';
  return raw
    .replace(/\r\n/g, '\n')
    .replace(/(\w)-\n(\w)/g, '$1$2')
    .replace(/[ \t]+/g, ' ')
    .replace(/([^\n])\n(?=[^\n])/g, '$1 ')
    .replace(/\n{3,}/g, '\n\n')
    .replace(/\n +/g, '\n')
    .trim();
}

function getScoreClass(score, isOriginality = false) {
  // If Originality Score: high is good, low is bad.
  // If Plagiarism Score: low is good, high is bad.
  if (isOriginality) {
    if (score >= 80) return 'success';
    if (score >= 50) return 'warning';
    return 'danger';
  } else {
    // Plagiarism percentage
    if (score <= 20) return 'success';
    if (score <= 50) return 'warning';
    return 'danger';
  }
}

export default function PlagiarismDetection() {
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
      formData.append('user_id', 'demo-user-1');
      formData.append('check_type', 'full');

      const response = await fetch('http://localhost:8000/plagiarism/check', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail || `Server error: ${response.status}`);
      }

      const data = await response.json();
      try {
        const token = localStorage.getItem("nexus_access_token"); // or wherever you store JWT

        await fetch("http://localhost:8000/usage", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "Authorization": `Bearer ${token}`
          },
          body: JSON.stringify({
            api_name: "plag_detection",
            request_data: {
              filename: file.name
            },
            response_data: data
          })
        });

      } catch (err) {
        console.error("Usage logging failed:", err);
      }
      setResults(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="pd-container">
      <div className="pd-content-wrapper">

        {/* Back button */}
        <button className="pd-back-btn" onClick={() => navigate('/')}>
          ← Back to Home
        </button>

        {/* Header */}
        <div className="pd-header">
          <h1 className="pd-title">Plagiarism Detection</h1>
          <p className="pd-subtitle">
            Scan your document against millions of published papers and web sources
            to ensure academic integrity, detect similarities, and identify missing citations.
          </p>
        </div>

        {/* Upload card */}
        <div className="pd-upload-card">
          <form onSubmit={handleSubmit}>
            <div
              className={`pd-dropzone${dragOver ? ' drag-over' : ''}`}
              onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
              onDragLeave={() => setDragOver(false)}
              onDrop={handleDrop}
            >
              <span className="pd-dropzone-icon">🛡️</span>
              <div className="pd-dropzone-text">
                {file ? file.name : 'Drop your PDF here'}
              </div>
              <div className="pd-dropzone-sub">
                {file
                  ? `${(file.size / 1024 / 1024).toFixed(2)} MB`
                  : <>or <span>click to browse</span> — PDF only</>
                }
              </div>
              <input
                ref={fileInputRef}
                type="file"
                accept="application/pdf"
                className="pd-file-input"
                onChange={handleFileChange}
                disabled={loading}
              />
            </div>

            {file && (
              <div style={{ textAlign: 'center' }}>
                <span className="pd-file-chip">
                  📎 {file.name}
                  <button type="button" onClick={removeFile} title="Remove file">✕</button>
                </span>
              </div>
            )}

            <button
              type="submit"
              className="pd-submit-btn"
              disabled={!file || loading}
            >
              {loading ? 'Scanning document…' : '🔍 Run Plagiarism Check'}
            </button>
          </form>
        </div>

        {/* Error */}
        {error && !loading && (
          <div className="pd-error">⚠️ {error}</div>
        )}

        {/* Loading */}
        {loading && (
          <div className="pd-loading">
            <div className="pd-spinner" />
            Analyzing for plagiarism…
            <div className="pd-loading-hint">
              This might take a moment. We're breaking down your document,
              extracting semantics, and matching against external databases.
            </div>
          </div>
        )}

        {/* Results */}
        {!loading && results && (
          <div className="pd-results-wrapper">

            {/* Score Board */}
            <div className="pd-score-board">
              <div className={`pd-score-card ${getScoreClass(results.originality_score, true)}`}>
                <div className="pd-score-label">Originality Score</div>
                <div className="pd-score-value">{results.originality_score}%</div>
                <div className="pd-score-label" style={{ fontSize: '11px', color: '#8c6a00' }}>
                  Coverage: {results.citation_coverage_percentage}%
                </div>
              </div>

              <div className={`pd-score-card ${getScoreClass(results.plagiarism_percentage, false)}`}>
                <div className="pd-score-label">Plagiarism Detected</div>
                <div className="pd-score-value">{results.plagiarism_percentage}%</div>
                <div className="pd-score-label" style={{ fontSize: '11px', color: '#8c6a00' }}>
                  Semantic Match: {results.semantic_match_percentage}%
                </div>
              </div>
            </div>

            {/* Matches Section */}
            {results.matches && results.matches.length > 0 && (
              <div className="pd-section-block">
                <div className="pd-sub-heading">📑 Discovered Matches</div>
                <div className="pd-match-list">
                  {results.matches.map((match, idx) => {
                    const simPercent = (match.similarity * 100).toFixed(1);
                    return (
                      <div key={idx} className="pd-match-card">
                        <div className="pd-match-header">
                          <div className="pd-match-source">
                            🌐 {match.source || 'External Source'}
                          </div>
                          <div className={`pd-match-similarity ${match.similarity < 0.8 ? 'moderate' : ''}`}>
                            {simPercent}% Match
                          </div>
                        </div>

                        <div className="pd-match-text">
                          "{cleanText(match.text)}"
                        </div>

                        <div className="pd-match-footer">
                          {match.type && (
                            <span className="pd-match-badge">Type: {match.type}</span>
                          )}
                          {match.doi && (
                            <span className="pd-match-badge">DOI: {match.doi}</span>
                          )}
                          {match.url && (
                            <a href={match.url} target="_blank" rel="noreferrer" className="pd-match-link">
                              View Source ↗
                            </a>
                          )}
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

            {/* Missing Citations Section */}
            {results.missing_citations && results.missing_citations.length > 0 && (
              <div className="pd-section-block">
                <div className="pd-sub-heading">⚠️ Missing Citations</div>
                <div className="pd-match-list">
                  {results.missing_citations.map((cite, idx) => (
                    <div key={idx} className="pd-match-card" style={{ borderColor: 'rgba(245, 158, 11, 0.4)' }}>
                      <div className="pd-match-header">
                        <div className="pd-match-source" style={{ color: '#b45309' }}>
                          Suggested Citation Source
                        </div>
                      </div>

                      <div className="pd-match-text">
                        "{cleanText(cite.text)}"
                      </div>

                      <div className="pd-match-footer">
                        {cite.suggested_source && (
                          <span className="pd-match-badge">Citation: {cite.suggested_source}</span>
                        )}
                        {cite.doi && (
                          <span className="pd-match-badge">DOI: {cite.doi}</span>
                        )}
                        {cite.url && (
                          <a href={cite.url} target="_blank" rel="noreferrer" className="pd-match-link">
                            View Reference ↗
                          </a>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {results.matches && results.matches.length === 0 && (
              <div className="pd-section-block" style={{ textAlign: 'center', padding: '60px 20px', color: '#10b981' }}>
                <span style={{ fontSize: '48px', display: 'block', marginBottom: '16px' }}>🎉</span>
                <div className="pd-sub-heading" style={{ justifyContent: 'center' }}>No Plagiarism Detected!</div>
                <p>Your document appears to be 100% original based on our semantic and exact-match indexing algorithms.</p>
              </div>
            )}

          </div>
        )}
      </div>
    </div>
  );
}
