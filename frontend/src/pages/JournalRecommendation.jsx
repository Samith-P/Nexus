import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import './JournalRecommendation.css';

export default function JournalRecommendation() {
  const navigate = useNavigate();
  const [abstract, setAbstract] = useState('');
  const [topK, setTopK] = useState(10);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [metadata, setMetadata] = useState(null);
  const [error, setError] = useState(null);

  const handleSearch = async (e) => {
    e.preventDefault();
    if (!abstract.trim()) return;

    setLoading(true);
    setError(null);

    try {
      const response = await fetch('http://localhost:8000/journals/recommend', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          abstract: abstract,
          top_k: parseInt(topK, 10) || 10
        })
      });

      if (!response.ok) {
        throw new Error('Failed to fetch journal recommendations. Please try again.');
      }

      const data = await response.json();
      setResults(data.journals || []);
      setMetadata(data.metadata || null);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const getQuartileClass = (quartile) => {
    if (!quartile) return '';
    return quartile.toLowerCase();
  };

  return (
    <div className="jr-container">
      <div className="jr-content-wrapper">
        <button className="jr-back-btn" onClick={() => navigate('/')}>
          ← Back to Home
        </button>

        <div className="jr-header">
          <h1 className="jr-title">Journal Recommendation</h1>
          <p className="jr-subtitle">
            Enter your research abstract or key topics and get intelligent recommendations
            for the best journals to submit your work — ranked by relevance and impact.
          </p>
        </div>

        <form className="jr-search-box" onSubmit={handleSearch}>
          <div className="jr-input-wrapper">
            <label htmlFor="abstract">Research Abstract</label>
            <textarea 
              id="abstract"
              className="jr-textarea" 
              placeholder="e.g., I am working on a project where we take a landing page and we check if the page is a clone page or not."
              value={abstract}
              onChange={(e) => setAbstract(e.target.value)}
              disabled={loading}
              required
            />
          </div>
          
          <div className="jr-settings-row">
            <div className="jr-input-wrapper">
              <label htmlFor="top_k">Results to Fetch</label>
              <input 
                id="top_k"
                type="number"
                min="1"
                max="50"
                className="jr-input" 
                value={topK}
                onChange={(e) => setTopK(e.target.value)}
                disabled={loading}
              />
            </div>

            <button type="submit" className="jr-button" disabled={loading || !abstract.trim()}>
              {loading ? 'Analyzing...' : 'Find Journals'}
            </button>
          </div>
        </form>

        {error && (
          <div style={{ textAlign: 'center', color: '#ff4d4f', marginBottom: 20 }}>
            {error}
          </div>
        )}

        {loading && (
          <div className="jr-loading">
            <div className="jr-spinner" />
            Analyzing semantic match and journal impact...
          </div>
        )}

        {!loading && results && (
          <div className="jr-results-wrapper">
            {metadata && (
              <div className="jr-metadata-bar">
                <span>Total Journals Evaluated: <strong>{metadata.total_journals_indexed?.toLocaleString()}</strong></span>
                <span>Scoring: <strong>{metadata.scoring_formula}</strong></span>
              </div>
            )}

            <div className="jr-results">
              {results.length === 0 ? (
                <div style={{ textAlign: 'center', color: '#4a5266' }}>
                  No journals found matching your abstract. Try adding more details.
                </div>
              ) : (
                results.map((journal, idx) => (
                  <div key={idx} className="jr-card">
                    <div className="jr-card-header">
                      <h3 className="jr-card-title">{journal.title}</h3>
                      <div className="jr-score-badge">
                        🏆 Score: {(journal.final_score * 100).toFixed(1)}%
                      </div>
                    </div>
                    
                    <div className="jr-meta">
                      {journal.quartile && (
                        <span className={`jr-meta-item ${getQuartileClass(journal.quartile)}`}>
                          📊 <strong>Quartile:</strong> {journal.quartile}
                        </span>
                      )}
                      {journal.sjr !== undefined && (
                        <span className="jr-meta-item">
                          ⭐ <strong>SJR:</strong> {journal.sjr}
                        </span>
                      )}
                      {journal.h_index !== undefined && (
                        <span className="jr-meta-item">
                          📈 <strong>H-Index:</strong> {journal.h_index}
                        </span>
                      )}
                      {journal.citations_per_doc_2y !== undefined && (
                        <span className="jr-meta-item">
                          📑 <strong>Citations/Doc (2y):</strong> {journal.citations_per_doc_2y}
                        </span>
                      )}
                    </div>

                    {journal.explanation && (
                      <div className="jr-explanation">
                        <div className="jr-explanation-title">Why this journal?</div>
                        <div>{journal.explanation}</div>
                      </div>
                    )}

                    <div className="jr-badges">
                      {journal.publisher && (
                        <span className="jr-badge">🏢 {journal.publisher}</span>
                      )}
                      {journal.country && (
                        <span className="jr-badge">🌍 {journal.country}</span>
                      )}
                      {journal.open_access === 'Yes' && (
                        <span className="jr-badge open-access">🔓 Open Access</span>
                      )}
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
