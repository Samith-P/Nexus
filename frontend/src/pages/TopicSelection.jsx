import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import './TopicSelection.css';

export default function TopicSelection() {
  const navigate = useNavigate();
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);

  const handleSearch = async (e) => {
    e.preventDefault();
    if (!query.trim()) return;

    setLoading(true);
    setError(null);

    try {
      const response = await fetch('http://localhost:8000/topic-recommendation', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: query,
          language: 'English',
          user_id: 'demo-user-1'
        })
      });

      if (!response.ok) {
        throw new Error('Failed to fetch recommendations. Please try again.');
      }

      const data = await response.json();
      setResults(data.recommended_topics || []);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="ts-container">
      <div className="ts-content-wrapper">
        <button className="ts-back-btn" onClick={() => navigate('/')}>
          ← Back to Home
        </button>

        <div className="ts-header">
          <h1 className="ts-title">Topic Selection Engine</h1>
          <p className="ts-subtitle">
            Discover high-impact, under-explored research topics based on current trends,
            citation gaps, and semantic analysis across millions of papers.
          </p>
        </div>

        <form className="ts-search-box" onSubmit={handleSearch}>
          <input 
            type="text" 
            className="ts-input" 
            placeholder="e.g., AI for crop yield prediction using satellite imagery"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            disabled={loading}
          />
          <button type="submit" className="ts-button" disabled={loading || !query.trim()}>
            {loading ? 'Analyzing...' : 'Discover'}
          </button>
        </form>

        {error && (
          <div style={{ textAlign: 'center', color: '#ff4d4f', marginBottom: 20 }}>
            {error}
          </div>
        )}

        {loading && (
          <div className="ts-loading">
            <div className="ts-spinner" />
            Analyzing research landscape...
          </div>
        )}

        {!loading && results && (
          <div className="ts-results">
            {results.length === 0 ? (
              <div style={{ textAlign: 'center', color: '#4a5266' }}>
                No recommendations found for this query. Try a different search term.
              </div>
            ) : (
              results.map((topic, idx) => (
                <div key={topic.topic_id || idx} className="ts-card">
                  <div className="ts-card-header">
                    <h3 className="ts-card-title">{topic.title}</h3>
                    <div className="ts-score-badge">
                      🏆 Score: {topic.final_score_100.toFixed(1)}%
                    </div>
                  </div>
                  
                  <div className="ts-meta">
                    <span className="ts-meta-item">
                      📚 <strong>Domain:</strong> {topic.domain}
                    </span>
                    {topic.year && (
                       <span className="ts-meta-item">
                         📅 <strong>Year:</strong> {topic.year}
                       </span>
                    )}
                    {topic.citations !== undefined && (
                       <span className="ts-meta-item">
                         📈 <strong>Citations:</strong> {topic.citations}
                       </span>
                    )}
                  </div>

                  {topic.reasons && topic.reasons.length > 0 && (
                    <div className="ts-reasons">
                      <div className="ts-reasons-title">Why this topic?</div>
                      <ul className="ts-reasons-list">
                        {topic.reasons.map((r, i) => (
                          <li key={i}>{r}</li>
                        ))}
                      </ul>
                    </div>
                  )}

                  {topic.keywords && topic.keywords.length > 0 && (
                    <div className="ts-keywords">
                      {topic.keywords.map((kw, i) => (
                        <span key={i} className="ts-keyword">{kw}</span>
                      ))}
                    </div>
                  )}
                </div>
              ))
            )}
          </div>
        )}
      </div>
    </div>
  );
}
