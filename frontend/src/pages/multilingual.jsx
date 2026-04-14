import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import './multilingual.css';

export default function Multilingual() {
  const navigate = useNavigate();
  const [text, setText] = useState('');
  const [targetLang, setTargetLang] = useState('te');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleTranslate = async (e) => {
    e.preventDefault();
    if (!text.trim()) return;

    setLoading(true);
    setError(null);

    try {
      const response = await fetch('http://localhost:8000/translate-text', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          text: text,
          to: targetLang
        })
      });

      if (!response.ok) {
        throw new Error('Failed to translate text. Please try again.');
      }

      const data = await response.json();
      setResult(data.translated);

      // Call /usage API
      try {
        const token = localStorage.getItem("nexus_access_token");
        await fetch("http://localhost:8000/usage", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "Authorization": `Bearer ${token}`
          },
          body: JSON.stringify({
            api_name: "multilingual_translation",
            request_data: {
              text: text,
              to: targetLang
            },
            response_data: data
          })
        });
      } catch (err) {
        console.error("Usage logging failed:", err);
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="ml-container">
      <div className="ml-content-wrapper">
        <button className="ml-back-btn" onClick={() => navigate('/')}>
          ← Back to Home
        </button>

        <div className="ml-header">
          <h1 className="ml-title">Multilingual Translation</h1>
          <p className="ml-subtitle">
            Translate your research text into regional languages quickly and accurately 
            to broaden your scholarly reach.
          </p>
        </div>

        <form className="ml-search-box" onSubmit={handleTranslate}>
          <div className="ml-input-wrapper">
            <label htmlFor="text">Text to Translate</label>
            <textarea 
              id="text"
              className="ml-textarea" 
              placeholder="e.g., Greetings to the world"
              value={text}
              onChange={(e) => setText(e.target.value)}
              disabled={loading}
              required
            />
          </div>
          
          <div className="ml-settings-row">
            <div className="ml-input-wrapper">
              <label htmlFor="targetLang">Translate To</label>
              <select 
                id="targetLang"
                className="ml-select"
                value={targetLang}
                onChange={(e) => setTargetLang(e.target.value)}
                disabled={loading}
              >
                <option value="te">Telugu</option>
                <option value="hi">Hindi</option>
                <option value="ur">Urdu</option>
                <option value="en">English</option>
              </select>
            </div>

            <button type="submit" className="ml-button" disabled={loading || !text.trim()}>
              {loading ? 'Translating...' : 'Translate'}
            </button>
          </div>
        </form>

        {error && (
          <div style={{ textAlign: 'center', color: '#ff4d4f', marginBottom: 20 }}>
            {error}
          </div>
        )}

        {loading && (
          <div className="ml-loading">
            <div className="ml-spinner" />
            Translating text...
          </div>
        )}

        {!loading && result && (
          <div className="ml-results-wrapper">
            <div className="ml-card">
              <div className="ml-card-header">
                <h3 className="ml-card-title">Translated Text</h3>
                <div className="ml-score-badge">
                  ✨ Done
                </div>
              </div>
              <div className="ml-result-text">
                {result}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
