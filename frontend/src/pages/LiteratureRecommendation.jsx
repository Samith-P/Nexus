import { useNavigate } from 'react-router-dom';
import './FeaturePage.css';

export default function LiteratureRecommendation() {
  const navigate = useNavigate();
  return (
    <div className="feature-page-wrapper">
      <div className="feature-page-orb" style={{ background: 'radial-gradient(circle, #00d4ff, #4f8ef7)' }} />
      <div className="feature-page-content">
        <span className="feature-page-icon">🔬</span>
        <div className="feature-page-badge">Semantic</div>
        <h1 className="feature-page-title">Literature Recommendation</h1>
        <p className="feature-page-sub">
          Get a curated reading list of the most relevant papers, books, and preprints
          using vector-based semantic matching and citation graph analysis.
        </p>
        <button className="feature-page-back" onClick={() => navigate('/')} aria-label="Back to home">
          ← Back to Home
        </button>
        <div className="coming-tag">🚧 Interface coming soon</div>
      </div>
    </div>
  );
}
