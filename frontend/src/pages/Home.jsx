import { useEffect, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import './Home.css';

// ── NAV CONFIG ────────────────────────────────────────────────────────────────
const NAV_ITEMS = [
  { label: 'Journal Recommender', icon: '📚', path: '/journal-recommendation' },
  { label: 'Topic Engine',        icon: '🧭', path: '/topic-selection'         },
  { label: 'Literature Finder',   icon: '🔬', path: '/literature-recommendation'},
  { label: 'Plagiarism Check',    icon: '🛡️', path: '/plagiarism-detection'    },
  { label: 'Multilingual Translation',    icon: '🛡️', path: '/multilingual'    },
];

// ── FEATURE CARDS CONFIG ──────────────────────────────────────────────────────
const FEATURES = [
  {
    id:       'journal',
    icon:     '📚',
    iconClass:'icon-blue',
    cardClass:'feature-card-blue',
    tag:      'AI-Powered',
    tagClass: 'tag-blue',
    ctaClass: 'cta-blue',
    dotClass: 'dot-blue',
    title:    'Journal Recommendation',
    desc:     'Discover the most relevant peer-reviewed journals for your research area. Our AI matches your manuscript\'s topic, methodology, and scope to the right publication venue.',
    bullets:  [
      'Matches based on research topic & scope',
      'Ranks journals by impact factor & relevance',
      'Covers 50,000+ indexed journals',
    ],
    path: '/journal-recommendation',
  },
  {
    id:       'topic',
    icon:     '🧭',
    iconClass:'icon-violet',
    cardClass:'feature-card-violet',
    tag:      'Smart',
    tagClass: 'tag-violet',
    ctaClass: 'cta-violet',
    dotClass: 'dot-violet',
    title:    'Topic Selection Engine',
    desc:     'Struggling to find your research focus? Our engine analyses research gaps, trend trajectories, and citation landscapes to suggest high-impact topics worth pursuing.',
    bullets:  [
      'Identifies under-explored research gaps',
      'Trend analysis across domains & years',
      'Semantic clustering of related topics',
    ],
    path: '/topic-selection',
  },
  {
    id:       'literature',
    icon:     '🔬',
    iconClass:'icon-cyan',
    cardClass:'feature-card-cyan',
    tag:      'Semantic',
    tagClass: 'tag-cyan',
    ctaClass: 'cta-cyan',
    dotClass: 'dot-cyan',
    title:    'Literature Recommendation',
    desc:     'Get a curated reading list of papers that are most relevant to your work. Powered by semantic similarity, citation graphs, and recency weighting.',
    bullets:  [
      'Vector-based semantic paper matching',
      'Citation graph traversal for hidden gems',
      'Filters by year, domain & access type',
    ],
    path: '/literature-recommendation',
  },
  {
    id:       'plagiarism',
    icon:     '🛡️',
    iconClass:'icon-indigo',
    cardClass:'feature-card-indigo',
    tag:      'Detection',
    tagClass: 'tag-indigo',
    ctaClass: 'cta-indigo',
    dotClass: 'dot-indigo',
    title:    'Plagiarism Detection',
    desc:     'Ensure academic integrity before submission. Our multi-layer detection checks against published papers, preprints, and web content for near-duplicate or paraphrased passages.',
    bullets:  [
      'Comparison against academic databases',
      'Paraphrase & near-duplicate detection',
      'Highlighted similarity report with sources',
    ],
    path: '/plagiarism-detection',
  },
];

// ── HOW IT WORKS STEPS ────────────────────────────────────────────────────────
const STEPS = [
  { num: '1', title: 'Describe Your Research',  desc: 'Enter your topic, abstract, or paste your draft to get started.' },
  { num: '2', title: 'AI Analysis',             desc: 'Our models analyse context, semantics, and literature landscapes.' },
  { num: '3', title: 'Get Recommendations',     desc: 'Receive tailored journals, topics, papers, or integrity reports.' },
  { num: '4', title: 'Iterate & Refine',        desc: 'Narrow results with filters and feedback to perfect your research.' },
];

// ─────────────────────────────────────────────────────────────────────────────

export default function Home() {
  const navigate  = useNavigate();
  const [scrolled, setScrolled] = useState(false);
  const revealRefs = useRef([]);
  const [token, setToken] = useState(null);
  const [userInitial, setUserInitial] = useState('U');

  // Fetch token & user info
  useEffect(() => {
    const access = localStorage.getItem('nexus_access_token');
    if (access) {
      setToken(access);
      fetch('http://localhost:8000/auth/me', {
         headers: { Authorization: `Bearer ${access}` }
      })
      .then(res => res.ok ? res.json() : null)
      .then(data => {
         if (data && data.full_name) {
             setUserInitial(data.full_name.charAt(0).toUpperCase());
         }
      })
      .catch(err => console.error(err));
    }
  }, []);

  // Navbar scroll shadow
  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 10);
    window.addEventListener('scroll', onScroll, { passive: true });
    return () => window.removeEventListener('scroll', onScroll);
  }, []);

  // Scroll-reveal observer
  useEffect(() => {
    const obs = new IntersectionObserver(
      (entries) => entries.forEach(e => { if (e.isIntersecting) e.target.classList.add('visible'); }),
      { threshold: 0.15 }
    );
    revealRefs.current.forEach(el => el && obs.observe(el));
    return () => obs.disconnect();
  }, []);

  const addReveal = (el) => { if (el && !revealRefs.current.includes(el)) revealRefs.current.push(el); };

  return (
    <div className="home-wrapper">
      {/* ── BACKGROUND ── */}
      <div className="bg-mesh" aria-hidden>
        <div className="orb orb-1" />
        <div className="orb orb-2" />
        <div className="orb orb-3" />
      </div>
      <div className="bg-grid" aria-hidden />

      {/* ══════════════════ NAVBAR ══════════════════ */}
      <nav className={`nexus-navbar${scrolled ? ' scrolled' : ''}`} role="navigation" aria-label="Main navigation">
        {/* Brand */}
        <div className="navbar-brand" onClick={() => navigate('/')} aria-label="Nexus home">
          <div className="brand-icon">✦</div>
          <span className="brand-name">Nexus</span>
        </div>

        {/* Links */}
        <ul className="navbar-links" role="list">
          {NAV_ITEMS.map(item => (
            <li key={item.path}>
              <button
                className="nav-link"
                onClick={() => navigate(item.path)}
                aria-label={`Navigate to ${item.label}`}
              >
                <span className="nav-icon" aria-hidden="true">{item.icon}</span>
                {item.label}
              </button>
            </li>
          ))}
        </ul>

        {/* CTA */}
        {token ? (
          <div 
             className="nav-profile-btn" 
             onClick={() => navigate('/profile')} 
             aria-label="User Profile"
             style={{
               width: '40px', height: '40px', borderRadius: '50%',
               background: 'linear-gradient(135deg, var(--accent-blue, #c5a059), var(--accent-violet, #8c6a00))',
               display: 'flex', alignItems: 'center', justifyContent: 'center',
               color: 'white', fontWeight: 'bold', fontSize: '1.2rem',
               cursor: 'pointer', boxShadow: '0 4px 12px rgba(197, 160, 89, 0.4)',
               border: '2px solid rgba(255,255,255,0.8)',
               transition: 'transform 0.2s, box-shadow 0.2s', marginLeft: '1rem'
             }}
             onMouseEnter={e => {
               e.currentTarget.style.transform = 'scale(1.08)';
               e.currentTarget.style.boxShadow = '0 6px 18px rgba(197, 160, 89, 0.6)';
             }}
             onMouseLeave={e => {
               e.currentTarget.style.transform = 'scale(1)';
               e.currentTarget.style.boxShadow = '0 4px 12px rgba(197, 160, 89, 0.4)';
             }}
             title="Go to Profile & History"
          >
             {userInitial}
          </div>
        ) : (
          <button className="nav-cta" onClick={() => navigate('/topic-selection')} aria-label="Start research">
            Start Research
          </button>
        )}
      </nav>

      {/* ══════════════════ PAGE ══════════════════ */}
      <main className="page-content">

        {/* ── HERO ── */}
        <section className="hero-section" aria-labelledby="hero-heading">
          <div className="hero-badge" aria-label="Platform badge">
            <span className="hero-badge-dot" aria-hidden />
            AI-Powered Research Platform
          </div>

          <h1 className="hero-title" id="hero-heading">
            <span className="line-1">Research Smarter,</span>
            <span className="line-2">Publish Faster</span>
          </h1>

          <p className="hero-subtitle">
            Nexus equips students and researchers with intelligent tools for discovering topics,
            finding literature, choosing journals, and verifying academic integrity — all in one place.
          </p>

          <div className="hero-actions">
            <button
              id="hero-get-started"
              className="btn-primary"
              onClick={() => navigate('/topic-selection')}
              aria-label="Explore features"
            >
              Explore Features <span className="btn-arrow" aria-hidden>→</span>
            </button>
            <button
              id="hero-learn-more"
              className="btn-outline"
              onClick={() => {
                document.getElementById('features')?.scrollIntoView({ behavior: 'smooth' });
              }}
              aria-label="Learn how Nexus works"
            >
              How it works
            </button>
          </div>

          {/* Stats */}
          <div className="stats-strip" role="list" aria-label="Platform statistics">
            {[
              { num: '50K+',   label: 'Journals indexed'   },
              { num: '10M+',   label: 'Papers indexed'      },
              { num: '4',      label: 'AI-powered tools'    },
              { num: '< 30s',  label: 'Results in seconds'  },
            ].map(s => (
              <div key={s.label} className="stat-item" role="listitem">
                <div className="stat-number">{s.num}</div>
                <div className="stat-label">{s.label}</div>
              </div>
            ))}
          </div>
        </section>

        {/* ── FEATURES ── */}
        <section id="features" className="features-section" aria-labelledby="features-heading">
          <div className="section-header" ref={addReveal}>
            <span className="section-eyebrow">Platform Tools</span>
            <h2 className="section-title" id="features-heading">Four Tools, One Platform</h2>
            <p className="section-desc">
              Every feature is purpose-built to remove friction at a different stage of the research workflow.
            </p>
          </div>

          <div className="features-grid">
            {FEATURES.map((f) => (
              <article
                key={f.id}
                className={`feature-card ${f.cardClass}`}
                onClick={() => navigate(f.path)}
                tabIndex={0}
                role="button"
                aria-label={`Open ${f.title}`}
                onKeyDown={e => (e.key === 'Enter' || e.key === ' ') && navigate(f.path)}
                ref={addReveal}
              >
                <div className="feature-top">
                  <div className={`feature-icon-wrap ${f.iconClass}`} aria-hidden>
                    {f.icon}
                  </div>
                  <span className={`feature-tag ${f.tagClass}`}>{f.tag}</span>
                </div>

                <h3 className="feature-title">{f.title}</h3>
                <p className="feature-desc">{f.desc}</p>

                <ul className="feature-bullets" aria-label={`${f.title} features`}>
                  {f.bullets.map(b => (
                    <li key={b}>
                      <span className={`bullet-dot ${f.dotClass}`} aria-hidden />
                      {b}
                    </li>
                  ))}
                </ul>

                <span className={`feature-cta ${f.ctaClass}`} aria-label={`Open ${f.title}`}>
                  Open tool <span className="arrow" aria-hidden>→</span>
                </span>
              </article>
            ))}
          </div>
        </section>

        {/* ── HOW IT WORKS ── */}
        <section className="how-section" aria-labelledby="how-heading">
          <div className="section-header" ref={addReveal}>
            <span className="section-eyebrow">Workflow</span>
            <h2 className="section-title" id="how-heading">How Nexus Works</h2>
            <p className="section-desc">
              A streamlined four-step process takes you from a vague idea to publication-ready insights.
            </p>
          </div>

          <div className="steps-grid">
            {STEPS.map((s, i) => (
              <div key={s.num} className="step-card" ref={addReveal} style={{ transitionDelay: `${i * 0.1}s` }}>
                <div className="step-num" aria-hidden>{s.num}</div>
                <h3 className="step-title">{s.title}</h3>
                <p className="step-desc">{s.desc}</p>
              </div>
            ))}
          </div>
        </section>

        {/* ── CTA BANNER ── */}
        <div className="cta-banner" ref={addReveal} aria-label="Call to action">
          <h2 className="cta-banner-title">Ready to accelerate your research?</h2>
          <p className="cta-banner-sub">
            Start with the Topic Selection Engine to find your research direction, or jump straight into the Literature Finder.
          </p>
          <button
            id="cta-banner-btn"
            className="btn-primary"
            onClick={() => navigate('/topic-selection')}
            aria-label="Get started with Nexus"
          >
            Get Started Free <span className="btn-arrow" aria-hidden>→</span>
          </button>
        </div>

        {/* ── FOOTER ── */}
        <footer className="nexus-footer" aria-label="Site footer">
          <div className="footer-left">
            <span className="footer-logo">Nexus</span>
            <span className="footer-copy">© 2025 · Research Helper Platform</span>
          </div>
          <ul className="footer-links" aria-label="Footer links">
            <li><a href="#features" aria-label="Go to features">Features</a></li>
            <li><a href="#" aria-label="Privacy policy">Privacy</a></li>
            <li><a href="#" aria-label="Terms of service">Terms</a></li>
          </ul>
        </footer>
      </main>
    </div>
  );
}
