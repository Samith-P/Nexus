import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import './Profile.css';

export default function Profile() {
  const navigate = useNavigate();
  const [user, setUser] = useState(null);
  const [usages, setUsages] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    const fetchData = async () => {
      try {
        const token = localStorage.getItem('nexus_access_token');
        if (!token) {
          navigate('/login');
          return;
        }

        // Fetch User Info
        const userRes = await fetch('http://localhost:8000/auth/me', {
          headers: { 'Authorization': `Bearer ${token}` }
        });
        if (!userRes.ok) throw new Error('Failed to fetch user profile.');
        const userData = await userRes.json();
        setUser(userData);

        // Fetch Usages
        const usageRes = await fetch('http://localhost:8000/usage', {
          headers: { 'Authorization': `Bearer ${token}` }
        });
        // Non-fatal if we don't fetch usage history immediately
        if (usageRes.ok) {
          const usageData = await usageRes.json();
          setUsages(usageData);
        }
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, [navigate]);

  const handleLogout = () => {
    localStorage.removeItem('nexus_access_token');
    navigate('/');
  };

  if (loading) return <div className="profile-loading"><div className="ts-spinner"></div><p>Loading Profile Data...</p></div>;
  if (error) return <div className="profile-error"><h3>Oops!</h3><p>{error}</p><button className="profile-logout-btn" style={{marginTop: '20px'}} onClick={() => window.location.reload()}>Retry</button></div>;
  if (!user) return null;

  const getInitial = name => name ? name.charAt(0).toUpperCase() : 'U';

  const formatDate = (dateStr) => {
    const date = new Date(dateStr);
    return date.toLocaleString(undefined, {
      year: 'numeric', month: 'short', day: 'numeric',
      hour: '2-digit', minute: '2-digit'
    });
  };
  
  const formatApiName = (name) => {
     let formatted = name.replace(/_/g, ' ');
     return formatted.charAt(0).toUpperCase() + formatted.slice(1);
  };

  return (
    <div className="profile-container">
      {/* Top Banner Context Area */}
      <div className="profile-hero-banner">
        <button className="profile-back-btn" onClick={() => navigate('/')}>
          <span aria-hidden>←</span> Back to Home
        </button>
      </div>

      <div className="profile-content-wrapper">
        {/* Floating User Info Card */}
        <div className="profile-header-card">
           <div className="profile-avatar-wrap">
              <div className="profile-avatar" data-initial={getInitial(user.full_name)}></div>
           </div>

           <div className="profile-info-row">
              <div className="profile-details">
                 <h1 className="profile-name">{user.full_name}</h1>
                 <p className="profile-email">{user.email}</p>
              </div>
              <button className="profile-logout-btn" onClick={handleLogout}>Log Out</button>
           </div>
           
           <div className="profile-badges">
              <span className="profile-badge badge-role">{user.role || 'User'}</span>
              {user.institution && <span className="profile-badge badge-inst">🏫 {user.institution}</span>}
              {user.department && <span className="profile-badge badge-dept">📚 {user.department}</span>}
           </div>
        </div>

        {/* History Area */}
        <section className="profile-history">
           <div className="history-header">
             <h2 className="history-title">Activity & History</h2>
             <span className="history-count">{usages.length} {usages.length === 1 ? 'Entry' : 'Entries'}</span>
           </div>

           {usages.length === 0 ? (
             <div className="profile-empty-state">
               <div className="profile-empty-icon">🍃</div>
               <h3>Your history is empty</h3>
               <p>No research activities found. Start exploring tools to build your intelligent research trajectory!</p>
               <button className="profile-cta-btn" onClick={() => navigate('/topic-selection')}>
                 Go to Topic Engine <span aria-hidden>→</span>
               </button>
             </div>
           ) : (
             <div className="profile-grid">
               {usages.map((usage, i) => (
                 <div className="profile-card" key={usage._id || i}>
                    <div className="profile-card-header">
                       <span className="profile-api-name">{formatApiName(usage.api_name)}</span>
                       <span className="profile-date">{formatDate(usage.timestamp)}</span>
                    </div>
                    
                    <div className="profile-card-body">
                       <details className="profile-details-block">
                          <summary>View Data Details</summary>
                          <div className="profile-details-content">
                             <h5>Request Parameters</h5>
                             <pre>{JSON.stringify(usage.request_data, null, 2)}</pre>
                             
                             {usage.response_data && (
                               <>
                                 <h5>Response Results</h5>
                                 <pre>{JSON.stringify(usage.response_data, null, 2)}</pre>
                               </>
                             )}
                          </div>
                       </details>
                    </div>
                 </div>
               ))}
             </div>
           )}
        </section>
      </div>
    </div>
  );
}
