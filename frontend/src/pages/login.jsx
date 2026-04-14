import React, { useState } from 'react';
import './login.css';
import { useNavigate } from 'react-router-dom';

export default function Login() {
  const navigate = useNavigate();
  const [isLogin, setIsLogin] = useState(true);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');

  const [formData, setFormData] = useState({
    full_name: '',
    email: '',
    password: '',
    role: 'student',
    institution: '',
    department: '',
    designation: 'Student',
    education_level: 'UG',
    preferred_language: 'English'
  });

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
  };

  const handleToggle = () => {
    setIsLogin(!isLogin);
    setError('');
    setSuccess('');
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    setSuccess('');

    try {
      const endpoint = isLogin ? '/auth/login' : '/auth/signup';
      const url = `http://localhost:8000${endpoint}`;
      
      const payload = isLogin 
        ? { email: formData.email, password: formData.password }
        : formData;

      const response = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || 'Authentication failed');
      }

      // Success
      localStorage.setItem('nexus_access_token', data.access_token);
      
      if (isLogin) {
        // Redirect to home or protected page
        navigate('/');
      } else {
        setSuccess('Account created successfully! Logging you in...');
        setTimeout(() => navigate('/'), 1500);
      }

    } catch (err) {
      setError(err.message || 'Something went wrong. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="auth-container">
      <div className={`auth-card ${!isLogin ? 'signup-mode' : ''}`}>
        <div className="auth-header">
          <div className="auth-icon">⚲</div>
          <h1 className="auth-title">
            {isLogin ? 'Welcome Back' : 'Create Account'}
          </h1>
          <p className="auth-subtitle">
            {isLogin 
              ? 'Enter your details to access your Nexus dashboard.' 
              : 'Join Nexus to elevate your academic research journey.'}
          </p>
        </div>

        {error && <div className="auth-error">{error}</div>}
        {success && <div className="auth-success">{success}</div>}

        <form className="auth-form" onSubmit={handleSubmit}>
          {isLogin ? (
            // --- LOGIN FORM ---
            <>
              <div className="input-group full-width">
                <label>Email Address</label>
                <input 
                  type="email" 
                  name="email" 
                  className="auth-input" 
                  placeholder="name@institution.edu" 
                  required 
                  value={formData.email}
                  onChange={handleChange}
                />
              </div>
              <div className="input-group full-width">
                <label>Password</label>
                <input 
                  type="password" 
                  name="password" 
                  className="auth-input" 
                  placeholder="••••••••" 
                  required 
                  value={formData.password}
                  onChange={handleChange}
                />
              </div>
            </>
          ) : (
            // --- SIGNUP FORM ---
            <div className="form-grid">
              <div className="input-group">
                <label>Full Name</label>
                <input type="text" name="full_name" className="auth-input" required value={formData.full_name} onChange={handleChange} placeholder="John Doe" />
              </div>
              
              <div className="input-group">
                <label>Email Address</label>
                <input type="email" name="email" className="auth-input" required value={formData.email} onChange={handleChange} placeholder="name@domain.edu" />
              </div>

              <div className="input-group">
                <label>Password</label>
                <input type="password" name="password" className="auth-input" required value={formData.password} onChange={handleChange} placeholder="••••••••" />
              </div>

              <div className="input-group">
                <label>Primary Role</label>
                <select name="role" className="auth-select" required value={formData.role} onChange={handleChange}>
                  <option value="student">Student</option>
                  <option value="faculty">Faculty</option>
                  <option value="researcher">Researcher</option>
                </select>
              </div>

              <div className="input-group">
                <label>Institution Name</label>
                <input type="text" name="institution" className="auth-input" required value={formData.institution} onChange={handleChange} placeholder="e.g. GDC West Godavari" />
              </div>

              <div className="input-group">
                <label>Department</label>
                <input type="text" name="department" className="auth-input" required value={formData.department} onChange={handleChange} placeholder="e.g. Computer Science" />
              </div>

              <div className="input-group">
                <label>Designation</label>
                <select name="designation" className="auth-select" required value={formData.designation} onChange={handleChange}>
                  <option value="Student">Student</option>
                  <option value="Assistant Professor">Assistant Professor</option>
                  <option value="Associate Professor">Associate Professor</option>
                  <option value="Professor">Professor</option>
                  <option value="Research Scholar">Research Scholar</option>
                </select>
              </div>

              <div className="input-group">
                <label>Education Level</label>
                <select name="education_level" className="auth-select" required value={formData.education_level} onChange={handleChange}>
                  <option value="UG">Undergraduate (UG)</option>
                  <option value="PG">Postgraduate (PG)</option>
                  <option value="PhD">Doctorate (PhD)</option>
                  <option value="PostDoc">Post-Doctoral</option>
                </select>
              </div>

              <div className="input-group full-width">
                <label>Preferred Language</label>
                <select name="preferred_language" className="auth-select" required value={formData.preferred_language} onChange={handleChange}>
                  <option value="English">English</option>
                  <option value="Telugu">Telugu</option>
                  <option value="Hindi">Hindi</option>
                  <option value="Urdu">Urdu</option>
                  <option value="Sanskrit">Sanskrit</option>
                </select>
              </div>
            </div>
          )}

          <button type="submit" className="auth-button" disabled={loading}>
            {loading ? 'Processing...' : (isLogin ? 'Sign In to Nexus' : 'Create Account')}
          </button>
        </form>

        <div className="auth-toggle">
          {isLogin ? "Don't have an account?" : "Already have an account?"}
          <span onClick={handleToggle}>
            {isLogin ? 'Sign up here' : 'Log in instead'}
          </span>
        </div>
      </div>
    </div>
  );
}
