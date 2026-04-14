import React from 'react';
import { Navigate, Outlet } from 'react-router-dom';

/**
 * Parses and verifies the JWT token.
 * Returns true if the token is present and its 'exp' claim is in the future.
 */
const isTokenValid = (token) => {
  if (!token) return false;
  
  try {
    // A JWT consists of 3 parts separated by dots: header.payload.signature
    const parts = token.split('.');
    if (parts.length !== 3) return false;

    // Decode the payload (part 2) from Base64Url
    const base64Url = parts[1];
    const base64 = base64Url.replace(/-/g, '+').replace(/_/g, '/');
    const jsonPayload = decodeURIComponent(atob(base64).split('').map(function(c) {
        return '%' + ('00' + c.charCodeAt(0).toString(16)).slice(-2);
    }).join(''));

    const decoded = JSON.parse(jsonPayload);
    const currentTime = Date.now() / 1000; // time in seconds
    
    // Check if token has expired
    if (decoded.exp && decoded.exp < currentTime) {
      return false; // Token expired
    }
    
    return true; // Token is structurally valid and not expired
  } catch (error) {
    console.error("Invalid token format", error);
    return false;
  }
};

const ProtectedRoute = ({ children }) => {
  const token = localStorage.getItem('nexus_access_token');
  const isValid = isTokenValid(token);

  if (!isValid) {
    // Clean up broken/expired token if present
    if (token) localStorage.removeItem('nexus_access_token');
    
    return <Navigate to="/login" replace />;
  }

  // Render children if passed directly, else render the nested routes (Outlet)
  return children ? children : <Outlet />;
};

export default ProtectedRoute;
