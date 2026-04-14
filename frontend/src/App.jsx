import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Home from './pages/Home';
import JournalRecommendation   from './pages/JournalRecommendation';
import TopicSelection          from './pages/TopicSelection';
import LiteratureReview          from './pages/LiteratureReview';
import PlagiarismDetection     from './pages/PlagiarismDetection';
import Login                   from './pages/login';
import Profile                 from './pages/Profile';
import Multilingual            from './pages/multilingual';
import ProtectedRoute          from './components/ProtectedRoute';

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        {/* Public Routes */}
        <Route path="/"                          element={<Home />} />
        <Route path="/login"                     element={<Login />} />
        
        {/* Protected Routes Group */}
        <Route element={<ProtectedRoute />}>
          <Route path="/profile"                   element={<Profile />} />
          <Route path="/journal-recommendation"    element={<JournalRecommendation />} />
          <Route path="/topic-selection"           element={<TopicSelection />} />
          <Route path="/literature-recommendation" element={<LiteratureReview />} />
          <Route path="/plagiarism-detection"      element={<PlagiarismDetection />} />
          <Route path="/multilingual"              element={<Multilingual />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
