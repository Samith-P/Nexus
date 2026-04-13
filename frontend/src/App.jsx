import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Home from './pages/Home';
import JournalRecommendation   from './pages/JournalRecommendation';
import TopicSelection          from './pages/TopicSelection';
import LiteratureReview          from './pages/LiteratureReview';
import PlagiarismDetection     from './pages/PlagiarismDetection';

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/"                          element={<Home />} />
        <Route path="/journal-recommendation"    element={<JournalRecommendation />} />
        <Route path="/topic-selection"           element={<TopicSelection />} />
        <Route path="/literature-recommendation" element={<LiteratureReview />} />
        <Route path="/plagiarism-detection"      element={<PlagiarismDetection />} />
      </Routes>
    </BrowserRouter>
  );
}
