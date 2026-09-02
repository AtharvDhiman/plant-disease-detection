import { Route, BrowserRouter as Router, Routes } from 'react-router-dom';

import { ToastProvider } from './components/ui/toast';
import AppLayout from './layouts/AppLayout';
import Benchmark from './pages/Benchmark';
import Dashboard from './pages/Dashboard';
import History, { HistoryDetail } from './pages/History';
import Home from './pages/Home';
import Library from './pages/Library';
import NotFound from './pages/NotFound';
import Research from './pages/Research';

export default function App() {
  return (
    <ToastProvider>
      <Router>
        <Routes>
          <Route element={<AppLayout />}>
            <Route index element={<Home />} />
            <Route path="dashboard" element={<Dashboard />} />
            <Route path="benchmark" element={<Benchmark />} />
            <Route path="research" element={<Research />} />
            <Route path="library" element={<Library />} />
            <Route path="history" element={<History />} />
            <Route path="history/:id" element={<HistoryDetail />} />
            <Route path="*" element={<NotFound />} />
          </Route>
        </Routes>
      </Router>
    </ToastProvider>
  );
}
