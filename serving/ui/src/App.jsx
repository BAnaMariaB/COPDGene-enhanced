import React, { Suspense, lazy } from 'react';
import { Navigate, Route, Routes } from 'react-router-dom';

const WelcomePage = lazy(() => import('./pages/WelcomePage'));
const DemoPage = lazy(() => import('./pages/DemoPage'));

function LoadingScreen() {
  return (
    <div className="loading-screen">
      <div className="loading-mark">COPDGene Enhanced</div>
      <div className="loading-copy">Loading interface...</div>
    </div>
  );
}

export default function App() {
  return (
    <Suspense fallback={<LoadingScreen />}>
      <Routes>
        <Route path="/" element={<Navigate to="/welcome" replace />} />
        <Route path="/welcome" element={<WelcomePage />} />
        <Route path="/demo" element={<DemoPage />} />
        <Route path="*" element={<Navigate to="/welcome" replace />} />
      </Routes>
    </Suspense>
  );
}
