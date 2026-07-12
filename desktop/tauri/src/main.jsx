/**
 * MedSpatial AI — Tauri Entry Point
 * ===================================
 * Wraps the existing App component with a LoadingScreen gate.
 * The app renders normally once the backend is healthy.
 *
 * The only change vs the original frontend/src/main.jsx is the
 * LoadingScreen wrapper — App.jsx is imported unchanged.
 */

import React, { useState } from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import LoadingScreen from "./LoadingScreen";

// Import original app styles (path adjusted for Tauri src/ layout)
import "./index.css";

function Root() {
  const [backendReady, setBackendReady] = useState(false);

  return (
    <>
      {!backendReady && (
        <LoadingScreen onReady={() => setBackendReady(true)} />
      )}
      {/*
        App is always mounted (not conditionally rendered) so that React
        doesn't lose state if the backend briefly blips. The loading screen
        sits on top (z-index 9999) and fades out when ready.
      */}
      <App />
    </>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <Root />
  </React.StrictMode>
);
