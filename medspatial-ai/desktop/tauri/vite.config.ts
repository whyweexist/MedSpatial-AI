/**
 * MedSpatial AI — Vite Config (Tauri build)
 * ==========================================
 * Production build optimisations for the Tauri desktop app:
 *  - Target: esnext (WebView2 on Win10+ always supports it — no polyfills)
 *  - Manual chunk splitting: Three.js separated from React core to
 *    improve cache granularity and reduce initial parse time on low-end PCs
 *  - Source maps: off in production (reduces bundle size, no dev tools needed)
 *  - Minification: esbuild (fastest; terser adds ~2 s but saves <1% more)
 *  - Asset inlining: files < 10 KB inlined as base64 to reduce HTTP requests
 *    (all requests are loopback, but fewer round-trips still helps)
 */

import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { resolve } from "path";

const IS_TAURI = !!process.env.TAURI_ENV_TARGET_TRIPLE;

export default defineConfig({
  plugins: [
    react({
      // Use the automatic JSX runtime — no need for `import React` in every file
      jsxRuntime: "automatic",
    }),
  ],

  // Resolve the existing frontend source so we can import components
  // without copying them into the tauri/src directory
  resolve: {
    alias: {
      // Override services/api.js with the Tauri-aware version
      "@/services/api": resolve(__dirname, "src/tauri_api.js"),
      // Everything else resolves from the original frontend/src
      "@": resolve(__dirname, "../../../frontend/src"),
    },
  },

  // Vite dev server — proxies API calls to the backend during development
  server: {
    port: 5173,
    strictPort: true,
    proxy: {
      "/api": {
        target:    "http://127.0.0.1:8765",
        changeOrigin: true,
      },
      "/ws": {
        target:    "ws://127.0.0.1:8765",
        ws:        true,
        changeOrigin: true,
      },
      "/static": {
        target:    "http://127.0.0.1:8765",
        changeOrigin: true,
      },
    },
  },

  // Env vars available in the frontend bundle
  define: {
    __APP_VERSION__: JSON.stringify("1.0.0"),
  },

  build: {
    // Output to dist/ which is what tauri.conf.json5 "frontendDist" points to
    outDir: "dist",

    // WebView2 is Chromium-based and always modern — target esnext
    target:    "esnext",
    sourcemap: false,      // off in production
    minify:    "esbuild",  // fast and good enough

    rollupOptions: {
      output: {
        // Split vendor chunks to improve cache hit rate
        manualChunks: {
          // React core — rarely changes
          "vendor-react": ["react", "react-dom"],

          // Three.js is the largest dependency (~700 KB minified)
          // Split it so it can be cached independently of app code
          "vendor-three": ["three", "@react-three/fiber", "@react-three/drei"],

          // Networking
          "vendor-axios": ["axios"],
        },

        // Deterministic file naming for Tauri's asset protocol
        chunkFileNames:   "assets/[name]-[hash].js",
        entryFileNames:   "assets/[name]-[hash].js",
        assetFileNames:   "assets/[name]-[hash][extname]",
      },
    },

    // Inline assets < 10 KB as base64 (reduces loopback request count)
    assetsInlineLimit: 10_240,

    // Chunk size warning threshold — Three.js chunks will exceed 500 KB,
    // which is expected and acceptable for a desktop app on a local server
    chunkSizeWarningLimit: 1500,

    // Clear dist/ before each build
    emptyOutDir: true,
  },

  // Tauri-specific: suppress eval warnings from Three.js GLSL shader compiler
  ...(IS_TAURI && {
    esbuild: {
      // Keep console.log in production for Tauri's log capture
      drop: [],
    },
  }),
});
