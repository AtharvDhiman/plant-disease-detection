import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// The dev server proxies /api to the FastAPI backend so the browser sees a
// single origin and no CORS preflight is needed during development.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: process.env.VITE_API_TARGET || 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
      // Swagger UI and ReDoc are served by the backend; proxying them keeps the
      // footer links working in development.
      '/docs': { target: process.env.VITE_API_TARGET || 'http://127.0.0.1:8000', changeOrigin: true },
      '/redoc': { target: process.env.VITE_API_TARGET || 'http://127.0.0.1:8000', changeOrigin: true },
      '/openapi.json': { target: process.env.VITE_API_TARGET || 'http://127.0.0.1:8000', changeOrigin: true },
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: false,
    rollupOptions: {
      output: {
        // Vite 8 (rolldown) requires the function form of manualChunks.
        // Recharts is only needed on the dashboard/benchmark/research routes, so
        // splitting it keeps the initial upload page small.
        manualChunks(id) {
          if (!id.includes('node_modules')) return undefined
          if (id.includes('recharts') || id.includes('d3-')) return 'charts'
          if (id.includes('react-router')) return 'router'
          if (id.includes('/react/') || id.includes('/react-dom/')) return 'react'
          return undefined
        },
      },
    },
  },
})
