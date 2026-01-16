/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'risk-green': '#22c55e',
        'risk-yellow': '#eab308',
        'risk-red': '#dc2626',
      },
    },
  },
  plugins: [],
}
