/** @type {import('tailwindcss').Config} */
export default {
  darkMode: "class",
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        canvas: "#f5f6f2",
        ink: "#182019",
        accent: {
          DEFAULT: "#c5f36b",
          deep: "#25351d",
        },
      },
      fontFamily: {
        sans: ["Inter", "ui-sans-serif", "system-ui", "sans-serif"],
      },
      boxShadow: {
        soft: "0 18px 50px rgba(24, 32, 25, 0.07)",
      },
    },
  },
  plugins: [],
};
