# Biometric Scannner UI

A futuristic, sci-fi themed biometric scanner interface built with React, TailwindCSS, and Framer Motion.

## 🚀 Getting Started

### Prerequisites

Ensure you have Node.js and npm installed.

### Installation

1. Navigate to the project directory:
   ```bash
   cd scanner-ui
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm run dev
   ```

## 📦 Dependencies

The project relies on the following key libraries:

### Core Framework
- **[React](https://react.dev/)** (`^19.0.0`): The library for web and native user interfaces.
- **[Vite](https://vitejs.dev/)** (`^7.0.0`): Next Generation Frontend Tooling.

### Styling & Design System
- **[TailwindCSS v4](https://tailwindcss.com/)** (`^4.0.0`): A utility-first CSS framework for rapid UI development. Used with the `@tailwindcss/vite` plugin.
- **Designed for Dark Mode**: The UI is optimized for dark, high-contrast displays with neon accents.

### Animations & Interactions
- **[Framer Motion](https://www.framer.com/motion/)** (`^11.0.0`): A production-ready motion library for React. Used for:
  - Page transitions
  - Scanner loop animations
  - Loading states (DNA/Ring loaders)
  - Interactive hover effects

### Icons & Assets
- **[Lucide React](https://lucide.dev/)** (`^0.300.0`): Beautiful & consistent icons (fingerprint, shield, cpu, etc.).

## 📂 Project Structure

```
scanner-ui/
├── src/
│   ├── components/
│   │   ├── Background.jsx       # Animated space gradient + particles
│   │   ├── GlassCard.jsx        # Reusable glassmorphism container
│   │   ├── ProcessingLoader.jsx # AI analysis simulation
│   │   └── Scanner.jsx          # Fingerprint scanning interaction
│   ├── App.jsx                  # Main application logic & state flow
│   ├── index.css                # Global styles & Tailwind configuration
│   └── main.jsx                 # Entry point
├── package.json
└── vite.config.js
```

## 🛠 Features

- **Glassmorphism Layout**: Translucent panels with blur effects.
- **Neon Aesthetics**: Glowing borders, text, and scan lines using CSS drop-shadows.
- **Interactive Scanner**: Click-to-upload functionality with visual feedback.
- **Simulated AI Processing**: Multi-step analysis loader with dynamic text.
