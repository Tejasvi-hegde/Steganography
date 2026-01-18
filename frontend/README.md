# Steganography Frontend

A modern, responsive web interface for the Deep Learning Image Steganography system.

## Design

Built with the **"Cipher Room" (Modern Security Aesthetic)** design:
- Dark navy theme (`#0F1419` background)
- Electric teal accents (`#00D4AA`)
- Space Grotesk + IBM Plex Sans typography
- Card-based modular layout
- Smooth animations with Framer Motion

## Features

- **Hide Secret**: Upload cover + secret images, get stego output with metrics
- **Extract Secret**: Upload stego image, recover hidden secret
- **Quality Metrics**: PSNR, SSIM, MSE calculations displayed
- **Download Results**: One-click download of stego and recovered images
- **Responsive Design**: Works on desktop and mobile

## Tech Stack

- React 18 + TypeScript
- Vite (dev server & build)
- Tailwind CSS (styling)
- Framer Motion (animations)
- React Dropzone (file uploads)
- Lucide React (icons)
- Axios (API calls)

## Quick Start

### 1. Install Dependencies

```bash
cd frontend
npm install
```

### 2. Start Backend (in separate terminal)

```bash
cd backend
pip install flask flask-cors
python app.py
```

Backend runs at: http://localhost:5000

### 3. Start Frontend

```bash
npm run dev
```

Frontend runs at: http://localhost:3000

## Project Structure

```
frontend/
├── public/
│   └── favicon.svg
├── src/
│   ├── components/       # Reusable UI components
│   │   ├── Button.tsx
│   │   ├── ImageUploader.tsx
│   │   ├── LoadingBar.tsx
│   │   ├── MetricCard.tsx
│   │   └── ResultImage.tsx
│   ├── sections/         # Page sections
│   │   ├── Navbar.tsx
│   │   ├── HeroSection.tsx
│   │   ├── WorkflowDiagram.tsx
│   │   ├── ApplicationPanel.tsx
│   │   ├── ResultsSection.tsx
│   │   ├── TechnicalDetails.tsx
│   │   └── Footer.tsx
│   ├── services/         # API layer
│   │   └── api.ts
│   ├── types/            # TypeScript types
│   │   └── index.ts
│   ├── App.tsx           # Main app component
│   ├── main.tsx          # Entry point
│   └── index.css         # Tailwind styles
├── tailwind.config.js
├── vite.config.ts
└── package.json
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check |
| `/api/hide` | POST | Hide secret in cover (multipart/form-data) |
| `/api/extract` | POST | Extract secret from stego (multipart/form-data) |

## Build for Production

```bash
npm run build
```

Output in `dist/` folder, ready for static hosting.

## Color Palette

| Element | Color | Hex |
|---------|-------|-----|
| Background | Deep Navy | `#0F1419` |
| Surface | Slate Blue | `#1C2938` |
| Accent | Electric Teal | `#00D4AA` |
| Secondary | Soft Amber | `#FFB84D` |
| Text | Crisp White | `#F5F5F5` |
| Muted | Blue-Gray | `#8899A6` |
| Error | Coral Red | `#FF6B6B` |
