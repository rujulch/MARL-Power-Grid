# Smart Grid MARL Frontend

Modern web dashboard for visualizing multi-agent reinforcement learning in smart grid optimization.

## Features

- 🎨 Modern, responsive UI with dark theme
- 📊 Real-time grid visualization using D3.js
- ⚡ Live energy flow animations
- 📈 Performance metrics dashboard
- 🔄 WebSocket connection for real-time updates
- 🎯 Individual agent status monitoring

## Tech Stack

- **Framework**: Next.js 14 with App Router
- **Language**: TypeScript
- **Styling**: TailwindCSS
- **Visualization**: D3.js, Recharts
- **Animation**: Framer Motion
- **State Management**: React Hooks
- **API**: Axios, WebSocket

## Getting Started

### Prerequisites

- Node.js 18+ 
- npm or yarn

### Installation

```bash
npm install
```

### Development

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

### Build for Production

```bash
npm run build
npm start
```

## Project Structure

```
src/
├── app/              # Next.js app router pages
├── components/       # React components
├── lib/             # Utilities and helpers
└── types/           # TypeScript type definitions
```

## Configuration

The frontend connects to the backend API at `http://localhost:8000` by default. Update `src/lib/api.ts` and `src/lib/websocket.ts` if your backend runs on a different address.

## Components

- **GridVisualization**: D3.js network visualization of energy grid
- **AgentCard**: Individual agent status display
- **MetricsDashboard**: Real-time performance metrics
- **ControlPanel**: Simulation controls







