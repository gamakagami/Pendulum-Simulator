# 🌀 Double Pendulum Simulator

![Double Pendulum Simulation](OIP.jpeg)

A Python-based physics simulation of a chaotic double pendulum system with realistic damping effects. This interactive application visualizes complex motion patterns and energy dissipation in a classical mechanics system.

## ✨ Features

- **Realistic Physics Engine**
  - Lagrangian mechanics formulation
  - Adjustable masses for bobs and rods
  - Configurable rod lengths and pivot friction
  - Quadratic air resistance modeling

- **Interactive Visualization**
  - Real-time animation with velocity-colored traces
  - Play/pause/reset controls
  - Parameter sliders for experimentation
  - Default presets for interesting behaviors

- **Advanced Analytics**
  - Angular position/velocity time-series
  - Mechanical energy tracking (KE + PE)
  - Chaos analysis via perturbation studies
  - Phase space visualization

## 🧮 Physics Concepts

- Double pendulum dynamics
- Chaotic systems behavior
- Energy dissipation mechanisms
- Numerical solutions to coupled ODEs
- Sensitivity to initial conditions

## 🛠️ Tech Stack

- **Core**: Python 3.x
- **Numerics**: NumPy, SciPy
- **Symbolics**: SymPy
- **Visualization**: Matplotlib
- **GUI**: Tkinter

## ⚙️ Installation & Usage

```bash
# Clone repository
git clone https://github.com/your-username/double-pendulum-simulator.git
cd double-pendulum-simulator

# Install dependencies
pip install numpy sympy scipy matplotlib

# Run simulation
python double_pendulum.py
```
## Interface Guide

Adjust parameters using sliders:
- Masses (bobs: 0.01-5kg, rods: 0.01-2kg)
- Lengths (0.1-3m)
- Friction coefficients (0-1 N·m·s/rad)
- Drag coefficient (0-1.5 Cd)
- Initial angles (-π to π)
- Angular velocities (0-10 rad/s)

Control buttons:
- Start/Stop animation
- Reset to default parameters
- View analytical graphs

## Sample Configurations

| Configuration | Parameters | Behavior |
|---------------|------------|----------|
| Chaotic (Default) | m1=2kg, m2=1kg, L1=1.5m, L2=1m, θ1=90°, θ2=-100° | Complex aperiodic motion |
| Periodic | Small angles (<10°), zero velocity | Regular oscillation |
| Energy Demo | High friction (0.5+) | Rapid energy dissipation |

## Project Structure

```
double-pendulum-simulator/
├── double_pendulum.py    # Main simulation code
├── requirements.txt      # Python dependencies
└── README.md             # This document
```

## License

MIT License - Free for educational and research use

## Contributors

- Your Name
- Collaborator Name

**Pro Tip:** Try the "Recommended Configurations" button for pre-tuned chaotic motion examples!

Just copy this into your README.md file and:
1. Replace image reference if needed
2. Update contributor info
3. Add actual repo link if available
4. Modify tech stack if you're using different libraries
