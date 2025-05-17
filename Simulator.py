import numpy as np
import sympy as smp
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import PillowWriter
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
import warnings

# ----------------------------- USER INPUT SECTION -----------------------------
def get_positive_input(prompt, default=1.0, min_val=0.01, param_name="parameter"):
    try:
        val = float(input(prompt))
        if val <= 0:
            warnings.warn(f"{param_name} must be > 0. Auto-fixing to {min_val}.")
            return min_val
        return val
    except Exception:
        warnings.warn(f"Invalid input for {param_name}. Using default value {default}.")
        return default

# Function to receive and clamp initial angles and angular velocities
def get_initial_conditions():
    try:
        raw = input("Enter initial conditions [theta1, z1, theta2, z2] separated by commas: ")
        vals = list(map(float, raw.strip().split(',')))
        if len(vals) != 4:
            raise ValueError
        clamped = [
            np.clip(vals[0], -np.pi, np.pi),
            np.clip(vals[1], -10, 10),
            np.clip(vals[2], -np.pi, np.pi),
            np.clip(vals[3], -10, 10),
        ]
        if not np.allclose(clamped, vals):
            warnings.warn("Some initial conditions were clamped for stability.")
        return clamped
    except Exception:
        warnings.warn("Invalid input. Using default initial conditions [1, -3, -1, 5].")
        return [1, -3, -1, 5]

# Collect inputs
print("Double Pendulum Simulation Parameter Input")
m1_val = get_positive_input("Enter m1 (kg): ", default=2, param_name="m1")
m2_val = get_positive_input("Enter m2 (kg): ", default=1, param_name="m2")
L1_val = get_positive_input("Enter L1 (m): ", default=2, param_name="L1")
L2_val = get_positive_input("Enter L2 (m): ", default=1, param_name="L2")
mr1_val = get_positive_input("Enter mass of rod 1 (kg): ", default=0.5, param_name="mr1")  # NEW: Clarified as mass
mr2_val = get_positive_input("Enter mass of rod 2 (kg): ", default=0.3, param_name="mr2")  # NEW: Clarified as mass
bf1_val = get_positive_input("Enter hinge friction 1 (N·m·s/rad): ", default=0.1, param_name="bf1")  # NEW: Units added
bf2_val = get_positive_input("Enter hinge friction 2 (N·m·s/rad): ", default=0.1, param_name="bf2")  # NEW: Units added
Cd = get_positive_input("Enter drag coefficient Cd (~0.47 for sphere): ", default=0.47)  # NEW: Explicit Cd input
initial_conditions_val = get_initial_conditions()
# -----------------------------------------------------------------------------------

# Physics calculations
t, g = smp.symbols('t g')
m1, m2 = smp.symbols('m1 m2')
L1, L2 = smp.symbols('L1, L2')

the1, the2 = smp.symbols(r'\theta_1, \theta_2', cls=smp.Function)
the1 = the1(t)
the2 = the2(t)

the1_d = smp.diff(the1, t)
the2_d = smp.diff(the2, t)
the1_dd = smp.diff(the1_d, t)
the2_dd = smp.diff(the2_d, t)

x1 = L1*smp.sin(the1)
y1 = -L1*smp.cos(the1)
x2 = L1*smp.sin(the1)+L2*smp.sin(the2)
y2 = -L1*smp.cos(the1)-L2*smp.cos(the2)

T1 = 1/2 * m1 * (smp.diff(x1, t)**2 + smp.diff(y1, t)**2)
T2 = 1/2 * m2 * (smp.diff(x2, t)**2 + smp.diff(y2, t)**2)
T = T1+T2
V1 = m1*g*y1
V2 = m2*g*y2
V = V1 + V2
L = T-V

LE1 = smp.diff(L, the1) - smp.diff(smp.diff(L, the1_d), t).simplify()
LE2 = smp.diff(L, the2) - smp.diff(smp.diff(L, the2_d), t).simplify()

sols = smp.solve([LE1, LE2], (the1_dd, the2_dd), simplify=False, rational=False)

dz1dt_f = smp.lambdify((t,g,m1,m2,L1,L2,the1,the2,the1_d,the2_d), sols[the1_dd])
dz2dt_f = smp.lambdify((t,g,m1,m2,L1,L2,the1,the2,the1_d,the2_d), sols[the2_dd])
dthe1dt_f = smp.lambdify(the1_d, the1_d)
dthe2dt_f = smp.lambdify(the2_d, the2_d)

def dSdt(S, t, g, m1, m2, L1, L2, Cd, rho, mr1, mr2, bf1, bf2):
     the1, z1, the2, z2 = S

     # Effective masses including rod rotational inertia (rod pivoted at end: I = m_r * L^2 / 3)
     I1 = mr1 * L1**2 / 3
     I2 = mr2 * L2**2 / 3

     # Cross-sectional areas (assuming spherical masses)
     A1 = 0.01
     A2 = 0.01

     # Drag torques (quadratic)
     drag1 = -0.5 * Cd * rho * A1 * (L1**2) * z1 * abs(z1)
     drag2 = -0.5 * Cd * rho * A2 * (L2**2) * z2 * abs(z2)

     # Hinge friction torques (viscous)
     fric1 = -bf1 * z1
     fric2 = -bf2 * z2

     # Original angular accelerations
     dz1dt = dz1dt_f(t, g, m1, m2, L1, L2, the1, the2, z1, z2)
     dz2dt = dz2dt_f(t, g, m1, m2, L1, L2, the1, the2, z1, z2)

     # Add corrections: drag and friction over total inertia
     dz1dt += (drag1 + fric1) / (m1 * L1**2 + I1)
     dz2dt += (drag2 + fric2) / (m2 * L2**2 + I2)

     return [
         dthe1dt_f(z1),
         dz1dt,
         dthe2dt_f(z2),
         dz2dt,
     ]

# Simulation parameters - can be adjusted
t = np.linspace(0, 40, 1001)
g = 9.81
m1 = m1_val
m2 = m2_val
L1 = L1_val
L2 = L2_val
Cd = 1.05  # drag coefficient (approx. for a sphere)
rho = 1.225  # air density (kg/m^3)
mr1 = mr1_val
mr2 = mr2_val
bf1 = bf1_val
bf2 = bf2_val
initial_conditions = initial_conditions_val  # [theta1, z1, theta2, z2]

 # Solve ODE with friction and rod inertia
ans = odeint(dSdt, y0=initial_conditions, t=t,
               args=(g, m1, m2, L1, L2, Cd, rho, mr1, mr2, bf1, bf2))

# Run perturbed simulation for chaos analysis
ic_perturbed = [initial_conditions[0], initial_conditions[1], initial_conditions[2] + 0.001, initial_conditions[3]]
ans_perturbed = odeint(dSdt, y0=ic_perturbed, t=t, args=(g, m1_val, m2_val, L1_val, L2_val, Cd, rho, mr1_val, mr2_val, bf1_val, bf2_val))

 # Positions & animation code follows unchanged
the1_arr, the2_arr = ans.T[0], ans.T[2]
def get_x1y1x2y2(t, the1, the2, L1, L2):
     return (L1*np.sin(the1),
             -L1*np.cos(the1),
             L1*np.sin(the1) + L2*np.sin(the2),
             -L1*np.cos(the1) - L2*np.cos(the2))

x1, y1, x2, y2 = get_x1y1x2y2(t, the1_arr, the2_arr, L1, L2)

# Calculate velocities for color mapping
v1 = np.sqrt(np.diff(x1)**2 + np.diff(y1)**2)
v2 = np.sqrt(np.diff(x2)**2 + np.diff(y2)**2)
v1 = np.append(v1, v1[-1])  # Add one more element to match array sizes
v2 = np.append(v2, v2[-1])

# Create a beautiful animation with improved aesthetics
fig, ax = plt.subplots(figsize=(10, 10), facecolor='#0E1117')
ax.set_facecolor('#0E1117')  # Dark blue-black background
ax.grid(color='#2A3459', linestyle='-', linewidth=0.3, alpha=0.7)  # Subtle grid
ax.set_xlim(-4, 4)
ax.set_ylim(-4, 4)
ax.set_aspect('equal')

# Remove axis ticks but keep a subtle frame
ax.spines['bottom'].set_color('#4A5568')
ax.spines['top'].set_color('#4A5568') 
ax.spines['right'].set_color('#4A5568')
ax.spines['left'].set_color('#4A5568')
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])

# Create elements for the animation
# Pendulum rods
line, = ax.plot([], [], '-', lw=2.5, color='#A3BFFA')  # Light purple/blue line
# Pendulum masses
mass1, = ax.plot([], [], 'o', markersize=15, color='#F687B3')  # Pink
mass2, = ax.plot([], [], 'o', markersize=10, color='#68D391')  # Green
# Center pivot point
pivot, = ax.plot([0], [0], 'o', markersize=8, color='#F6E05E')  # Yellow

# Trace of the second pendulum's path
trace_length = 100  # How many points to keep in the trace
trace = ax.plot([], [], '-', lw=1.5, alpha=0.6)[0]
trace_x, trace_y = [], []

# Color map for the trace based on velocity
colormap = plt.cm.plasma
trace_colors = ax.scatter([], [], c=[], cmap=colormap, s=0)

# Text for time display and parameters
time_text = ax.text(-3.8, 3.7, '', fontsize=10, color='white')
param_text = ax.text(-3.8, -3.7, f'$m_1$={m1}, $m_2$={m2}, $L_1$={L1}, $L_2$={L2}', 
                    fontsize=10, color='white')

# Title with nice font
plt.title('Double Pendulum Simulation', color='white', fontsize=16, pad=20, 
         fontweight='bold', fontfamily='serif')

# Animation function
def animate(i):
    # Update pendulum position
    line.set_data([0, x1[i], x2[i]], [0, y1[i], y2[i]])
    mass1.set_data([x1[i]], [y1[i]])
    mass2.set_data([x2[i]], [y2[i]])
    
    # Update trace
    trace_x.append(x2[i])
    trace_y.append(y2[i])
    
    # Keep only a certain number of points in the trace
    if len(trace_x) > trace_length:
        trace_x.pop(0)
        trace_y.pop(0)
    
    trace.set_data(trace_x, trace_y)
    
    # Update trace color based on velocity
    if i > 0:
        # Create segments for colored line collection
        points = np.array([trace_x, trace_y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        # Get velocities for the segments in the trace
        start_idx = max(0, i - len(trace_x))
        end_idx = start_idx + len(segments)
        v_segment = v2[start_idx:end_idx]
        
        # Create a line collection with velocity-based colors
        lc = LineCollection(segments, cmap=colormap, norm=plt.Normalize(0, max(v2)*1.2))
        lc.set_array(v_segment)
        lc.set_linewidth(2)
        
        # Remove old line collection and add new one
        for coll in ax.collections:
            if coll != trace_colors:
                coll.remove()
        ax.add_collection(lc)
    
    # Update time display
    time_text.set_text(f'Time: {t[i]:.2f}s')
    
    return line, mass1, mass2, trace, time_text

# Create the animation
ani = animation.FuncAnimation(fig, animate, frames=len(t), 
                             interval=40, blit=True)

# --- Energy Tracking ---
# Extract solution arrays
theta1 = ans[:, 0]
omega1 = ans[:, 1]
theta2 = ans[:, 2]
omega2 = ans[:, 3]

# Positions
x1_e = L1 * np.sin(theta1)
y1_e = -L1 * np.cos(theta1)
x2_e = L1 * np.sin(theta1) + L2 * np.sin(theta2)
y2_e = -L1 * np.cos(theta1) - L2 * np.cos(theta2)

# Linear velocities
vx1 = np.gradient(x1_e, t)
vy1 = np.gradient(y1_e, t)
vx2 = np.gradient(x2_e, t)
vy2 = np.gradient(y2_e, t)

# Kinetic energies (point masses)
KE1 = 0.5 * m1 * (vx1**2 + vy1**2)
KE2 = 0.5 * m2 * (vx2**2 + vy2**2)

# Rotational KE of rods
I1 = mr1 * L1**2 / 3
I2 = mr2 * L2**2 / 3
RE1 = 0.5 * I1 * omega1**2
RE2 = 0.5 * I2 * omega2**2

# Potential energies
PE1 = m1 * g * y1_e
PE2 = m2 * g * y2_e

# Total energy
E_total = KE1 + KE2 + RE1 + RE2 + PE1 + PE2

# Angular Position and Velocity Time-Series Plot
plt.figure(figsize=(12, 8))

# Angular positions
plt.subplot(2, 1, 1)
plt.plot(t, theta1, label=r'$\theta_1$', color='cyan')
plt.plot(t, theta2, label=r'$\theta_2$', color='magenta')
plt.ylabel('Angle (rad)')
plt.title('Angular Position vs. Time')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()

# Angular velocities
plt.subplot(2, 1, 2)
plt.plot(t, omega1, label=r'$\omega_1$', color='blue')
plt.plot(t, omega2, label=r'$\omega_2$', color='red')
plt.xlabel('Time (s)')
plt.ylabel('Angular Velocity (rad/s)')
plt.title('Angular Velocity vs. Time')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()

# Plot for debugging
plt.figure()
plt.plot(t, E_total)
plt.xlabel('Time (s)')
plt.ylabel('Total Energy (J)')
plt.title('Total Mechanical Energy vs. Time')
plt.tight_layout()
plt.show()
# --- End Energy Tracking ---

# ------------------------------------------------------------------------------
# Chaos analysis plot
plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.plot(t, np.abs(ans[:, 2] - ans_perturbed[:, 2]), 'r')
plt.xlabel('Time (s)')
plt.ylabel('|Δθ₂| (rad)')
plt.title('Chaos: Angle Divergence from Δθ₂(0) = 0.001 rad')

plt.subplot(122)
plt.plot(t, E_total)
plt.xlabel('Time (s)')
plt.ylabel('Total Energy (J)')
plt.title('Energy Dissipation')
plt.tight_layout()
plt.show()

# Save as a high-quality GIF
ani.save('double_pendulum.gif', writer=PillowWriter(fps=25), 
        dpi=100)