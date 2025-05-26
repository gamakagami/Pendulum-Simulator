import numpy as np
import sympy as smp
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import PillowWriter
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
import warnings
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk

# Main application window
root = tk.Tk()
root.title("Double Pendulum Simulation")
root.geometry("1920x1080")

# Initialize the figure variable that will hold our animation
fig = None

# Create a welcome canvas with title and image
canvas1 = tk.Canvas(root, width=500, height=500, bg="black")
canvas1.place(x=700, y=50)
canvas1.create_text(250, 100, text="Double Pendulum", fill="white", font=("Comic Sans MS", 30))
canvas1.create_text(250, 170, text="Simulation", fill="white", font=("Comic Sans MS", 25))

# Load and display welcome image
try:
    img = Image.open("./OIP.jpeg")
    img = img.resize((240, 200))
    tk_img = ImageTk.PhotoImage(img)
    canvas1.create_image(250, 340, image=tk_img)
except FileNotFoundError:
    # Create a text message if image isn't found
    canvas1.create_text(250, 340, text="Image not found", fill="white", font=("Comic Sans MS", 12))

if fig:
    canvas1.destroy()

def main(m1_val, m2_val, L1_val, L2_val, drag_coeff, mass_rod1, mass_rod2, hinge_friction1, hinge_friction2, theta_val1, ang_velocity1, theta_val2, ang_velocity2, root):
    global ani

    # Define symbolic variables for deriving equations of motion
    t, g = smp.symbols('t g')  # time and gravitational acceleration
    m1, m2 = smp.symbols('m1 m2')  # point masses
    L1, L2 = smp.symbols('L1, L2')  # rod lengths
    mr1, mr2 = smp.symbols('mr1 mr2')  # rod masses

    # Define the angle functions of time
    the1, the2 = smp.symbols(r'\theta_1, \theta_2', cls=smp.Function)
    the1 = the1(t)
    the2 = the2(t)

    # Calculate derivatives (angular velocities and accelerations)
    the1_d = smp.diff(the1, t) # First derivative: angular velocity of rod 1
    the2_d = smp.diff(the2, t) # First derivative: angular velocity of rod 2
    the1_dd = smp.diff(the1_d, t) # Second derivative: angular acceleration of rod 1
    the2_dd = smp.diff(the2_d, t) # Second derivative: angular acceleration of rod 2

    # Calculate positions of the point masses
    x1 = L1*smp.sin(the1)
    y1 = -L1*smp.cos(the1)
    x2 = L1*smp.sin(the1)+L2*smp.sin(the2)
    y2 = -L1*smp.cos(the1)-L2*smp.cos(the2)

    # Calculate kinetic energies including rod inertia from the start
    # Point mass kinetic energies
    T1_point = smp.Rational(1,2) * m1 * (smp.diff(x1, t)**2 + smp.diff(y1, t)**2)
    T2_point = smp.Rational(1,2) * m2 * (smp.diff(x2, t)**2 + smp.diff(y2, t)**2)
    
    # Rod rotational kinetic energies (moment of inertia for rod about end: I = mL²/3)
    I1 = mr1 * L1**2 / 3  # Moment of inertia of first rod
    I2 = mr2 * L2**2 / 3  # Moment of inertia of second rod
    T1_rod = smp.Rational(1,2) * I1 * the1_d**2
    T2_rod = smp.Rational(1,2) * I2 * the2_d**2
    
    # Total kinetic energy (properly coupled from the beginning)
    T = T1_point + T2_point + T1_rod + T2_rod

    # Calculate potential energies for each mass (relative to y=0)
    V1 = m1*g*y1
    V2 = m2*g*y2
    V = V1 + V2 # Total potential energy

    # Lagrangian = T - V (Kinetic - Potential Energy)
    L = T - V

    # Euler-Lagrange equations for each angle
    # d/dt(∂L/∂θ̇) - ∂L/∂θ = 0
    LE1 = smp.diff(L, the1) - smp.diff(smp.diff(L, the1_d), t).simplify()
    LE2 = smp.diff(L, the2) - smp.diff(smp.diff(L, the2_d), t).simplify()

    # Solve the Euler-Lagrange equations for the angular accelerations
    print("Solving Lagrangian equations... (this may take a moment)")
    sols = smp.solve([LE1, LE2], (the1_dd, the2_dd), simplify=False, rational=False)

    # Convert symbolic solutions to numerical functions
    dz1dt_f = smp.lambdify((t,g,m1,m2,L1,L2,mr1,mr2,the1,the2,the1_d,the2_d), sols[the1_dd])
    dz2dt_f = smp.lambdify((t,g,m1,m2,L1,L2,mr1,mr2,the1,the2,the1_d,the2_d), sols[the2_dd])
    dthe1dt_f = smp.lambdify(the1_d, the1_d)
    dthe2dt_f = smp.lambdify(the2_d, the2_d)

    """
        system of differential equations for the double pendulum.
        Rod inertia is now properly included in the Lagrangian derivation.
        
        Parameters:
        - S: State vector [theta1, omega1, theta2, omega2]
        - t: Time
        - g: Gravity
        - m1, m2: Point masses
        - L1, L2: Rod lengths
        - mr1, mr2: Rod masses
        - Cd: Drag coefficient
        - rho: Air density
        - bf1, bf2: Hinge friction coefficients
        
        Returns: 
        - Derivatives of the state vector
    """
    def dSdt(S, t, g, m1, m2, L1, L2, mr1, mr2, Cd, rho, bf1, bf2):
        the1, z1, the2, z2 = S

        # Get angular accelerations from the Lagrangian solution
        # (Rod inertia is now properly included in the coupling)
        dz1dt = dz1dt_f(t, g, m1, m2, L1, L2, mr1, mr2, the1, the2, z1, z2)
        dz2dt = dz2dt_f(t, g, m1, m2, L1, L2, mr1, mr2, the1, the2, z1, z2)

        # Add dissipative forces (drag and friction)
        # These are still added as corrections since they're non-conservative
        
        # Cross-sectional areas for drag calculation
        A1 = 0.01
        A2 = 0.01

        # Aerodynamic drag torques (proportional to velocity squared)
        drag1 = -0.5 * Cd * rho * A1 * (L1**2) * z1 * abs(z1)
        drag2 = -0.5 * Cd * rho * A2 * (L2**2) * z2 * abs(z2)

        # Rotational friction at hinges (proportional to velocity)
        fric1 = -bf1 * z1
        fric2 = -bf2 * z2

        # Effective moments of inertia (including both point mass and rod contributions)
        I_eff1 = m1 * L1**2 + mr1 * L1**2 / 3
        I_eff2 = m2 * L2**2 + mr2 * L2**2 / 3

        # Add dissipative corrections to the accelerations
        dz1dt += (drag1 + fric1) / I_eff1
        dz2dt += (drag2 + fric2) / I_eff2

        # Return derivatives [dθ₁/dt, dω₁/dt, dθ₂/dt, dω₂/dt]
        return [
            dthe1dt_f(z1), # Angular velocity of first pendulum
            dz1dt, # Angular acceleration of first pendulum
            dthe2dt_f(z2), # Angular velocity of second pendulum
            dz2dt, # Angular acceleration of second pendulum
        ]

    # Simulation parameters
    t = np.linspace(0, 40, 1001) # Time array: 0 to 40 seconds with 1001 points
    g = 9.81 # Gravitational acceleration (m/s²)
    m1 = m1_val # Mass of first pendulum (kg)
    m2 = m2_val # Mass of second pendulum (kg)
    L1 = L1_val # Length of first rod (m)
    L2 = L2_val  # Length of second rod (m)
    mr1 = mass_rod1  # Mass of first rod (kg)
    mr2 = mass_rod2 # Mass of second rod (kg)
    Cd = drag_coeff  # drag coefficient (approx. for a sphere)
    rho = 1.225  # air density (kg/m^3)
    bf1 = hinge_friction1 # Friction coefficient at pivot 1 (N·m·s/rad)
    bf2 = hinge_friction2 # Friction coefficient at pivot 2 (N·m·s/rad)
    initial_conditions = [theta_val1, ang_velocity1, theta_val2, ang_velocity2]  # [theta1, z1, theta2, z2]

    # Solve the system of ODEs with the parameters
    print("Solving differential equations...")
    ans = odeint(dSdt, y0=initial_conditions, t=t,
                args=(g, m1, m2, L1, L2, mr1, mr2, Cd, rho, bf1, bf2))

    # Create a slightly perturbed initial condition for chaos analysis
    ic_perturbed = [initial_conditions[0], initial_conditions[1], initial_conditions[2] + 0.001, initial_conditions[3]]
    ans_perturbed = odeint(dSdt, y0=ic_perturbed, t=t, 
                          args=(g, m1, m2, L1, L2, mr1, mr2, Cd, rho, bf1, bf2))

    # Extract angles from solution
    the1_arr, the2_arr = ans.T[0], ans.T[2]

    # Function to calculate positions from angles
    def get_x1y1x2y2(t, the1, the2, L1, L2):
        return (L1*np.sin(the1), # x1: x-position of first mass
                -L1*np.cos(the1), # y1: y-position of first mass
                L1*np.sin(the1) + L2*np.sin(the2), # x2: x-position of second mass
                -L1*np.cos(the1) - L2*np.cos(the2)) # y2: y-position of second mass

    # Calculate positions of both masses for all time steps
    x1, y1, x2, y2 = get_x1y1x2y2(t, the1_arr, the2_arr, L1, L2)

    # Calculate velocities for color mapping
    v1 = np.sqrt(np.diff(x1)**2 + np.diff(y1)**2) # Speed of first mass
    v2 = np.sqrt(np.diff(x2)**2 + np.diff(y2)**2) # Speed of second mass

    # Append last elements to match array sizes
    v1 = np.append(v1, v1[-1])
    v2 = np.append(v2, v2[-1])

    global fig

    # Create animation with improved aesthetics
    fig, ax = plt.subplots(figsize=(5, 5), facecolor='#0E1117')
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

    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().place(x=700, y=50)

    # Animation function
    def animate(i):
        # Update pendulum position
        line.set_data([0, x1[i], x2[i]], [0, y1[i], y2[i]])
        mass1.set_data([x1[i]], [y1[i]])
        mass2.set_data([x2[i]], [y2[i]])
        
        # Update trace of second pendulum
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
    print("Creating animation...")
    ani = animation.FuncAnimation(fig, animate, frames=len(t), 
                                interval=40, blit=True)
    ani.event_source.stop()

    # --- Energy Tracking ---
    # Extract solution arrays
    theta1 = ans[:, 0] # Angle of first pendulum
    omega1 = ans[:, 1] # Angular velocity of first pendulum
    theta2 = ans[:, 2] # Angle of second pendulum
    omega2 = ans[:, 3] # Angular velocity of second pendulum

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

    # Rotational KE of rods (properly calculated)
    I1 = mr1 * L1**2 / 3 # Moment of inertia of first rod
    I2 = mr2 * L2**2 / 3 # Moment of inertia of second rod
    RE1 = 0.5 * I1 * omega1**2 # Rotational energy of first rod
    RE2 = 0.5 * I2 * omega2**2 # Rotational energy of second rod

    # Potential energies
    PE1 = m1 * g * y1_e # Potential energy of first mass
    PE2 = m2 * g * y2_e # Potential energy of second mass

    # Calculate total mechanical energy
    E_total = KE1 + KE2 + RE1 + RE2 + PE1 + PE2

    # --- Functions for additional analysis graphs ---
    global show_angular_positions

    def show_angular_positions():
        angular_position_popup = tk.Toplevel()
        angular_position_popup.geometry(("700x500"))
        angular_position_popup.title("Angular Positions Graph")

        fig1, ax1 = plt.subplots()
        ax1.plot(t, theta1, label=r'$\theta_1$', color='cyan')
        ax1.plot(t, theta2, label=r'$\theta_2$', color='magenta')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Angle (rad)')
        ax1.set_title('Angular Position vs. Time')
        ax1.grid(True, linestyle='--', alpha=0.5)
        ax1.legend()
        
        canvas_angular_positions = FigureCanvasTkAgg(fig1, master=angular_position_popup)
        canvas_angular_positions.draw()
        canvas_angular_positions.get_tk_widget().place(x=0, y=0)

    global show_angular_velocities

    def show_angular_velocities():
        angular_velocity_popup = tk.Toplevel()
        angular_velocity_popup.geometry(("700x500"))
        angular_velocity_popup.title("Angular Velocity Graph")

        fig2, ax2 = plt.subplots()
        ax2.plot(t, omega1, label=r'$\omega_1$', color='blue')
        ax2.plot(t, omega2, label=r'$\omega_2$', color='red')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Angular Velocity (rad/s)')
        ax2.set_title('Angular Velocity vs. Time')
        ax2.grid(True, linestyle='--', alpha=0.5)
        ax2.legend()

        canvas_angular_velocities = FigureCanvasTkAgg(fig2, master=angular_velocity_popup)
        canvas_angular_velocities.draw()
        canvas_angular_velocities.get_tk_widget().place(x=0, y=0)

    global show_mechanical_energy

    def show_mechanical_energy():
        energy_loss_popup = tk.Toplevel()
        energy_loss_popup.geometry(("700x500"))
        energy_loss_popup.title("Mechanical Energy Graph")

        fig3, ax3 = plt.subplots()
        ax3.plot(t, E_total)
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Mechanical Energy (J)')
        ax3.set_title('Energy Dissipation Over Time')
        ax3.grid(True, linestyle='--', alpha=0.5)

        canvas_total_energy = FigureCanvasTkAgg(fig3, master=energy_loss_popup)
        canvas_total_energy.draw()
        canvas_total_energy.get_tk_widget().place(x=0, y=0)

    global show_chaos_analysis

    def show_chaos_analysis():
        chaos_analysis_popup = tk.Toplevel()
        chaos_analysis_popup.geometry(("700x500"))
        chaos_analysis_popup.title("Chaos Analysis Graph")

        fig4, ax4 = plt.subplots()
        ax4.plot(t, np.abs(ans[:, 2] - ans_perturbed[:, 2]), 'r')
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('|Δθ₂| (rad)')
        ax4.set_title('Chaos: Angle Divergence from Δθ₂(0) = 0.001 rad')
        ax4.grid(True, linestyle='--', alpha=0.5)

        canvas_chaos_analysis = FigureCanvasTkAgg(fig4, master=chaos_analysis_popup)
        canvas_chaos_analysis.draw()
        canvas_chaos_analysis.get_tk_widget().place(x=0, y=0)

def update_mass1(value):
    slider_value1.config(text=f"{value} kg")

def update_mass2(value):
    slider_value2.config(text=f"{value} kg")

def update_length1(value):
    slider_value3.config(text=f"{value} m")

def update_mass3(value):
    slider_value4.config(text=f"{value} kg")

def update_mass4(value):
    slider_value6.config(text=f"{value} kg")

def update_length2(value):
    slider_value5.config(text=f"{value} m")

def update_friction1(value):
    slider_value7.config(text=f"{value} N·m·s/rad")

def update_friction2(value):
    slider_value8.config(text=f"{value} N·m·s/rad")

def update_resistance(value):
    slider_value9.config(text=f"{value} Cd")

def update_angle1(value):
    slider_value10.config(text=f"{value} rad")

def update_angle2(value):
    slider_value11.config(text=f"{value} rad")

def update_velocity1(value):
    slider_value12.config(text=f"{value} rad/s")

def update_velocity2(value):
    slider_value13.config(text=f"{value} rad/s")

def set_default_settings():
    slider1.set(1.0)
    slider2.set(1.0)
    slider3.set(1.0)
    slider4.set(0.1)
    slider5.set(1.0)
    slider6.set(0.1)
    slider7.set(0.05)
    slider8.set(0.05)
    slider9.set(0.1)
    slider10.set(np.pi/4)
    slider11.set(-np.pi/4)
    slider12.set(0.0)
    slider13.set(0.0)

def reset_settings():
    slider1.set(0)
    slider2.set(0)
    slider3.set(0)
    slider4.set(0)
    slider5.set(0)
    slider6.set(0)
    slider7.set(0)
    slider8.set(0)
    slider9.set(0)
    slider10.set(0)
    slider11.set(0)
    slider12.set(0)
    slider13.set(0)    

def stop_animation():
    ani.event_source.stop()

label1 = tk.Label(root, text="Mass of first pendulum:", font=("Comic Sans MS", 10))
label1.place(x=100, y=50)

slider1 = tk.Scale(root, from_=0.01, to=5.0, resolution=0.01, orient="horizontal", width=10, length=90, showvalue=False, command=update_mass1)
slider1.place(x=320, y=53)

slider_value1 = tk.Label(root, text="0.01 kg", font=("Comic Sans MS", 10))
slider_value1.place(x=430, y=50)

label2 = tk.Label(root, text="Mass of second pendulum:", font=("Comic Sans MS", 10))
label2.place(x=100, y=81)

slider2 = tk.Scale(root, from_=0.01, to=5.0, resolution=0.01, orient="horizontal", width=10, length=90, showvalue=False, command=update_mass2)
slider2.place(x=320, y=84)

slider_value2 = tk.Label(root, text="0.01 kg", font=("Comic Sans MS", 10))
slider_value2.place(x=430, y=81)

label3 = tk.Label(root, text="Length of first rod:", font=("Comic Sans MS", 10))
label3.place(x=100, y=112)

slider3 = tk.Scale(root, from_=0.1, to=3.0, resolution=0.1, orient="horizontal", width=10, length=90, showvalue=False, command=update_length1)
slider3.place(x=320, y=115)

slider_value3 = tk.Label(root, text="0.1 m", font=("Comic Sans MS", 10))
slider_value3.place(x=430, y=112)

label4 = tk.Label(root, text="Mass of first rod:", font=("Comic Sans MS", 10))
label4.place(x=100, y=143)

slider4 = tk.Scale(root, from_=0.01, to=2.0, resolution=0.01, orient="horizontal", width=10, length=90, showvalue=False, command=update_mass3)
slider4.place(x=320, y=146)
 
slider_value4 = tk.Label(root, text="0.01 kg", font=("Comic Sans MS", 10))
slider_value4.place(x=430, y=143)

label5 = tk.Label(root, text="Length of second rod:", font=("Comic Sans MS", 10))
label5.place(x=100, y=174)

slider5 = tk.Scale(root, from_=0.1, to=3.0, resolution=0.1, orient="horizontal", width=10, length=90, showvalue=False, command=update_length2)
slider5.place(x=320, y=177)

slider_value5 = tk.Label(root, text="0.1 m", font=("Comic Sans MS", 10))
slider_value5.place(x=430, y=174)

label6 = tk.Label(root, text="Mass of second rod:", font=("Comic Sans MS", 10))
label6.place(x=100, y=205)

slider6 = tk.Scale(root, from_=0.01, to=2.0, resolution=0.01, orient="horizontal", width=10, length=90, showvalue=False, command=update_mass4)
slider6.place(x=320, y=208)

slider_value6 = tk.Label(root, text="0.01 kg", font=("Comic Sans MS", 10))
slider_value6.place(x=430, y=205)

label7 = tk.Label(root, text="Hinge friction on first rod:", font=("Comic Sans MS", 10))
label7.place(x=100, y=236)

slider7 = tk.Scale(root, from_=0.001, to=1, resolution=0.001, orient="horizontal", width=10, length=90, showvalue=False, command=update_friction1)
slider7.place(x=320, y=239)

slider_value7 = tk.Label(root, text="0.001 N·m·s/rad", font=("Comic Sans MS", 10))
slider_value7.place(x=430, y=236)

label8 = tk.Label(root, text="Hinge friction on second rod:", font=("Comic Sans MS", 10))
label8.place(x=100, y=267)

slider8 = tk.Scale(root, from_=0.001, to=1, resolution=0.01, orient="horizontal", width=10, length=90, showvalue=False, command=update_friction2)
slider8.place(x=320, y=270)

slider_value8 = tk.Label(root, text="0.001 N·m·s/rad", font=("Comic Sans MS", 10))
slider_value8.place(x=430, y=267)

label9 = tk.Label(root, text="Drag coefficient (Air resistance):", font=("Comic Sans MS", 10))
label9.place(x=100, y=298)

slider9 = tk.Scale(root, from_=0, to=1.5, resolution=0.1, orient="horizontal", width=10, length=90, showvalue=False, command=update_resistance)
slider9.place(x=320, y=301)

slider_value9 = tk.Label(root, text="0 Cd", font=("Comic Sans MS", 10))
slider_value9.place(x=430, y=298)

label10 = tk.Label(root, text="Angle of first rod", font=("Comic Sans MS", 10))
label10.place(x=100, y=329)

slider10 = tk.Scale(root, from_=-np.pi, to=np.pi, resolution=0.01, orient="horizontal", width=10, length=90, showvalue=False, command=update_angle1)
slider10.place(x=320, y=332)

slider_value10 = tk.Label(root, text="0 rad", font=("Comic Sans MS", 10))
slider_value10.place(x=430, y=329)

label11 = tk.Label(root, text="Angle of second rod", font=("Comic Sans MS", 10))
label11.place(x=100, y=360)

slider11 = tk.Scale(root, from_=-np.pi, to=np.pi, resolution=0.01, orient="horizontal", width=10, length=90, showvalue=False, command=update_angle2)
slider11.place(x=320, y=363)

slider_value11 = tk.Label(root, text="0 rad", font=("Comic Sans MS", 10))
slider_value11.place(x=430, y=360)

label12 = tk.Label(root, text="Angular velocity of rod 1", font=("Comic Sans MS", 10))
label12.place(x=100, y=391)

slider12 = tk.Scale(root, from_=0, to=10, resolution=0.1, orient="horizontal", width=10, length=90, showvalue=False, command=update_velocity1)
slider12.place(x=320, y=394)

slider_value12 = tk.Label(root, text="0 rad/s", font=("Comic Sans MS", 10))
slider_value12.place(x=430, y=391)

label13 = tk.Label(root, text="Angular velocity of rod 2", font=("Comic Sans MS", 10))
label13.place(x=100, y=422)

slider13 = tk.Scale(root, from_=0, to=10, resolution=0.1, orient="horizontal", width=10, length=90, showvalue=False, command=update_velocity2)
slider13.place(x=320, y=425)

slider_value13 = tk.Label(root, text="0 rad/s", font=("Comic Sans MS", 10))
slider_value13.place(x=430, y=422)

start_button = tk.Button(root, text="Start simulation", bg="green", fg="white", activebackground="darkgreen", activeforeground="white",
    command=lambda: main(slider1.get(), slider2.get(), slider3.get(), slider5.get(), slider9.get(), slider4.get(), slider6.get(), slider7.get(), slider8.get(), slider10.get(), slider12.get(), slider11.get(), slider13.get(), root)
)
start_button.place(x=100, y=473)

default_button = tk.Button(root, text="Recommended configurations", command=set_default_settings, bg="blue", fg="white", activebackground="darkblue", activeforeground="white")
default_button.place(x=210, y=473)

reset_button = tk.Button(root, text="Reset configurations", command=reset_settings, bg="orange", activebackground="darkorange")
reset_button.place(x=400, y=473)

stop_button = tk.Button(root, text="Stop Animation", command=lambda: stop_animation(), bg="red", activebackground="darkred", foreground="white", activeforeground="white")
stop_button.place(x=540, y=473)

show_angular_button = tk.Button(root, text="Show Angular Position", command=lambda: show_angular_positions())
show_angular_button.place(x=100, y=513)

show_velocities_button = tk.Button(root, text="Show Angular Velocity", command=lambda: show_angular_velocities())
show_velocities_button.place(x=250, y=513)

show_energy_button = tk.Button(root, text="Show Mechanical Energy", command=lambda: show_mechanical_energy())
show_energy_button.place(x=400, y=513)

show_chaos_button = tk.Button(root, text="Show Chaos Analysis", command=lambda: show_chaos_analysis())
show_chaos_button.place(x=560, y=513)

root.mainloop()
