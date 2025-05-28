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
import tkinter.messagebox as messagebox

# Main application window
root = tk.Tk()
root.title("Double Pendulum Simulation")
root.geometry("1920x1080")

# Initialize global variables
fig = None
ani = None
canvas_widget = None

# Create a welcome canvas with title
canvas1 = tk.Canvas(root, width=500, height=500, bg="black")
canvas1.place(x=700, y=50)
canvas1.create_text(250, 100, text="Double Pendulum", fill="white", font=("Arial", 30))
canvas1.create_text(250, 170, text="Simulation", fill="white", font=("Arial", 25))

# Try to load welcome image, but don't crash if it doesn't exist
try:
    img = Image.open("./OIP.jpeg")
    img = img.resize((240, 200))
    tk_img = ImageTk.PhotoImage(img)
    canvas1.create_image(250, 340, image=tk_img)
    # Keep a reference to prevent garbage collection
    canvas1.image = tk_img
except (FileNotFoundError, Exception):
    # Create a simple placeholder if image isn't found
    canvas1.create_rectangle(130, 240, 370, 440, outline="white", width=2)
    canvas1.create_text(250, 340, text="Double Pendulum\nDiagram", fill="white", font=("Arial", 12))

def validate_inputs(m1, m2, L1, L2, mr1, mr2, bf1, bf2, Cd, theta1, theta2, omega1, omega2):
    """Validate input parameters before simulation."""
    errors = []
    
    # Mass checks
    if m1 <= 0 or m2 <= 0:
        errors.append("Masses must be positive.")
    if mr1 < 0 or mr2 < 0:
        errors.append("Rod masses cannot be negative.")
    
    # Length checks
    if L1 <= 0 or L2 <= 0:
        errors.append("Rod lengths must be positive.")
    
    # Friction/drag checks
    if bf1 < 0 or bf2 < 0 or Cd < 0:
        errors.append("Friction/drag coefficients cannot be negative.")
    
    # Angle checks (should be within -π to π)
    if not (-np.pi <= theta1 <= np.pi) or not (-np.pi <= theta2 <= np.pi):
        errors.append("Angles must be between -π and π radians.")
    
    if errors:
        messagebox.showerror("Invalid Input", "\n".join(errors))
        return False
    return True

def set_boundary_case(case):
    """Set inputs to demonstrate a boundary case."""
    if case == "min_masses":
        slider1.set(0.01)
        slider2.set(0.01)
    elif case == "max_masses":
        slider1.set(5.0)
        slider2.set(5.0)
    elif case == "min_lengths":
        slider3.set(0.1)
        slider5.set(0.1)
    elif case == "max_lengths":
        slider3.set(3.0)
        slider5.set(3.0)
    elif case == "high_energy":
        slider10.set(np.pi)
        slider11.set(np.pi)
        slider12.set(10.0)
        slider13.set(10.0)
    elif case == "no_drag":
        slider9.set(0.0)
        slider7.set(0.001)
        slider8.set(0.001)
    elif case == "high_drag":
        slider9.set(1.5)
        slider7.set(1.0)
        slider8.set(1.0)
    elif case == "inverted":
        slider10.set(np.pi)
        slider11.set(np.pi)
        slider12.set(0.0)
        slider13.set(0.0)
    elif case == "asymmetric":
        slider1.set(5.0)
        slider2.set(0.01)
        slider3.set(3.0)
        slider5.set(0.1)

def main(m1_val, m2_val, L1_val, L2_val, drag_coeff, mass_rod1, mass_rod2, hinge_friction1, hinge_friction2, theta_val1, ang_velocity1, theta_val2, ang_velocity2, root):
    global fig, ani, canvas_widget
    
    if not validate_inputs(
        m1_val, m2_val, L1_val, L2_val, mass_rod1, mass_rod2,
        hinge_friction1, hinge_friction2, drag_coeff,
        theta_val1, theta_val2, ang_velocity1, ang_velocity2
    ):
        return
    
    # Clean up previous animation and canvas
    if ani is not None:
        ani.event_source.stop()
    if canvas_widget is not None:
        canvas_widget.get_tk_widget().destroy()
    if fig is not None:
        plt.close(fig)
    
    # Hide welcome canvas
    canvas1.place_forget()

    # Define symbolic variables for deriving equations of motion
    t, g = smp.symbols('t g')
    m1, m2 = smp.symbols('m1 m2')
    L1, L2 = smp.symbols('L1, L2')
    mr1, mr2 = smp.symbols('mr1 mr2')

    # Define the angle functions of time
    the1, the2 = smp.symbols(r'\theta_1, \theta_2', cls=smp.Function)
    the1 = the1(t)
    the2 = the2(t)

    # Calculate derivatives
    the1_d = smp.diff(the1, t)
    the2_d = smp.diff(the2, t)
    the1_dd = smp.diff(the1_d, t)
    the2_dd = smp.diff(the2_d, t)

    # Calculate positions of the point masses
    x1 = L1*smp.sin(the1)
    y1 = -L1*smp.cos(the1)
    x2 = L1*smp.sin(the1)+L2*smp.sin(the2)
    y2 = -L1*smp.cos(the1)-L2*smp.cos(the2)

    # Calculate kinetic energies including rod inertia
    T1_point = smp.Rational(1,2) * m1 * (smp.diff(x1, t)**2 + smp.diff(y1, t)**2)
    T2_point = smp.Rational(1,2) * m2 * (smp.diff(x2, t)**2 + smp.diff(y2, t)**2)
    
    # Rod rotational kinetic energies
    I1 = mr1 * L1**2 / 3
    I2 = mr2 * L2**2 / 3
    T1_rod = smp.Rational(1,2) * I1 * the1_d**2
    T2_rod = smp.Rational(1,2) * I2 * the2_d**2
    
    T = T1_point + T2_point + T1_rod + T2_rod

    # Calculate potential energies
    V1 = m1*g*y1
    V2 = m2*g*y2
    V = V1 + V2

    # Lagrangian
    L_eq = T - V

    # Euler-Lagrange equations
    LE1 = smp.diff(L_eq, the1) - smp.diff(smp.diff(L_eq, the1_d), t).simplify()
    LE2 = smp.diff(L_eq, the2) - smp.diff(smp.diff(L_eq, the2_d), t).simplify()

    print("Solving Lagrangian equations... (this may take a moment)")
    sols = smp.solve([LE1, LE2], (the1_dd, the2_dd), simplify=False, rational=False)

    # Convert to numerical functions
    dz1dt_f = smp.lambdify((t,g,m1,m2,L1,L2,mr1,mr2,the1,the2,the1_d,the2_d), sols[the1_dd])
    dz2dt_f = smp.lambdify((t,g,m1,m2,L1,L2,mr1,mr2,the1,the2,the1_d,the2_d), sols[the2_dd])
    dthe1dt_f = smp.lambdify(the1_d, the1_d)
    dthe2dt_f = smp.lambdify(the2_d, the2_d)

    def dSdt(S, t, g, m1, m2, L1, L2, mr1, mr2, Cd, rho, bf1, bf2):
        the1, z1, the2, z2 = S

        try:
            dz1dt = dz1dt_f(t, g, m1, m2, L1, L2, mr1, mr2, the1, the2, z1, z2)
            dz2dt = dz2dt_f(t, g, m1, m2, L1, L2, mr1, mr2, the1, the2, z1, z2)
        except (ZeroDivisionError, OverflowError):
            # Handle numerical instabilities
            return [z1, 0, z2, 0]

        # Add dissipative forces
        A1 = 0.01
        A2 = 0.01

        drag1 = -0.5 * Cd * rho * A1 * (L1**2) * z1 * abs(z1)
        drag2 = -0.5 * Cd * rho * A2 * (L2**2) * z2 * abs(z2)

        fric1 = -bf1 * z1
        fric2 = -bf2 * z2

        I_eff1 = m1 * L1**2 + mr1 * L1**2 / 3
        I_eff2 = m2 * L2**2 + mr2 * L2**2 / 3

        # Prevent division by zero
        if I_eff1 > 0:
            dz1dt += (drag1 + fric1) / I_eff1
        if I_eff2 > 0:
            dz2dt += (drag2 + fric2) / I_eff2

        return [dthe1dt_f(z1), dz1dt, dthe2dt_f(z2), dz2dt]

    # Simulation parameters
    t_sim = np.linspace(0, 40, 1001)
    g_val = 9.81
    m1_val = m1_val
    m2_val = m2_val
    L1_val = L1_val
    L2_val = L2_val
    mr1_val = mass_rod1
    mr2_val = mass_rod2
    Cd_val = drag_coeff
    rho_val = 1.225
    bf1_val = hinge_friction1
    bf2_val = hinge_friction2
    initial_conditions = [theta_val1, ang_velocity1, theta_val2, ang_velocity2]

    print("Solving differential equations...")
    try:
        ans = odeint(dSdt, y0=initial_conditions, t=t_sim,
                    args=(g_val, m1_val, m2_val, L1_val, L2_val, mr1_val, mr2_val, Cd_val, rho_val, bf1_val, bf2_val))
        
        # Create perturbed solution for chaos analysis
        ic_perturbed = [initial_conditions[0], initial_conditions[1], initial_conditions[2] + 0.001, initial_conditions[3]]
        ans_perturbed = odeint(dSdt, y0=ic_perturbed, t=t_sim, 
                              args=(g_val, m1_val, m2_val, L1_val, L2_val, mr1_val, mr2_val, Cd_val, rho_val, bf1_val, bf2_val))
    except Exception as e:
        messagebox.showerror("Simulation Error", f"Error solving equations: {str(e)}")
        return

    the1_arr, the2_arr = ans.T[0], ans.T[2]

    def get_x1y1x2y2(t, the1, the2, L1, L2):
        return (L1*np.sin(the1),
                -L1*np.cos(the1),
                L1*np.sin(the1) + L2*np.sin(the2),
                -L1*np.cos(the1) - L2*np.cos(the2))

    x1, y1, x2, y2 = get_x1y1x2y2(t_sim, the1_arr, the2_arr, L1_val, L2_val)

    # Calculate velocities for color mapping
    v1 = np.sqrt(np.diff(x1)**2 + np.diff(y1)**2)
    v2 = np.sqrt(np.diff(x2)**2 + np.diff(y2)**2)
    v1 = np.append(v1, v1[-1])
    v2 = np.append(v2, v2[-1])

    # Create animation
    fig, ax = plt.subplots(figsize=(5, 5), facecolor='#0E1117')
    ax.set_facecolor('#0E1117')
    ax.grid(color='#2A3459', linestyle='-', linewidth=0.3, alpha=0.7)
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_aspect('equal')

    ax.spines['bottom'].set_color('#4A5568')
    ax.spines['top'].set_color('#4A5568') 
    ax.spines['right'].set_color('#4A5568')
    ax.spines['left'].set_color('#4A5568')
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])

    # Animation elements
    line, = ax.plot([], [], '-', lw=2.5, color='#A3BFFA')
    mass1, = ax.plot([], [], 'o', markersize=15, color='#F687B3')
    mass2, = ax.plot([], [], 'o', markersize=10, color='#68D391')
    pivot, = ax.plot([0], [0], 'o', markersize=8, color='#F6E05E')

    trace_length = 100
    trace = ax.plot([], [], '-', lw=1.5, alpha=0.6)[0]
    trace_x, trace_y = [], []

    time_text = ax.text(-3.8, 3.7, '', fontsize=10, color='white')
    param_text = ax.text(-3.8, -3.7, f'$m_1$={m1_val}, $m_2$={m2_val}, $L_1$={L1_val}, $L_2$={L2_val}', 
                        fontsize=10, color='white')

    plt.title('Double Pendulum Simulation', color='white', fontsize=16, pad=20, 
            fontweight='bold', fontfamily='serif')

    canvas_widget = FigureCanvasTkAgg(fig, master=root)
    canvas_widget.draw()
    canvas_widget.get_tk_widget().place(x=700, y=50)

    def animate(i):
        if i >= len(x1):
            return line, mass1, mass2, trace, time_text
            
        line.set_data([0, x1[i], x2[i]], [0, y1[i], y2[i]])
        mass1.set_data([x1[i]], [y1[i]])
        mass2.set_data([x2[i]], [y2[i]])
        
        trace_x.append(x2[i])
        trace_y.append(y2[i])
        
        if len(trace_x) > trace_length:
            trace_x.pop(0)
            trace_y.pop(0)
        
        trace.set_data(trace_x, trace_y)
        time_text.set_text(f'Time: {t_sim[i]:.2f}s')
        
        return line, mass1, mass2, trace, time_text

    print("Creating animation...")
    ani = animation.FuncAnimation(fig, animate, frames=len(t_sim), 
                                interval=40, blit=True, repeat=True)

    # Energy calculations for analysis functions
    theta1 = ans[:, 0]
    omega1 = ans[:, 1]
    theta2 = ans[:, 2]
    omega2 = ans[:, 3]

    x1_e = L1_val * np.sin(theta1)
    y1_e = -L1_val * np.cos(theta1)
    x2_e = L1_val * np.sin(theta1) + L2_val * np.sin(theta2)
    y2_e = -L1_val * np.cos(theta1) - L2_val * np.cos(theta2)

    vx1 = np.gradient(x1_e, t_sim)
    vy1 = np.gradient(y1_e, t_sim)
    vx2 = np.gradient(x2_e, t_sim)
    vy2 = np.gradient(y2_e, t_sim)

    KE1 = 0.5 * m1_val * (vx1**2 + vy1**2)
    KE2 = 0.5 * m2_val * (vx2**2 + vy2**2)

    I1 = mr1_val * L1_val**2 / 3
    I2 = mr2_val * L2_val**2 / 3
    RE1 = 0.5 * I1 * omega1**2
    RE2 = 0.5 * I2 * omega2**2

    PE1 = m1_val * g_val * y1_e
    PE2 = m2_val * g_val * y2_e

    E_total = KE1 + KE2 + RE1 + RE2 + PE1 + PE2

    # Analysis functions
    def show_angular_positions():
        popup = tk.Toplevel()
        popup.geometry("700x500")
        popup.title("Angular Positions Graph")

        fig_pos, ax_pos = plt.subplots()
        ax_pos.plot(t_sim, theta1, label=r'$\theta_1$', color='cyan')
        ax_pos.plot(t_sim, theta2, label=r'$\theta_2$', color='magenta')
        ax_pos.set_xlabel('Time (s)')
        ax_pos.set_ylabel('Angle (rad)')
        ax_pos.set_title('Angular Position vs. Time')
        ax_pos.grid(True, linestyle='--', alpha=0.5)
        ax_pos.legend()
        
        canvas_pos = FigureCanvasTkAgg(fig_pos, master=popup)
        canvas_pos.draw()
        canvas_pos.get_tk_widget().pack()

    def show_angular_velocities():
        popup = tk.Toplevel()
        popup.geometry("700x500")
        popup.title("Angular Velocity Graph")

        fig_vel, ax_vel = plt.subplots()
        ax_vel.plot(t_sim, omega1, label=r'$\omega_1$', color='blue')
        ax_vel.plot(t_sim, omega2, label=r'$\omega_2$', color='red')
        ax_vel.set_xlabel('Time (s)')
        ax_vel.set_ylabel('Angular Velocity (rad/s)')
        ax_vel.set_title('Angular Velocity vs. Time')
        ax_vel.grid(True, linestyle='--', alpha=0.5)
        ax_vel.legend()

        canvas_vel = FigureCanvasTkAgg(fig_vel, master=popup)
        canvas_vel.draw()
        canvas_vel.get_tk_widget().pack()

    def show_mechanical_energy():
        popup = tk.Toplevel()
        popup.geometry("700x500")
        popup.title("Mechanical Energy Graph")

        fig_energy, ax_energy = plt.subplots()
        ax_energy.plot(t_sim, E_total)
        ax_energy.set_xlabel('Time (s)')
        ax_energy.set_ylabel('Mechanical Energy (J)')
        ax_energy.set_title('Energy Dissipation Over Time')
        ax_energy.grid(True, linestyle='--', alpha=0.5)

        canvas_energy = FigureCanvasTkAgg(fig_energy, master=popup)
        canvas_energy.draw()
        canvas_energy.get_tk_widget().pack()

    def show_chaos_analysis():
        popup = tk.Toplevel()
        popup.geometry("700x500")
        popup.title("Chaos Analysis Graph")

        fig_chaos, ax_chaos = plt.subplots()
        ax_chaos.plot(t_sim, np.abs(ans[:, 2] - ans_perturbed[:, 2]), 'r')
        ax_chaos.set_xlabel('Time (s)')
        ax_chaos.set_ylabel('|Δθ₂| (rad)')
        ax_chaos.set_title('Chaos: Angle Divergence from Δθ₂(0) = 0.001 rad')
        ax_chaos.grid(True, linestyle='--', alpha=0.5)

        canvas_chaos = FigureCanvasTkAgg(fig_chaos, master=popup)
        canvas_chaos.draw()
        canvas_chaos.get_tk_widget().pack()

    # Store functions globally for button access
    global show_angular_positions_func, show_angular_velocities_func, show_mechanical_energy_func, show_chaos_analysis_func
    show_angular_positions_func = show_angular_positions
    show_angular_velocities_func = show_angular_velocities
    show_mechanical_energy_func = show_mechanical_energy
    show_chaos_analysis_func = show_chaos_analysis

# Update functions for sliders
def update_mass1(value):
    slider_value1.config(text=f"{float(value):.2f} kg")

def update_mass2(value):
    slider_value2.config(text=f"{float(value):.2f} kg")

def update_length1(value):
    slider_value3.config(text=f"{float(value):.1f} m")

def update_mass3(value):
    slider_value4.config(text=f"{float(value):.2f} kg")

def update_mass4(value):
    slider_value6.config(text=f"{float(value):.2f} kg")

def update_length2(value):
    slider_value5.config(text=f"{float(value):.1f} m")

def update_friction1(value):
    slider_value7.config(text=f"{float(value):.3f} N·m·s/rad")

def update_friction2(value):
    slider_value8.config(text=f"{float(value):.3f} N·m·s/rad")

def update_resistance(value):
    slider_value9.config(text=f"{float(value):.1f} Cd")

def update_angle1(value):
    slider_value10.config(text=f"{float(value):.2f} rad")

def update_angle2(value):
    slider_value11.config(text=f"{float(value):.2f} rad")

def update_velocity1(value):
    slider_value12.config(text=f"{float(value):.1f} rad/s")

def update_velocity2(value):
    slider_value13.config(text=f"{float(value):.1f} rad/s")

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
    slider1.set(0.01)
    slider2.set(0.01)
    slider3.set(0.1)
    slider4.set(0.01)
    slider5.set(0.1)
    slider6.set(0.01)
    slider7.set(0.001)
    slider8.set(0.001)
    slider9.set(0)
    slider10.set(0)
    slider11.set(0)
    slider12.set(0)
    slider13.set(0)    

def stop_animation():
    global ani
    if ani is not None:
        ani.event_source.stop()

def open_boundary_case_menu():
    """Open a menu to select boundary cases."""
    menu = tk.Toplevel(root)
    menu.title("Select Boundary Case")
    menu.geometry("300x400")
    
    cases = [
        ("Minimum Masses (0.01 kg)", "min_masses"),
        ("Maximum Masses (5.0 kg)", "max_masses"),
        ("Minimum Lengths (0.1 m)", "min_lengths"),
        ("Maximum Lengths (2.0 m)", "max_lengths"),
        ("High Energy (Chaotic)", "high_energy"),
        ("No Drag/Friction", "no_drag"),
        ("High Drag/Friction", "high_drag"),
        ("Inverted Pendulum", "inverted"),
        ("Asymmetric System", "asymmetric")
    ]
    
    for i, (label, case) in enumerate(cases):
        tk.Button(
            menu, 
            text=label, 
            command=lambda c=case: (set_boundary_case(c), menu.destroy()),
            width=30, 
            background="grey", foreground="white", activebackground="grey", activeforeground="white"
        ).pack(pady=5)

# Create sliders and labels

head = tk.Label(root, text="Computational Physics Final Project", font=("Arial", 16))
head.place(x=100, y=25)

label1 = tk.Label(root, text="Mass of first pendulum:", font=("Arial", 10))
label1.place(x=100, y=80)

slider1 = tk.Scale(root, from_=0.01, to=5.0, resolution=0.01, orient="horizontal", width=10, length=90, showvalue=False, command=update_mass1)
slider1.place(x=360, y=83)
slider1.set(1.0)

slider_value1 = tk.Label(root, text="1.00 kg", font=("Arial", 10))
slider_value1.place(x=470, y=80)

label2 = tk.Label(root, text="Mass of second pendulum:", font=("Arial", 10))
label2.place(x=100, y=111)

slider2 = tk.Scale(root, from_=0.01, to=5.0, resolution=0.01, orient="horizontal", width=10, length=90, showvalue=False, command=update_mass2)
slider2.place(x=360, y=114)
slider2.set(1.0)

slider_value2 = tk.Label(root, text="1.00 kg", font=("Arial", 10))
slider_value2.place(x=470, y=111)

label3 = tk.Label(root, text="Length of first rod:", font=("Arial", 10))
label3.place(x=100, y=142)

slider3 = tk.Scale(root, from_=0.1, to=2.0, resolution=0.1, orient="horizontal", width=10, length=90, showvalue=False, command=update_length1)
slider3.place(x=360, y=145)
slider3.set(1.0)

slider_value3 = tk.Label(root, text="1.0 m", font=("Arial", 10))
slider_value3.place(x=470, y=142)

label4 = tk.Label(root, text="Mass of first rod:", font=("Arial", 10))
label4.place(x=100, y=173)

slider4 = tk.Scale(root, from_=0.01, to=2.0, resolution=0.01, orient="horizontal", width=10, length=90, showvalue=False, command=update_mass3)
slider4.place(x=360, y=176)
slider4.set(0.1)
 
slider_value4 = tk.Label(root, text="0.10 kg", font=("Arial", 10))
slider_value4.place(x=470, y=173)

label5 = tk.Label(root, text="Length of second rod:", font=("Arial", 10))
label5.place(x=100, y=204)

slider5 = tk.Scale(root, from_=0.1, to=2.0, resolution=0.1, orient="horizontal", width=10, length=90, showvalue=False, command=update_length2)
slider5.place(x=360, y=207)
slider5.set(1.0)

slider_value5 = tk.Label(root, text="1.0 m", font=("Arial", 10))
slider_value5.place(x=470, y=204)

label6 = tk.Label(root, text="Mass of second rod:", font=("Arial", 10))
label6.place(x=100, y=235)

slider6 = tk.Scale(root, from_=0.01, to=2.0, resolution=0.01, orient="horizontal", width=10, length=90, showvalue=False, command=update_mass4)
slider6.place(x=360, y=238)
slider6.set(0.1)

slider_value6 = tk.Label(root, text="0.10 kg", font=("Arial", 10))
slider_value6.place(x=470, y=235)

label7 = tk.Label(root, text="Hinge friction on first rod:", font=("Arial", 10))
label7.place(x=100, y=266)

slider7 = tk.Scale(root, from_=0.001, to=1.0, resolution=0.001, orient="horizontal", width=10, length=90, showvalue=False, command=update_friction1)
slider7.place(x=360, y=269)
slider7.set(0.05)

slider_value7 = tk.Label(root, text="0.050 N·m·s/rad", font=("Arial", 10))
slider_value7.place(x=470, y=266)

label8 = tk.Label(root, text="Hinge friction on second rod:", font=("Arial", 10))
label8.place(x=100, y=297)

slider8 = tk.Scale(root, from_=0.001, to=1.0, resolution=0.001, orient="horizontal", width=10, length=90, showvalue=False, command=update_friction2)
slider8.place(x=360, y=300)
slider8.set(0.05)

slider_value8 = tk.Label(root, text="0.050 N·m·s/rad", font=("Arial", 10))
slider_value8.place(x=470, y=297)

label9 = tk.Label(root, text="Air resistance coefficient:", font=("Arial", 10))
label9.place(x=100, y=328)

slider9 = tk.Scale(root, from_=0.0, to=1.5, resolution=0.1, orient="horizontal", width=10, length=90, showvalue=False, command=update_resistance)
slider9.place(x=360, y=331)
slider9.set(0.1)

slider_value9 = tk.Label(root, text="0.1 Cd", font=("Arial", 10))
slider_value9.place(x=470, y=328)

label10 = tk.Label(root, text="Initial angle of first pendulum:", font=("Arial", 10))
label10.place(x=100, y=359)

slider10 = tk.Scale(root, from_=-np.pi, to=np.pi, resolution=0.01, orient="horizontal", width=10, length=90, showvalue=False, command=update_angle1)
slider10.place(x=360, y=362)
slider10.set(np.pi/4)

slider_value10 = tk.Label(root, text="0.79 rad", font=("Arial", 10))
slider_value10.place(x=470, y=359)

label11 = tk.Label(root, text="Initial angle of second pendulum:", font=("Arial", 10))
label11.place(x=100, y=390)

slider11 = tk.Scale(root, from_=-np.pi, to=np.pi, resolution=0.01, orient="horizontal", width=10, length=90, showvalue=False, command=update_angle2)
slider11.place(x=360, y=393)
slider11.set(-np.pi/4)

slider_value11 = tk.Label(root, text="-0.79 rad", font=("Arial", 10))
slider_value11.place(x=470, y=390)

label12 = tk.Label(root, text="Initial angular velocity of first pendulum:", font=("Arial", 10))
label12.place(x=100, y=421)

slider12 = tk.Scale(root, from_=-10.0, to=10.0, resolution=0.1, orient="horizontal", width=10, length=90, showvalue=False, command=update_velocity1)
slider12.place(x=360, y=424)
slider12.set(0.0)

slider_value12 = tk.Label(root, text="0.0 rad/s", font=("Arial", 10))
slider_value12.place(x=470, y=421)

label13 = tk.Label(root, text="Initial angular velocity of second pendulum:", font=("Arial", 10))
label13.place(x=100, y=452)

slider13 = tk.Scale(root, from_=-10.0, to=10.0, resolution=0.1, orient="horizontal", width=10, length=90, showvalue=False, command=update_velocity2)
slider13.place(x=360, y=455)
slider13.set(0.0)

slider_value13 = tk.Label(root, text="0.0 rad/s", font=("Arial", 10))
slider_value13.place(x=470, y=452)

# Create buttons
start_button = tk.Button(root, text="Start Simulation", background="green", foreground="white", activebackground="green", activeforeground="white", command=lambda: main(
    slider1.get(), slider2.get(), slider3.get(), slider5.get(),
    slider9.get(), slider4.get(), slider6.get(), slider7.get(),
    slider8.get(), slider10.get(), slider12.get(), slider11.get(),
    slider13.get(), root), width=15, height=1)
start_button.place(x=100, y=510)

stop_button = tk.Button(root, text="Stop Animation", background="darkred", foreground="white", activebackground="darkred", activeforeground="white", command=stop_animation, width=15, height=1)
stop_button.place(x=240, y=510)

default_button = tk.Button(root, text="Default Settings", background="grey", foreground="white", activebackground="grey", activeforeground="white", command=set_default_settings, width=15, height=1)
default_button.place(x=100, y=580)

reset_button = tk.Button(root, text="Reset Settings", command=reset_settings, background="orange", foreground="black", activebackground="orange", activeforeground="black", width=15, height=1)
reset_button.place(x=100, y=545)

boundary_button = tk.Button(root, text="Boundary Cases", background="grey", foreground="white", activebackground="grey", activeforeground="white", command=open_boundary_case_menu, width=15, height=1)
boundary_button.place(x=240, y=545)

# Analysis buttons
analysis_frame = tk.LabelFrame(root, text="Analysis Tools")
analysis_frame.place(x=380, y=510, width=260, height=95)

pos_button = tk.Button(analysis_frame, text="Angular Positions", background="#4A1F8B", foreground="white", activebackground="#4A1F8B", activeforeground="white", command=lambda: show_angular_positions_func(), width=15)
pos_button.grid(row=0, column=0, padx=5, pady=5)

vel_button = tk.Button(analysis_frame, text="Angular Velocities", background="#4A1F8B", foreground="white", activebackground="#4A1F8B", activeforeground="white", command=lambda: show_angular_velocities_func(), width=15)
vel_button.grid(row=0, column=1, padx=5, pady=5)

energy_button = tk.Button(analysis_frame, text="Energy Analysis", background="#4A1F8B", foreground="white", activebackground="#4A1F8B", activeforeground="white", command=lambda: show_mechanical_energy_func(), width=15)
energy_button.grid(row=1, column=0, padx=5, pady=5)

chaos_button = tk.Button(analysis_frame, text="Chaos Analysis", background="#4A1F8B", foreground="white", activebackground="#4A1F8B", activeforeground="white", command=lambda: show_chaos_analysis_func(), width=15)
chaos_button.grid(row=1, column=1, padx=5, pady=5)

# Initialize analysis functions as None
show_angular_positions_func = None
show_angular_velocities_func = None
show_mechanical_energy_func = None
show_chaos_analysis_func = None

# Run the application
root.mainloop()