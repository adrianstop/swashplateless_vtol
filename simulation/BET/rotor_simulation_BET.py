import aerosandbox as asb
import numpy as np
import matplotlib.pyplot as plt
import os

fig_path = os.path.abspath(os.curdir) + '\\figures\\'
# Check and create output directory if it doesn't exist
if not os.path.exists(fig_path):
    os.makedirs(fig_path)

Omega = 5000 / 60
R_root = 33/1000 # m
R_tip = 190/1000 # m
alpha_neutral = np.radians(3)

N_steps = 360
thetas = np.linspace(0, 2 * np.pi, N_steps)
alphas = np.sin(thetas)*np.radians(5) + alpha_neutral

section_chord = 30/1000 # m
r = np.linspace(R_root, R_tip, N_steps)
section_width = (R_tip - R_root)/N_steps

airspeed = Omega * r

rho = 1.225 # kg/m^3
my = 1.42e-5 # Pa*s
re = rho * airspeed * section_chord / my

Alpha, Re = np.meshgrid(np.degrees(alphas), re)

airfoil_name = "NACA2208"
af = asb.Airfoil(airfoil_name)

aero_flattened = af.get_aero_from_neuralfoil(
    alpha=Alpha.flatten(),
    Re=Re.flatten(),
    mach=0,
    model_size="xxxlarge",
)
Aero = {
    key: value.reshape(Alpha.shape)
    for key, value in aero_flattened.items()
}

L = np.zeros(N_steps)

for i in range(N_steps):
    dL = 0.5*rho*airspeed**2*Aero["CL"][:,i]*section_chord
    #print(f"tip CL: {Aero['CL'][-1,i]}, root_CL: {Aero['CL'][0,i]}")
    #print(f"tip_airspeed: {airspeed[-1]}, root_airspeed: {airspeed[0]}")
    L[i] = np.trapz(dL, dx=section_width)
    
    
    
fig, ax1 = plt.subplots(figsize=(8,4))

color = 'tab:blue'
ax1.set_xlabel('Theta [deg]')
ax1.set_ylabel('L [N]', color=color)
ax1.plot(np.degrees(thetas), L, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:red'
ax2.set_ylabel(r'$\alpha_{{b}}$ [deg]', color=color)  # we already handled the x-label with ax1
ax2.plot(np.degrees(thetas), np.degrees(alphas), color=color)
ax2.tick_params(axis='y', labelcolor=color)

#fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.title(r'L and $\alpha_{{b}}$ vs Theta')
plt.grid(True)
plt.savefig(f'{fig_path}BET_varying_alpha.png')
