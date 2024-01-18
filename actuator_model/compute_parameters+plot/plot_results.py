import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import aerosandbox.tools.pretty_plots as p
import os
import numpy as np

# Read the CSV file
data = pd.read_csv('results_elevation_20DEC23.csv', skipinitialspace=True)

fig_path = os.path.join(os.getcwd(), 'figures')
# Check and create output directory if it doesn't exist
if not os.path.exists(fig_path):
    os.makedirs(fig_path)
    
# Convert FT_elevation_avg, FT_azimuth_avg and img_elevation to radians
data['FT_elevation_avg'] = np.radians(data['FT_elevation_avg'])
data['FT_azimuth_avg'] = np.radians(data['FT_azimuth_avg'])
data['img_elevation'] = np.radians(data['img_elevation'])

CAL_AZM_OFFSET = data.loc[data['azimuth_input'] == 0.0, 'FT_azimuth_avg'].mean()
print(f"CAL_AZM_OFFSET = {CAL_AZM_OFFSET}")
filtered_data = data[data['elevation_input'] > 0.2]
filtered_data.loc[:, 'FT_azimuth_avg'] += -CAL_AZM_OFFSET

#### THRUST ####
# Prepare the input data for a second degree polynomial fit
x = data['rps']
y = data['FT_thrust_avg']
A = np.vstack([x**2]).T
# Perform the least squares fit
coeffs, residuals, _, _ = np.linalg.lstsq(A, y, rcond=None)
# The coefficients are returned in the reverse order, so reverse them
coeffs_rps_thrust = coeffs[::-1]
residuals_rps_thrust = residuals
print(f"FT_thrust: square term: {coeffs_rps_thrust[0]}, residual: {residuals_rps_thrust}")

#### ELEVATION ####
# Prepare the input data for a second degree polynomial fit
x = data['elevation_input']
y = data['FT_elevation_avg']
A = np.vstack([x]).T
# Perform the least squares fit
coeffs, residuals, _, _ = np.linalg.lstsq(A, y, rcond=None)
# The coefficients are returned in the reverse order, so reverse them
coeffs_elev = coeffs[::-1]
residuals_elev = residuals
print(f"FT_elev: slope: {coeffs_elev[0]}, residual: {residuals_elev}")

#### AZIMUTH ####
# Prepare the input data for a second degree polynomial fit
x = filtered_data['azimuth_input']
y = filtered_data['FT_azimuth_avg']
A = np.vstack([x]).T
# Perform the least squares fit
coeffs, residuals, _, _ = np.linalg.lstsq(A, y, rcond=None)
# The coefficients are returned in the reverse order, so reverse them
coeffs_azim = coeffs[::-1]
residuals_azim = residuals
print(f"FT_azimuth: slope: {coeffs_azim[0]}, residual: {residuals_azim}")

#### TORQUE ####
# Prepare the input data for a second degree polynomial fit
x = data['FT_thrust_avg']
y = data['FT_torque_avg']
A = np.vstack([x]).T
# Perform the least squares fit
coeffs, residuals, _, _ = np.linalg.lstsq(A, y, rcond=None)
# The coefficients are returned in the reverse order, so reverse them
coeffs_thrust_torque = coeffs[::-1]
residuals_thrust_torque = residuals
print(f"FT_torque: slope: {coeffs_thrust_torque[0]}, residual: {residuals_thrust_torque}")

#### IMAGE ELEVATION ####
# Prepare the input data for a second degree polynomial fit
x = data['elevation_input']
y = data['img_elevation']
A = np.vstack([x]).T
# Perform the least squares fit
coeffs, residuals, _, _ = np.linalg.lstsq(A, y, rcond=None)
# The coefficients are returned in the reverse order, so reverse them
coeffs_elev_img = coeffs[::-1]
residuals_elev_img = residuals
print(f"Image elevation: slope: {coeffs_elev_img[0]}, residual: {residuals_elev_img}")

# Create a figure and three subplots
fig, axs = plt.subplots(4, 1, figsize=(8,12))

# Plot rps against FT_thrust_avg
rps_x_range = np.linspace(np.min(data['rps']), np.max(data['rps']), 100)
axs[0].plot(data['rps'], data['FT_thrust_avg'], 'o')
axs[0].plot(rps_x_range, np.polyval([coeffs_rps_thrust[0],0, 0], rps_x_range), 'r-')
axs[0].set_xlabel(r'rps, $\Omega$')
axs[0].set_ylabel('FT_thrust_avg [N]')
axs[0].legend(["FT thrust avg", "Best fit"],loc = 'upper left')
axs[0].annotate(r'$T = {:.2f}\Omega^2$'.format(coeffs_thrust_torque[0]), xy=(0.6, 0.1), xycoords='axes fraction', fontsize=12)

# Plot elevation_input against FT_elevation_avg
axs[1].plot(data['elevation_input'], data['FT_elevation_avg'], 'o')
axs[1].plot(data['elevation_input'], np.polyval([coeffs_elev[0], 0], data['elevation_input']), 'r-')
axs[1].set_xlabel(r'Elevation_input, $A$')
axs[1].set_ylabel('FT_elevation_avg [rad]')
axs[1].legend(["FT elevation avg", "Best fit"],loc = 'upper left')
axs[1].annotate(r'$\phi_{{elev}} = {:.2f}A$'.format(coeffs_elev[0]), xy=(0.6, 0.1), xycoords='axes fraction', fontsize=12)

# Plot azimuth_input against FT_azimuth_avg
axs[2].plot(filtered_data['azimuth_input'], filtered_data['FT_azimuth_avg'], 'o')
axs[2].plot(filtered_data['azimuth_input'], np.polyval([coeffs_azim[0], 0], filtered_data['azimuth_input']), 'r-')
axs[2].set_xlabel(r'azimuth_input, $\psi_k$')
axs[2].set_ylabel('FT_azimuth_avg [rad]')
axs[2].legend(["FT azimuth avg", "Best fit"],loc = 'upper left')
axs[2].annotate(r'$\psi_{{azm}} = {:.2f}(\psi_k + \psi_{{zero}})$'.format(coeffs_azim[0]), xy=(0.6, 0.24), xycoords='axes fraction', fontsize=12)
axs[2].annotate(r'$\psi_{{zero}} = {:.2f}$'.format(CAL_AZM_OFFSET), xy=(0.6, 0.1), xycoords='axes fraction', fontsize=12)

# Plot FT_thrust_avg against FT_torque_avg
axs[3].plot(data['FT_thrust_avg'], data['FT_torque_avg'], 'o')
axs[3].plot(data['FT_thrust_avg'], np.polyval([coeffs_thrust_torque[0], 0], data['FT_thrust_avg']), 'r-')
axs[3].set_xlabel(r'FT_thrust_avg, $T$')
axs[3].set_ylabel('FT_torque_avg [Nm]')
axs[3].legend(["FT torque avg", "Best fit"],loc = 'upper left')
axs[3].annotate(r'$\tau_{{drag}} = {:.2f}T$'.format(coeffs_thrust_torque[0]), xy=(0.6, 0.1), xycoords='axes fraction', fontsize=12)


# Show the plots
fig.suptitle(f'FT parameter estimation results')
plt.tight_layout()
#plt.savefig(os.path.join(fig_path, 'ft_param_estim.png'))
plt.close()


# Group the data by 'rps'
grouped = data.groupby('rps')

colors = LinearSegmentedColormap.from_list(
    "custom_cmap",
    colors=[
        p.adjust_lightness(c, 0.8) for c in
        ["orange", "darkseagreen", "dodgerblue"]
    ]
)(np.linspace(0, 1, grouped.ngroups))
#colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

fig_elev, axs_elev = plt.subplots(figsize=(8, 5.5))
# For each group, plot 'elevation_input' against 'FT_elevation_avg'
for (name, group), color in zip(grouped, colors):
    # Calculate the offset for 'FT_elevation_avg'
    #offset = group['FT_elevation_avg'].iloc[0]
    # Subtract the offset from 'FT_elevation_avg'
    #group['FT_elevation_avg'] -= offset
    #axs.errorbar(group['elevation_input'], group['FT_elevation_avg'], yerr=np.sqrt(group['FT_elevation_avg_var']), label=f'FT_elevation_avg for rps {name}', color=color, linestyle='-', marker='o', capsize=3)
    axs_elev.plot(group['elevation_input'], np.abs(group['FT_elevation_avg']), 'o', label=f'FT elev for {name} rps', color=color)
    #axs.fill_between(group['elevation_input'], group['FT_elevation_avg'] - np.sqrt(group['FT_elevation_avg_var']), group['FT_elevation_avg'] + np.sqrt(group['FT_elevation_avg_var']), color=color, alpha=0.5)
    axs_elev.plot(group['elevation_input'], group['img_elevation'], 'x', label=f'image elev for {name} rps', color=color)
    

axs_elev.plot(data['elevation_input'], 1.1693*data['elevation_input'], label=f'FT best fit', color='red', linestyle='-')
axs_elev.annotate(r'$\phi_{{elev}} = {:.2f}A$'.format(1.17), xy=(0.6, 0.1), xycoords='axes fraction', fontsize=12, color='red')
axs_elev.plot(data['elevation_input'], coeffs_elev_img*data['elevation_input'] , label=f'Image best fit', color='purple', linestyle='--')
axs_elev.annotate(r'$\phi_{{elev}} = {:.2f}A$'.format(coeffs_elev_img[0]), xy=(0.6, 0.18), xycoords='axes fraction', fontsize=12, color='purple')

# Add labels and title
axs_elev.set_title('Elevation input vs output grouped by RPS')
axs_elev.set_xlabel(r'Elevation input, $A$')
axs_elev.set_ylabel('Elevation angle [rad]')

# Add legend
plt.legend(loc = 'upper left')

# Save the plot


fig_elev.tight_layout()
plt.savefig(os.path.join(fig_path, 'elevation_angle_vs_input_inverted.png'))
plt.close()