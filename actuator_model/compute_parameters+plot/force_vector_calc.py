import matplotlib.pyplot as plt
import numpy as np
import csv
import os

def moving_average(input_arr, window_size):
    return np.convolve(input_arr, np.ones(window_size), 'valid') / window_size

# PARAMETERS
sampling_rate = 1000/16
force_threshold = 1 # N
padding_time = 1 # seconds
test_duration = 10 # seconds (TODO auto detect end of test)
average_samples = 10 # number of samples to average over

path = os.path.abspath(os.pardir) + "\log_files\\20DEC23\\"
fig_path = path + 'figures\\'
# Check and create output directory if it doesn't exist
if not os.path.exists(fig_path):
    os.makedirs(fig_path)
    
dirlist = os.listdir(path)
dirlist = [i for i in dirlist if ( i[-14:] != '_formatted.csv' and i != 'figures' ) ]
print (dirlist)

# data format: ["Force X [N]", "Force Y [N]", "Force Z [N]", "Torque X [N-m]", "Torque Y (N-m)", "Torque Z (N-m)"]
print("testno.,FT_thrust_avg,FT_elevation_avg,FT_azimuth_avg,FT_torque_avg,FT_thrust_avg_stddev,FT_elevation_avg_stddev,FT_azimuth_avg_stddev,FT_torque_avg_stddev")
for i in range(len(dirlist)):
    filepath_i = path + dirlist[i]
    fixed_filepath_i = filepath_i.replace('.csv', '_formatted.csv')
    
    # Fix format of data files
    text = open(filepath_i, "r")
    text = ''.join([i for i in text])
    text = text.replace(',', '.')
    text = text.replace('"', '')
    x = open(fixed_filepath_i,"w")
    x.writelines(text)
    x.close()
    
    # Read csv data file
    data = np.loadtxt(fixed_filepath_i, skiprows=1, usecols=(0,1,2,3,4,5), delimiter=';')
    num_rows = len(data[:, 0]) - len(data[:, 0])%average_samples # To make even number for averaging
    t = np.linspace(0, len(data[:, 0])/sampling_rate, int(num_rows/average_samples))
    
    x_force = data[:num_rows, 0]
    y_force = data[:num_rows, 1]
    z_force = data[:num_rows, 2]
    x_force_mavg = moving_average(x_force, average_samples)
    y_force_mavg = moving_average(y_force, average_samples)
    z_force_mavg = moving_average(z_force, average_samples)
    force_magnitude_mavg = np.sqrt(x_force_mavg**2 + y_force_mavg**2 + z_force_mavg**2)
    #force_magnitude_mavg = moving_average(force_magnitude, average_samples)
    azimuth = np.arctan2(y_force_mavg, x_force_mavg)*180/np.pi
    #elevation = 90 - np.arcsin(z_force_mavg/force_magnitude_mavg)*180/np.pi
    elevation = np.arctan2(np.sqrt(x_force_mavg**2 + y_force_mavg**2), np.abs(z_force_mavg))*180/np.pi
    max_force = np.max(np.max([x_force_mavg, y_force_mavg, z_force_mavg], axis=0))
    min_force = np.min(np.min([x_force_mavg, y_force_mavg, z_force_mavg], axis=0))
    
    x_torque = data[:num_rows, 3]
    y_torque = data[:num_rows, 4]
    z_torque = data[:num_rows, 5]
    x_torque_mavg = moving_average(x_torque, average_samples)
    y_torque_mavg = moving_average(y_torque, average_samples)
    z_torque_mavg = moving_average(z_torque, average_samples)
    torque_magnitude = np.sqrt(x_torque**2 + y_torque**2 + z_torque**2)
    torque_magnitude_mavg = moving_average(torque_magnitude, average_samples)
    max_torque = np.max([np.max(x_torque_mavg), np.max(y_torque_mavg), np.max(z_torque_mavg)])
    min_torque = np.min([np.min(x_torque_mavg), np.min(y_torque_mavg), np.min(z_torque_mavg)])
    
    start_sample = np.argmax(force_magnitude_mavg > force_threshold) - int(padding_time*sampling_rate)
    if start_sample < average_samples:
        start_sample = average_samples
    
    end_sample = start_sample + int(test_duration*sampling_rate) + int(padding_time*sampling_rate)
    if end_sample > len(force_magnitude_mavg):
        end_sample = len(force_magnitude_mavg)
    time_duration = (end_sample - start_sample)/sampling_rate
    t_adjusted = np.linspace(0, time_duration, end_sample - start_sample)
    t_adjusted_mavg = t_adjusted[average_samples:]
    
    avg_offset = int(sampling_rate*3)
    avg_duration = int(sampling_rate*3)
    center = (t_adjusted[avg_offset] + t_adjusted[avg_offset + avg_duration]) / 2 - 0.6
    
    thrust_avg = np.average(force_magnitude_mavg[start_sample+avg_offset:start_sample+avg_offset+avg_duration])
    azimuth_avg = np.average(azimuth[start_sample+avg_offset:start_sample+avg_offset+avg_duration])
    elevation_avg = np.average(elevation[start_sample+avg_offset:start_sample+avg_offset+avg_duration])
    torque_avg = np.average(torque_magnitude_mavg[start_sample+avg_offset:start_sample+avg_offset+avg_duration])
    avg_data = [thrust_avg, azimuth_avg, elevation_avg, torque_avg]
    
    thrust_avg_var = np.var(force_magnitude_mavg[start_sample+avg_offset:start_sample+avg_offset+avg_duration])
    azimuth_avg_var = np.var(azimuth[start_sample+avg_offset:start_sample+avg_offset+avg_duration])
    elevation_avg_var = np.var(elevation[start_sample+avg_offset:start_sample+avg_offset+avg_duration])
    torque_avg_var = np.var(torque_magnitude_mavg[start_sample+avg_offset:start_sample+avg_offset+avg_duration])
    
    thrust_avg_stddev = np.sqrt(thrust_avg_var)
    azimuth_avg_stddev = np.sqrt(azimuth_avg_var)
    elevation_avg_stddev = np.sqrt(elevation_avg_var)
    torque_avg_stddev = np.sqrt(torque_avg_var)
    print(f"{i},{thrust_avg},{elevation_avg},{azimuth_avg},{torque_avg},{torque_avg_stddev},{elevation_avg_stddev},{azimuth_avg_stddev},{torque_avg_stddev}")
    
   

    # Define the titles, y-labels, and data for the subplots
    subplots = [
        {"title": "Force magnitude", "ylabel": "force (N)", "data": force_magnitude_mavg[start_sample:end_sample]},
        {"title": "Elevation angle", "ylabel": "angle (deg)", "data": elevation[start_sample:end_sample]},
        {"title": "Azimuth angle", "ylabel": "angle (deg)", "data": azimuth[start_sample:end_sample]},
        {"title": "Torque magnitude", "ylabel": "torque (Nm)", "data": torque_magnitude_mavg[start_sample:end_sample]}
    ]
    # Plot data
    fig, axs = plt.subplots(len(subplots),1, figsize=(8, 8))
    fig.suptitle(f'Force and torque data for {dirlist[i]}')
    colors = ['purple', 'tab:blue', 'tab:green', 'tab:orange']
    annotate_y = [1, 5, -110, -0.03]
    force_padding = 0.1*(np.abs(max_force - min_force))
    torque_padding = 0.1*(np.abs(max_torque - min_torque))

    # For each subplot
    for j, subplot in enumerate(subplots):
        axs[j].set_title(subplot["title"])
        axs[j].set_xlabel('time (s)')
        axs[j].set_ylabel(subplot["ylabel"], color=colors[j])
        axs[j].plot(t_adjusted, subplot["data"], label=subplot["title"].lower(), color=colors[j])
        axs[j].tick_params(axis='y', labelcolor=colors[j])
        axs[j].axvspan(t_adjusted[avg_offset], t_adjusted[avg_offset + avg_duration], facecolor='yellow', alpha=0.5)
        #axs[j].annotate(f'avg = {avg_data[j]:.2f}', xy=(center, annotate_y[j]), xytext=(center, avg_data[j]), fontsize = 12 )

    # Set the y-limits manually for each subplot
    axs[0].set_ylim(min_force - force_padding, max_force + force_padding)
    axs[1].set_ylim(-3,40)
    axs[2].set_ylim(-185,185)
    axs[3].set_ylim(min_torque - torque_padding, max_torque + torque_padding)

    fig.tight_layout()
    plt.savefig(f'{fig_path}K_plot_{i}.png')
    plt.close()
    
    # Plot raw force data
    fig2, axs_raw = plt.subplots(3,2, figsize=(8, 8))
    fig2.suptitle(f'Force and torque data for {dirlist[i]}')
    colors = ['tab:red', 'tab:blue', 'tab:green']
    force_padding = 0.05*(max_force - min_force)
    torque_padding = 0.05*(max_torque - min_torque)
    colors = ['tab:red', 'tab:green','tab:blue', 'tab:orange', 'tab:purple', 'tab:brown']

    # Define the titles, y-labels, and data for the subplots
    subplots = [
        {"title": "X force", "ylabel": "Force [N]", "data": x_force_mavg[start_sample:end_sample]},
        {"title": "X torque", "ylabel": "Torque [Nm]", "data": x_torque_mavg[start_sample:end_sample]},
        {"title": "Y force", "ylabel": "Force [N]", "data": y_force_mavg[start_sample:end_sample]},
        {"title": "Y torque", "ylabel": "Torque [Nm]", "data": y_torque_mavg[start_sample:end_sample]},
        {"title": "Z force", "ylabel": "Force [N]", "data": z_force_mavg[start_sample:end_sample]},
        {"title": "Z torque", "ylabel": "Torque [Nm]", "data": z_torque_mavg[start_sample:end_sample]}
    ]

    # For each subplot
    for k, subplot in enumerate(subplots):
        row = k // 2
        col = k % 2
        axs_raw[row, col].set_title(subplot["title"])
        axs_raw[row, col].set_xlabel('time (s)')
        axs_raw[row, col].set_ylabel(subplot["ylabel"], color=colors[row])
        axs_raw[row, col].plot(t_adjusted, subplot["data"], label=subplot["title"].lower(), color=colors[row])
        axs_raw[row, col].tick_params(axis='y', labelcolor=colors[row])
        if col == 0:
            axs_raw[row, col].set_ylim(min_force - force_padding, max_force + force_padding)
        else:
            axs_raw[row, col].set_ylim(min_torque - torque_padding, max_torque + torque_padding)

    fig2.tight_layout()
    plt.savefig(f'{fig_path}force_torque_{i}.png')
    plt.close()
    