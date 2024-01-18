import matplotlib.pyplot as plt
import numpy as np
import csv
import os

# PARAMETERS
sampling_rate = 1000/16
force_threshold = 0.5 # N
padding_time = 1 # seconds
test_duration = 110 # seconds (TODO auto detect end of test)
average_samples = 10 # number of samples to average over

path = os.path.abspath(os.pardir) + "\log_files\\30OCT23\\"
fig_path = path + 'figures\\'
dirlist = os.listdir(path)
dirlist = [i for i in dirlist if ( i[-14:] != '_formatted.csv' and i != 'figures' ) ]
print (dirlist)

# data format: ["Force X [N]", "Force Y [N]", "Force Z [N]", "Torque X [N-m]", "Torque Y (N-m)", "Torque Z (N-m)"]
all_x_force = np.empty([len(dirlist), ])
all_y_force = np.array([])
all_z_force = np.array([])
all_x_torque = np.array([])
all_y_torque = np.array([])
all_z_torque = np.array([])

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
    print(np.shape(data))
    num_rows = len(data[:, 0]) - len(data[:, 0])%average_samples # To make even number for averaging
    averaged_rows = int(num_rows/average_samples)
    t = np.linspace(0, len(data[:, 0])/sampling_rate, averaged_rows)
    x_force = np.average(data[:num_rows, 0].reshape(-1, average_samples), axis=1)
    all_x_force = np.append(all_x_force, x_force)
    y_force = np.average(data[:num_rows, 1].reshape(-1, average_samples), axis=1)
    all_y_force = np.append(all_y_force, y_force)
    z_force = np.average(data[:num_rows, 2].reshape(-1, average_samples), axis=1)
    all_z_force = np.append(all_z_force, z_force)
    force_magnitude = np.sqrt(x_force**2 + y_force**2 + z_force**2)
    x_torque = np.average(data[:num_rows, 3].reshape(-1, average_samples), axis=1)
    all_x_torque = np.append(all_x_torque, x_torque)
    y_torque = np.average(data[:num_rows, 4].reshape(-1, average_samples), axis=1)
    all_y_torque = np.append(all_y_torque, y_torque)
    z_torque = np.average(data[:num_rows, 5].reshape(-1, average_samples), axis=1)
    all_z_torque = np.append(all_z_torque, z_torque)
    torque_magnitude = np.sqrt(x_torque**2 + y_torque**2 + z_torque**2)
    max_force = np.max([np.max(x_force), np.max(y_force), np.max(z_force)])
    print(f"Max force: {max_force}")
    print(f"Max x-force: {np.max(x_force)}")
    print(f"Max y-force: {np.max(y_force)}")
    print(f"Max z-force: {np.max(z_force)}")
    min_force = np.min([np.min(x_force), np.min(y_force), np.min(z_force)])
    print(f"Min force: {min_force}")
    print(f"Min x-force: {np.min(x_force)}")
    print(f"Min y-force: {np.min(y_force)}")
    print(f"Min z-force: {np.min(z_force)}")
    max_torque = np.max([np.max(x_torque), np.max(y_torque), np.max(z_torque)])
    print(f"Max torque: {max_torque}")
    print(f"Max x-torque: {np.max(x_torque)}")
    print(f"Max y-torque: {np.max(y_torque)}")
    print(f"Max z-torque: {np.max(z_torque)}")
    min_torque = np.min([np.min(x_torque), np.min(y_torque), np.min(z_torque)])
    print(f"Min torque: {min_torque}")
    print(f"Min x-torque: {np.min(x_torque)}")
    print(f"Min y-torque: {np.min(y_torque)}")
    print(f"Min z-torque: {np.min(z_torque)}")
    start_sample = np.argmax(force_magnitude > force_threshold) - int(padding_time*sampling_rate/average_samples)
    end_sample = start_sample + int(test_duration*sampling_rate/average_samples) + int(padding_time*sampling_rate/average_samples)
    time_duration = (end_sample - start_sample)*average_samples/sampling_rate
    t_adjusted = np.linspace(0, time_duration, end_sample - start_sample)
    
    
    # Plot data
    fig, axs = plt.subplots(3,2, figsize=(15, 15))
    fig.suptitle(f'Force and torque data for {dirlist[i]}')
    colors = ['tab:red', 'tab:blue', 'tab:green']
    force_padding = 0.05*(max_force - min_force)
    torque_padding = 0.05*(max_torque - min_torque)

    # X force
    axs[0,0].set_title(f'X force')
    axs[0,0].set_xlabel('time (s)')
    axs[0,0].set_ylabel('Force [N]', color=colors[0])
    axs[0,0].plot(t_adjusted, all_x_force[start_sample:end_sample,:], label='X force', color=colors[0])
    axs[0,0].tick_params(axis='y', labelcolor=colors[0])
    axs[0,0].set_ylim(min_force - force_padding, max_force + force_padding)
    
    # Y force 
    axs[1,0].set_title(f'Y force')
    axs[1,0].set_xlabel('time (s)')
    axs[1,0].set_ylabel('Force [N]', color=colors[1])
    axs[1,0].plot(t_adjusted, y_force[start_sample:end_sample], label='Y force', color=colors[1])
    axs[1,0].tick_params(axis='y', labelcolor=colors[1])
    axs[1,0].set_ylim(min_force - force_padding, max_force + force_padding)
    
    # Z force
    axs[2,0].set_title(f'Z force')
    axs[2,0].set_xlabel('time (s)')
    axs[2,0].set_ylabel('Force [N]', color=colors[2])
    axs[2,0].plot(t_adjusted, z_force[start_sample:end_sample], label='Z force', color=colors[2])
    axs[2,0].tick_params(axis='y', labelcolor=colors[2])
    axs[2,0].set_ylim(min_force - force_padding, max_force + force_padding)
    
    # X torque
    axs[0,1].set_title(f'X torque')
    axs[0,1].set_xlabel('time (s)')
    axs[0,1].set_ylabel('Torque [Nm]', color=colors[0])
    axs[0,1].plot(t_adjusted, x_torque[start_sample:end_sample], label='X torque', color=colors[0])
    axs[0,1].tick_params(axis='y', labelcolor=colors[0])
    axs[0,1].set_ylim(min_torque - torque_padding, max_torque + torque_padding)
    
    # Y torque
    axs[1,1].set_title(f'Y torque')
    axs[1,1].set_xlabel('time (s)')
    axs[1,1].set_ylabel('Torque [Nm]', color=colors[1])
    axs[1,1].plot(t_adjusted, y_torque[start_sample:end_sample], label='Y torque', color=colors[1])
    axs[1,1].tick_params(axis='y', labelcolor=colors[1])
    axs[1,1].set_ylim(min_torque - torque_padding, max_torque + torque_padding)
    
    # Z torque
    axs[2,1].set_title(f'Z torque')
    axs[2,1].set_xlabel('time (s)')
    axs[2,1].set_ylabel('Torque [Nm]', color=colors[2])
    axs[2,1].plot(t_adjusted, z_torque[start_sample:end_sample], label='Z torque', color=colors[2])
    axs[2,1].tick_params(axis='y', labelcolor=colors[2])
    axs[2,1].set_ylim(min_torque - torque_padding, max_torque + torque_padding)
    
    fig.tight_layout()
    plt.savefig(f'{fig_path}force_torque_{i}.png')
    