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
test_duration = 9# seconds (TODO auto detect end of test)
average_samples = 10 # number of samples to average over

path = os.path.abspath(os.pardir) + "\log_files\\inverted_vs_normal_comparison\\"
fig_path = path + 'figures\\'
# Check and create output directory if it doesn't exist
if not os.path.exists(fig_path):
    os.makedirs(fig_path)
    
dirlist = os.listdir(path)
dirlist = [i for i in dirlist if ( i[-14:] != '_formatted.csv' and i != 'figures' ) ]
print (dirlist)

all_data_mavg = []
t_adj = []

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
    
    
    x_torque = data[:num_rows, 3]
    y_torque = data[:num_rows, 4]
    z_torque = data[:num_rows, 5]
    x_torque_mavg = moving_average(x_torque, average_samples)
    y_torque_mavg = moving_average(y_torque, average_samples)
    z_torque_mavg = moving_average(z_torque, average_samples)
    torque_magnitude = np.sqrt(x_torque**2 + y_torque**2 + z_torque**2)
    torque_magnitude_mavg = moving_average(torque_magnitude, average_samples)
    
    start_sample = np.argmax(force_magnitude_mavg > force_threshold) - int(padding_time*sampling_rate)
    if start_sample < average_samples:
        start_sample = average_samples
    
    end_sample = start_sample + int(test_duration*sampling_rate) + int(padding_time*sampling_rate)
    if end_sample > len(force_magnitude_mavg):
        end_sample = len(force_magnitude_mavg)
    time_duration = (end_sample - start_sample)/sampling_rate
    t_adjusted = np.linspace(0, time_duration, end_sample - start_sample)
    t_adjusted_mavg = t_adjusted[average_samples:]
    
    t_adj.append(t_adjusted)
    
    all_data_mavg.append([x_force_mavg[start_sample:end_sample],
                          y_force_mavg[start_sample:end_sample],
                          z_force_mavg[start_sample:end_sample],
                          force_magnitude_mavg[start_sample:end_sample],
                          x_torque_mavg[start_sample:end_sample],
                          y_torque_mavg[start_sample:end_sample],
                          z_torque_mavg[start_sample:end_sample],
                          torque_magnitude_mavg[start_sample:end_sample]])

all_data_mavg = np.array(all_data_mavg)

max_force = np.max(all_data_mavg[:, 3, :])
min_force =  np.min(all_data_mavg[:, 3, :])
max_torque =  np.max(all_data_mavg[:, 7, :])
min_torque =  np.min(all_data_mavg[:, 7, :])


avg_offset = int(sampling_rate*2)
avg_duration = int(sampling_rate*2)
# Calculate the difference in force magnitude between the two orientations
inverted_force_avg = np.average(all_data_mavg[0, 3, avg_offset:avg_offset+avg_duration])
normal_force_avg = np.average(all_data_mavg[1, 3, avg_offset:avg_offset+avg_duration])
force_diff = np.abs(all_data_mavg[0, 3, avg_offset:avg_offset+avg_duration] - all_data_mavg[1, 3, avg_offset:avg_offset+avg_duration])
force_diff_avg = np.average(force_diff)
force_diff_percent = force_diff_avg / inverted_force_avg * 100
print(f"force_diff_avg : {force_diff_avg}, which is {force_diff_percent}% of the inverted force average")

# Calculate the difference in torque magnitude between the two orientations
torque_diff = np.abs(all_data_mavg[0, 7, avg_offset:avg_offset+avg_duration] - all_data_mavg[1, 7, avg_offset:avg_offset+avg_duration])
torque_diff_avg = np.average(torque_diff)
print(f"torque_diff_avg : {torque_diff_avg}")

# Plot raw force data
fig, axs_raw = plt.subplots(3,2, figsize=(7, 8))
fig.suptitle(f'FT comparison inverted vs normal rotor orientation')
colors = ['tab:red', 'tab:blue', 'tab:green']
force_padding = 0.05*(max_force - min_force)
torque_padding = 0.05*(max_torque - min_torque)
colors = [['tab:red', 'tab:orange'], ['tab:green','lightgreen'],['tab:blue', 'lightblue']]
legend_labels = ['Inverted', 'Normal']

# Define the titles, y-labels, and data for the subplots
for i in range(len(all_data_mavg)):
    subplots = [
        {"title": "X force", "ylabel": "Force [N]", "data": np.abs(all_data_mavg[i, 0])},
        {"title": "X torque", "ylabel": "Torque [Nm]", "data": np.abs(all_data_mavg[i, 4])},
        {"title": "Y force", "ylabel": "Force [N]", "data": np.abs(all_data_mavg[i, 1])},
        {"title": "Y torque", "ylabel": "Torque [Nm]", "data": np.abs(all_data_mavg[i, 5])},
        {"title": "Z force", "ylabel": "Force [N]", "data": np.abs(all_data_mavg[i, 3])},
        {"title": "Z torque", "ylabel": "Torque [Nm]", "data": np.abs(all_data_mavg[i, 6])}
    ]

    # For each subplot
    for k, subplot in enumerate(subplots):
        row = k // 2
        col = k % 2
        axs_raw[row, col].set_title(subplot["title"])
        axs_raw[row, col].set_xlabel('time (s)')
        axs_raw[row, col].set_ylabel(subplot["ylabel"])
        axs_raw[row, col].plot(t_adj[i], subplot["data"], label=subplot["title"].lower(), color=colors[row][i])
        axs_raw[row, col].tick_params(axis='y')
        axs_raw[row, col].grid(True, linestyle='--', color='gray', alpha=0.7)
        if k != 4:
            axs_raw[row,col].legend(legend_labels,loc = 'upper right')
        else:
            axs_raw[row,col].legend(legend_labels,loc = 'lower right')
        if col == 0:
            axs_raw[row, col].set_ylim(min_force - force_padding, max_force + force_padding)
        else:
            axs_raw[row, col].set_ylim(min_torque - torque_padding, max_torque + torque_padding)

fig.tight_layout()
plt.savefig(f'{fig_path}force_torque_{i}.png')
plt.close()