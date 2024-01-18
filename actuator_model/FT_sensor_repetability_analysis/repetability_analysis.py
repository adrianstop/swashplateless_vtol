import matplotlib.pyplot as plt
import numpy as np
import csv
import os
import scipy.stats as stats

def moving_average(input_arr, window_size):
    return np.convolve(input_arr, np.ones(window_size), 'valid') / window_size

# PARAMETERS
sampling_rate = 1000/16
force_threshold = 3 # N
padding_time = 1 # seconds
test_duration = 106 # seconds (TODO auto detect end of test)
average_samples = 20 # number of samples to average over

path = os.path.abspath(os.pardir) + "\log_files\\30OCT23\\"
fig_path = path + 'figures\\'
# Check and create output directory if it doesn't exist
if not os.path.exists(fig_path):
    os.makedirs(fig_path)
    
dirlist = os.listdir(path)
dirlist = [i for i in dirlist if ( i[-14:] != '_formatted.csv' and i != 'figures' ) ]
print (dirlist)

# data format: ["Force X [N]", "Force Y [N]", "Force Z [N]", "Torque X [N-m]", "Torque Y (N-m)", "Torque Z (N-m)"]
all_data_mavg = []
start_samples = []
end_samples = []

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
    
    x_force_mavg = moving_average(data[:num_rows, 0], average_samples)
    y_force_mavg = moving_average(data[:num_rows, 1], average_samples)
    z_force_mavg = moving_average(data[:num_rows, 2], average_samples)
    force_magnitude_mavg = np.sqrt(x_force_mavg**2 + y_force_mavg**2 + z_force_mavg**2)
    max_force = np.max(np.max([x_force_mavg, y_force_mavg, z_force_mavg], axis=0))
    min_force = np.min(np.min([x_force_mavg, y_force_mavg, z_force_mavg], axis=0))
    
    x_torque_mavg = moving_average(data[:num_rows, 3], average_samples)
    y_torque_mavg = moving_average(data[:num_rows, 4], average_samples)
    z_torque_mavg = moving_average(data[:num_rows, 5], average_samples)
    torque_magnitude_mavg = np.sqrt(x_torque_mavg**2 + y_torque_mavg**2 + z_torque_mavg**2)
    max_torque = np.max([np.max(x_torque_mavg), np.max(y_torque_mavg), np.max(z_torque_mavg)])
    min_torque = np.min([np.min(x_torque_mavg), np.min(y_torque_mavg), np.min(z_torque_mavg)])
    
    start_sample = np.argmax(force_magnitude_mavg > force_threshold) - int(padding_time*sampling_rate)
    if start_sample < average_samples:
        start_sample = average_samples
    
    end_sample = start_sample + int(test_duration*sampling_rate) + int(padding_time*sampling_rate)
    if end_sample > len(force_magnitude_mavg):
        end_sample = len(force_magnitude_mavg)
    all_data_mavg.append(np.vstack((x_force_mavg[start_sample:end_sample],
                                  y_force_mavg[start_sample:end_sample],
                                  z_force_mavg[start_sample:end_sample],
                                  force_magnitude_mavg[start_sample:end_sample],
                                  x_torque_mavg[start_sample:end_sample],
                                  y_torque_mavg[start_sample:end_sample],
                                  z_torque_mavg[start_sample:end_sample],
                                  torque_magnitude_mavg[start_sample:end_sample])))



# Calculate the average and standard deviation for each test
avg = np.mean(all_data_mavg, axis=0)
std_dev = np.std(all_data_mavg, axis=0)
max_std_dev = np.max(std_dev, axis=1)
min_std_dev = np.min(std_dev, axis=1)
avg_std_dev = np.mean(std_dev, axis=1)
upper_percentile = np.percentile(std_dev, 95, axis=1)
print(f"Max standard deviation: {max_std_dev}")
print(f"Min standard deviation: {min_std_dev}")
print(f"Average standard deviation: {avg_std_dev}")
print(f"Upper percentile: {upper_percentile}")
 

cols, rows = 2, 4
fig, axs = plt.subplots(rows, cols, figsize=(9, 9))

# Plot the data for each test
labels = [['Force X', 'Torque X'], ['Force Y', 'Torque Y'], ['Force Z', 'Torque Z'], ['Force Magnitude', 'Torque Magnitude']]
colors = [['tab:red', 'tab:red'], ['tab:green', 'tab:green'], ['tab:blue', 'tab:blue'], ['tab:purple', 'tab:purple']]
background_color = [['tab:orange', 'tab:orange'], ['lightgreen', 'lightgreen'], ['lightblue', 'lightblue'], ['plum', 'plum']]

for i in range(cols):
    for j in range(rows):    
        t = np.linspace(0, len(avg[i*rows+j])/sampling_rate, len(avg[i*rows+j]))
        axs[j, i].plot(t, avg[i*rows+j], label='average', color=colors[j][i])
        axs[j, i].fill_between(t, avg[i*rows+j] - std_dev[i*rows+j], avg[i*rows+j] + std_dev[i*rows+j], color=background_color[j][i], alpha=0.8)
        axs[j, i].set_title(labels[j][i])
        axs[j,i].grid(True, linestyle='--', color='gray', alpha=0.7)
        axs[j, i].set_xlabel('s', labelpad=-11, x=1.0)
        if i == 0:
            axs[j, i].set_ylabel('N')
        else:
            axs[j, i].set_ylabel('Nm')
# Show the plot
plt.tight_layout()
plt.savefig(os.path.join(fig_path, 'repetability.eps'))
plt.savefig(os.path.join(fig_path, 'repetability.png'))
plt.close()