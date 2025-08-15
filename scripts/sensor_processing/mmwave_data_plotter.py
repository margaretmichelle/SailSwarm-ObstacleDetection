# ****************************************************************************
# *  Script to plot mmWave data from a CSV file as a 2D scatter plot with depth/range represented through a colourbar.
# *  CSV file columns are expected to be Date, Time, X, Y, Z

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# === Load and preprocess CSV ===

CSV_PATH = ""

df = pd.read_csv(CSV_PATH)

# Combine Date and Time into datetime objects
df['Timestamp'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
df['RoundedTime'] = df['Timestamp'].apply(lambda x: x.strftime("%H:%M:%S.%f")[:-5])

# Group points by rounded timestamp
grouped = df.groupby('RoundedTime')
timestamps = sorted(grouped.groups.keys())
current_index = [0]

# === Determine axis and color limits ===
# x_min, x_max = df['X'].min() - 0.5, df['X'].max() + 0.5
x_min, x_max = -5,5 #(df['X'].max() * -1.0) - 0.2, df['X'].max() + 0.2
z_min, z_max = -5,5 #df['Z'].min() - 0.2, df['Z'].max() + 0.2
y_min, y_max = 0, 16 #df['Y'].min(), df['Y'].max()

# === Setup plot ===
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.2)

# Initialize scatter and colorbar
initial_group = grouped.get_group(timestamps[current_index[0]])
filtered_group = initial_group[(initial_group['Y'] >= 0.1) & (initial_group['Y'] <= 5)]
current_filtered_data = [filtered_group]
sc = ax.scatter(filtered_group['X'], filtered_group['Z'], c=filtered_group['Y'], cmap='turbo', s=50, vmin=y_min, vmax=y_max, picker=True)
ax.set_xlim(x_min, x_max)
ax.set_ylim(z_min, z_max)
ax.set_xlabel("X")
ax.set_ylabel("Z")
ax.set_title(f"Timestamp: {timestamps[current_index[0]]} Frame: {current_index}")
cb = fig.colorbar(sc, ax=ax)
cb.set_label("Y")

# === Plot update function ===
def plot_group(index):
    group = grouped.get_group(timestamps[index])
    filtered_group = group[(group['Y'] >= 0.1) & (group['Y'] <= 5)]
    current_filtered_data[0] = filtered_group
    sc.set_offsets(filtered_group[['X', 'Z']].values)
    sc.set_array(filtered_group['Y'].values)
    ax.set_title(f"Timestamp: {timestamps[index]} Frame: {current_index}")
    plt.draw()

def plot_radar_angles(index):
    """Live update radar angle line plot."""
    group = grouped.get_group(timestamps[index])
    filtered_group = group[(group['Y'] >= 0.1) & (group['Y'] <= 5)]
    current_filtered_data[0] = filtered_group
    angles = np.degrees(np.arctan2(group['X'], group['Y']))
    angles = np.clip(angles, -60, 60)
    plt.clf()
    plt.scatter(angles, np.zeros_like(angles), marker='|', color='red')
    plt.xlim(-60, 60)
    plt.ylim(-1, 1)
    plt.yticks([])
    plt.xlabel("Angle (°)")
    plt.title("Radar Detection Angles")
    plt.grid(True, axis='x', linestyle='--', alpha=0.5)
    plt.pause(0.001)

# === Keyboard interaction ===
def on_key(event):
    if event.key == 'd' and current_index[0] < len(timestamps) - 1:
        current_index[0] += 1
        plot_group(current_index[0])
    elif event.key == 'a' and current_index[0] > 0:
        current_index[0] -= 1
        plot_group(current_index[0])

def on_pick(event):
    ind = event.ind[0] # index of picked point
    data = current_filtered_data[0].iloc[ind]
    x, y, z = data['X'], data['Y'], data['Z']
    print(f"Clicked point -> X: {x:.4f}, Y: {y:.4f}, Z: {z:.4f}")


fig.canvas.mpl_connect('key_press_event', on_key)
fig.canvas.mpl_connect('pick_event', on_pick)
plt.show()