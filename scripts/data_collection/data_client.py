# ****************************************************************************
# *  Script to receive camera and radar data over socket from sensor module.

import socket
import cv2
import numpy as np
import struct
import threading
import csv
from datetime import datetime
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import queue
from matplotlib.animation import FuncAnimation

# === Configuration ===
# SERVER_IP = '134.34.226.239' # External WiFi
SERVER_IP = '192.168.25.1' # Pi WiFi AP
VIDEO_PORT = 9999
RADAR_PORT = 10000

# === Utility Functions ===

def recv_image(conn):
    raw_len = conn.recv(4)
    if not raw_len:
        return None
    img_len = struct.unpack('>L', raw_len)[0]

    img_data = b''
    while len(img_data) < img_len:
        packet = conn.recv(img_len - len(img_data))
        if not packet:
            return None
        img_data += packet

    img_array = np.frombuffer(img_data, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return img

def recv_exact(sock, size):
    data = b''
    while len(data) < size:
        packet = sock.recv(size - len(data))
        if not packet:
            print(f"[Radar] Socket closed. Needed {size}, got {len(data)}.")
            return None
        data += packet
    return data

def recv_array(sock):
    meta = recv_exact(sock, 8)

    if not meta:
        print("[Radar] Failed to read header")
        return None
    rows, cols = struct.unpack('>II', meta)

    # Sanity check
    if rows <= 0 or cols <= 0 or rows > 1000 or cols > 1000:
        print(f"[Radar] Invalid shape received: ({rows}, {cols})")
        return None

    dtype = np.float32
    total_bytes = rows * cols * dtype().nbytes

    array_data = recv_exact(sock, total_bytes)
    if array_data is None:
        print("[Radar] Failed to receive full array")
        return None

    try:
        array = np.frombuffer(array_data, dtype=dtype).reshape((rows, cols))
    except ValueError as e:
        print(f"[Radar] Reshape error: {e}")
        return None

    return array

# === Threaded Receivers ===

def video_receiver():
    try:
        video_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        video_sock.connect((SERVER_IP, VIDEO_PORT))
        print("[Video] Connected")

        timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        fisheye_path = os.path.join('data', f"fisheye_{timestamp_str}.mp4")
        thermal_path = os.path.join('data', f"thermal_{timestamp_str}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        fisheye_writer = cv2.VideoWriter(fisheye_path, fourcc, 3, (int(2592/3), int(1944/3)))
        thermal_writer = cv2.VideoWriter(thermal_path, fourcc, 3, (160, 120))

        while not stop_event.is_set():
            fisheye = recv_image(video_sock)
            thermal = recv_image(video_sock)

            if fisheye is None or thermal is None:
                break

            fisheye_writer.write(fisheye)
            thermal_writer.write(thermal)

            cv2.imshow('Fisheye', fisheye)
            scale_factor = 4
            resized_thermal = cv2.resize(thermal, (thermal.shape[1] * scale_factor, thermal.shape[0] * scale_factor), interpolation=cv2.INTER_NEAREST)
            cv2.imshow('Thermal', resized_thermal)

            if cv2.waitKey(1) == ord('q'):
                stop_event.set()
                break

        video_sock.close()
    except Exception as e:
        print(f"[Video] Error: {e}")
        stop_event.set()
    finally:
        cv2.destroyAllWindows()
        if fisheye_writer:
            fisheye_writer.release()
        if thermal_writer:
            thermal_writer.release()

radar_queue = queue.Queue()

def radar_receiver():
    try:
        radar_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        radar_sock.connect((SERVER_IP, RADAR_PORT))
        print("[Radar] Connected")

        # --- Setup CSV File ---
        timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        mmwave_path = os.path.join('data', f"mmwave_{timestamp_str}.csv")

        if not os.path.exists(mmwave_path):
            with open(mmwave_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Date', 'Time', 'X', 'Y', 'Z'])

        while not stop_event.is_set():
            radar_data = recv_array(radar_sock)

            if radar_data is None:
                print("No radar data")
                continue

            points = radar_data.T

            with open(mmwave_path, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)

                now = datetime.now()
                date_str = now.strftime("%Y-%m-%d")
                time_str = now.strftime("%H:%M:%S.%f")[:-5] # to one-tenth of a second

                for point in points:
                    row = [date_str, time_str] + point.tolist()
                    writer.writerow(row)

            # --- Real-time Plot Update ---
            radar_queue.put(points)

        radar_sock.close()
    except Exception as e:
        print(f"[Radar] Error: {e}")
        stop_event.set()

# === Main Execution ===

stop_event = threading.Event()

fig, ax = plt.subplots()
scatter = ax.scatter([], [], c=[], cmap='viridis')
ax.set_xlabel('X')
ax.set_ylabel('Z')
ax.set_xlim([-2, 1.5])
ax.set_ylim([-1, 0.8])
ax.set_title('Live Radar Point Cloud (X-Z, color=Y)')

norm = Normalize(vmin=0, vmax=1)
cbar = fig.colorbar(scatter, ax=ax)
cbar.set_label('Y')

def update(frame):
    while not radar_queue.empty():
        points = radar_queue.get()
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        scatter.set_offsets(np.column_stack((x, z)))
        scatter.set_array(y)
        scatter.set_clim(vmin=0, vmax=3)
    return scatter,

ani = FuncAnimation(fig, update, interval=200)

video_thread = threading.Thread(target=video_receiver)
radar_thread = threading.Thread(target=radar_receiver)

video_thread.start()
radar_thread.start()

try:
    while not stop_event.is_set():
        # Keep main thread alive until stop_event is triggered
        video_thread.join(timeout=1)
        radar_thread.join(timeout=1)
        plt.show()
except KeyboardInterrupt:
    print("Stopping...")
    stop_event.set()

# video_thread.join()
# radar_thread.join()