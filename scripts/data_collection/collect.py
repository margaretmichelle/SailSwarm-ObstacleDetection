# ****************************************************************************
# *  Script to collect camera and radar data as videos and csv file entries from sensor module.
# *  Will run for 2 minutes if no keyboard interrupt stops the script.

import socket
import cv2
import struct
import time
import numpy as np
import serial
import binascii
import codecs
import threading
from picamera2 import Picamera2
import os
from datetime import datetime
import csv

# === Constants ===
CLI_PORT = '/dev/ttyACM0'
DATA_PORT = '/dev/ttyACM1'
CLI_BAUD = 115200
DATA_BAUD = 921600
CONFIG_FILE = '/home/onyxpearl/test_config.cfg'
MAGIC_WORD = b'\x02\x01\x04\x03\x06\x05\x08\x07'

# === Radar data helpers ===

def getUint32(data):
    return data[0] + data[1]*256 + data[2]*65536 + data[3]*16777216

def getUint16(data):
    return data[0] + data[1]*256

def getHex(data):
    return binascii.hexlify(data[::-1])

def send_config(cli_port, config_path):
    with serial.Serial(cli_port, CLI_BAUD, timeout=1) as cli:
        with open(config_path, 'r') as f:
            for line in f:
                if line.strip() and not line.startswith('%'):
                    cli.write((line.strip() + '\n').encode())
                    time.sleep(0.05)

def read_frame(ser):
    sync = b''
    while MAGIC_WORD not in sync:
        sync += ser.read(1)
        sync = sync[-8:]

    header = ser.read(32)
    if len(header) < 32:
        return None

    packet_len = getUint32(header[4:8])
    num_det_obj = getUint32(header[20:24])
    num_tlvs = getUint32(header[24:28])

    payload = ser.read(packet_len - 40)
    if len(payload) < packet_len - 40:
        return None

    return payload, num_tlvs, num_det_obj

def parse_tlvs(data, num_tlvs, num_det_obj):
    offset = 0
    x, y, z = [], [], []

    for _ in range(num_tlvs):
        if offset + 8 > len(data): break
        tlv_type = getUint32(data[offset:offset+4])
        tlv_length = getUint32(data[offset+4:offset+8])
        offset += 8

        if tlv_type == 1:
            tlv_offset = offset
            for _ in range(num_det_obj):
                x.append(struct.unpack('<f', data[tlv_offset:tlv_offset+4])[0])
                y.append(struct.unpack('<f', data[tlv_offset+4:tlv_offset+8])[0])
                z.append(struct.unpack('<f', data[tlv_offset+8:tlv_offset+12])[0])
                tlv_offset += 16
        offset += tlv_length
    return x, y, z

# === Main Program ===

os.makedirs('data', exist_ok=True)

print("Connecting to cameras and radar...")

send_config(CLI_PORT, CONFIG_FILE)
print("Radar config sent")

ser = serial.Serial(DATA_PORT, DATA_BAUD, timeout=1)

fisheye_cam = Picamera2()
fisheye_cam.configure(fisheye_cam.create_preview_configuration(main={"format": "RGB888", "size": (int(2592/3), int(1944/3))}))
fisheye_cam.start()

thermal_cam = None
for idx in [0,1]:
    thermal_cam = cv2.VideoCapture(idx)
    if thermal_cam.isOpened():
        break

timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
fisheye_path = os.path.join('data', f"fisheye_{timestamp_str}.mp4")
thermal_path = os.path.join('data', f"thermal_{timestamp_str}.mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

fisheye_writer = cv2.VideoWriter(fisheye_path, fourcc, 3, (int(2592/3), int(1944/3)))
thermal_writer = cv2.VideoWriter(thermal_path, fourcc, 3, (160, 120))

timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
mmwave_path = os.path.join('data', f"mmwave_{timestamp_str}.csv")

if not os.path.exists(mmwave_path):
    with open(mmwave_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Date', 'Time', 'X', 'Y', 'Z'])

print("Connected")

capture_start = time.time()

capture_length = 2 # minutes

try:
    while True:
        print("Collecting data...")
        loop_start = time.time()

        fisheye_frame = fisheye_cam.capture_array()
        ret_thermal, thermal_frame = thermal_cam.read()
        if not ret_thermal:
            print("Thermal cam error")
            break

        thermal_display = cv2.rotate(thermal_frame, cv2.ROTATE_180)

        frame = read_frame(ser)
        if not frame:
            continue
        payload, num_tlvs, num_det_obj = frame
        if num_det_obj == 0 or num_det_obj > 100:
            continue
        x, y, z = parse_tlvs(payload, num_tlvs, num_det_obj)
        radar_data = np.array([x, y, z])

        fisheye_writer.write(fisheye_frame)
        thermal_writer.write(thermal_frame)

        with open(mmwave_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)

            now = datetime.now()
            date_str = now.strftime("%Y-%m-%d")
            time_str = now.strftime("%H:%M:%S.%f")[:-5] # to one-tenth of a second

            for point in radar_data.T:
                row = [date_str, time_str] + point.tolist()
                writer.writerow(row)

        elapsed = time.time() - loop_start
        print(1.0/3 - elapsed)
        time.sleep(max(0, 1.0/3 - elapsed))  # target ~3 FPS

        if capture_start - time.time() > 60*capture_length:
            print("Capture complete")
            thermal_cam.release()
            fisheye_cam.stop()
            ser.close()
            if fisheye_writer:
                fisheye_writer.release()
            if thermal_writer:
                thermal_writer.release()
            break

except KeyboardInterrupt:
    print("Stopping...")
except Exception as e:
    print(f"Error occurred: {e}")
finally:
    thermal_cam.release()
    fisheye_cam.stop()
    ser.close()
    if fisheye_writer:
        fisheye_writer.release()
    if thermal_writer:
        thermal_writer.release()