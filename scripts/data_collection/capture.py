# ****************************************************************************
# *  Script to collect camera and radar data as images and csv file entries from sensor module.
# *  Pressing x will kill the script, any other keyboard input will capture data.

import cv2
import struct
import time
import numpy as np
import serial
import binascii
from picamera2 import Picamera2
import os
from datetime import datetime
import csv
import sys
import tty
import termios

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

def wait_for_key(prompt="Press x to exit. Press any other key to capture..."):
    print(prompt)
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        if ch == 'x':
            return False
        else:
            return True
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

# === Main Program ===

os.makedirs('data/captures', exist_ok=True)

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

print("Connected")

timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M")
mmwave_path = os.path.join('data/captures', f"mmwave_{timestamp_str}.csv")

while True:
    if wait_for_key():
        now = datetime.now()
        timestamp_str = now.strftime("%Y-%m-%d_%H-%M-%S")
        fisheye_path = os.path.join('data/captures', f"fisheye_{timestamp_str}.jpg")
        thermal_path = os.path.join('data/captures', f"thermal_{timestamp_str}.jpg")

        if not os.path.exists(mmwave_path):
            with open(mmwave_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Date', 'Time', 'X', 'Y', 'Z'])

        print("Collecting data...")
        fisheye_frame = fisheye_cam.capture_array()
        ret_thermal, thermal_frame = thermal_cam.read()
        if not ret_thermal:
            print("Thermal cam error")
            break
        thermal_frame = cv2.rotate(thermal_frame, cv2.ROTATE_180)

        frame = read_frame(ser)
        if not frame:
            print("mmWave frame error")
        payload, num_tlvs, num_det_obj = frame
        if num_det_obj == 0 or num_det_obj > 100:
            print("Too many or few mmWave objects")
        else:
            x, y, z = parse_tlvs(payload, num_tlvs, num_det_obj)
            radar_data = np.array([x, y, z])

            cv2.imwrite(fisheye_path, fisheye_frame)
            cv2.imwrite(thermal_path, thermal_frame)

            with open(mmwave_path, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)

                date_str = now.strftime("%Y-%m-%d")
                time_str = now.strftime("%H:%M:%S.%f")[:-5] # to one-tenth of a second

                for point in radar_data.T:
                    row = [date_str, time_str] + point.tolist()
                    writer.writerow(row)
    else:
        thermal_cam.release()
        fisheye_cam.stop()
        ser.close()
        break

thermal_cam.release()
fisheye_cam.stop()
ser.close()