# ****************************************************************************
# *  Script to send camera and radar data over socket from Raspberry Pi in sensor module.

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

IMAGE_PORT = 9999
RADAR_PORT = 10000

stop_event = threading.Event()

# === Functions for data serialization ===

def send_image(sock, frame):
    _, img_encoded = cv2.imencode('.jpg', frame)
    data = img_encoded.tobytes()
    sock.sendall(struct.pack('>L', len(data)))
    sock.sendall(data)

def send_array(sock, array):
    array_data = array.astype(np.float32)
    shape = array.shape
    sock.sendall(struct.pack('>II', shape[0], shape[1]))
    sock.sendall(array_data)

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
    while MAGIC_WORD not in sync and not stop_event.is_set():
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

# === Threaded socket handlers ===

def image_stream_thread(cam1, cam2, conn):
    try:
        while not stop_event.is_set():
            loop_start = time.time()

            fisheye_frame = cam1.capture_array()
            ret_thermal, thermal_frame = cam2.read()
            if not ret_thermal:
                print("Thermal cam error")
                break

            thermal_display = cv2.rotate(thermal_frame, cv2.ROTATE_180)

            send_image(conn, fisheye_frame)
            send_image(conn, thermal_display)

            elapsed = time.time() - loop_start
            time.sleep(max(0, 1.0/3 - elapsed))  # target ~3 FPS

    except Exception as e:
        print(f"[Image Thread] Error: {e}")
    finally:
        conn.close()

def radar_stream_thread(ser, conn):
    try:
        while not stop_event.is_set():
            loop_start = time.time()

            frame = read_frame(ser)
            if not frame:
                continue
            payload, num_tlvs, num_det_obj = frame
            if num_det_obj == 0 or num_det_obj > 100:
                continue
            x, y, z = parse_tlvs(payload, num_tlvs, num_det_obj)
            radar_data = np.array([x, y, z])

            send_array(conn, radar_data)

            elapsed = time.time() - loop_start
            time.sleep(max(0, 1.0/3 - elapsed))  # target ~3 FPS

    except Exception as e:
        print(f"[Radar Thread] Error: {e}")
    finally:
        conn.close()

# === Main Program ===

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

img_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
radar_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

img_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
radar_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

img_server.bind(('0.0.0.0', IMAGE_PORT))
radar_server.bind(('0.0.0.0', RADAR_PORT))

img_server.listen(1)
radar_server.listen(1)

while not stop_event.is_set():
    try:
        print("Waiting for image and radar connections...")

        img_conn, img_addr = img_server.accept()
        print(f"Image connection from {img_addr}")

        radar_conn, radar_addr = radar_server.accept()
        print(f"Radar connection from {radar_addr}")

        img_thread = threading.Thread(target=image_stream_thread, args=(fisheye_cam, thermal_cam, img_conn))
        radar_thread = threading.Thread(target=radar_stream_thread, args=(ser, radar_conn))

        img_thread.start()
        radar_thread.start()

        img_thread.join()
        radar_thread.join()

    except Exception as e:
        print(f"[Main Loop] Error: {e}")
        time.sleep(5)
        continue