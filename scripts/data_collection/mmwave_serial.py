# ****************************************************************************
# *  Script to read data directly from AWR1843BOOST evaluation board.
# *  Includes functions from TI's demo mmWave parsing script.

import serial
import struct
import time
import binascii
import codecs
import numpy as np

from matplotlib.animation import FuncAnimation

CLI_PORT = 'COM10'
DATA_PORT = 'COM9'
CLI_BAUD = 115200
DATA_BAUD = 921600
CONFIG_FILE = 'test_config.cfg'
LOG_TO_CSV = True
CSV_FILE = 'point_cloud.csv'

MAGIC_WORD = b'\x02\x01\x04\x03\x06\x05\x08\x07'

def getUint32(data):
    """!
       This function coverts 4 bytes to a 32-bit unsigned integer.

        @param data : 1-demension byte array  
        @return     : 32-bit unsigned integer
    """ 
    return (data[0] +
            data[1]*256 +
            data[2]*65536 +
            data[3]*16777216)

def getUint16(data):
    """!
       This function coverts 2 bytes to a 16-bit unsigned integer.

        @param data : 1-demension byte array
        @return     : 16-bit unsigned integer
    """ 
    return (data[0] +
            data[1]*256)

def getHex(data):
    """!
       This function coverts 4 bytes to a 32-bit unsigned integer in hex.

        @param data : 1-demension byte array
        @return     : 32-bit unsigned integer in hex
    """ 
    return (binascii.hexlify(data[::-1]))


def send_config(cli_port, config_path):
    with serial.Serial(cli_port, CLI_BAUD, timeout=1) as cli:
        with open(config_path, 'r') as f:
            for line in f:
                if line.strip() and not line.startswith('%'):
                    cli.write((line.strip() + '\n').encode())
                    time.sleep(0.05)

def read_frame(ser):
    '''
    Read single frame from mmWwave board.

    Inputs:
        ser: Serial connection to board.

    Returns:
        payload: Data read from serial.
        num_tlvs: Number of TLVs in the data.
        num_det_obj: Number of detected objects by mmWave sensor.
    '''
    sync = b''
    while MAGIC_WORD not in sync:
        sync += ser.read(1)
        sync = sync[-8:]

    # Read the next 32 bytes for the rest of the frame header
    header = ser.read(32)
    if len(header) < 32:
        return None

    packet_len = getUint32(header[4:8])
    platform            = getHex(header[8:12])
    frame_num         = getUint32(header[12:16])
    time_cpu_cyc       = getUint32(header[16:20])
    num_det_obj           = getUint32(header[20:24])
    num_tlvs              = getUint32(header[24:28])
    sub_frame_num      = getUint32(header[28:32])

    payload = ser.read(packet_len - 40)
    if len(payload) < packet_len - 40:
        return None

    print("Time: %d" % (time_cpu_cyc))
    print("Detected objects: %d" % (num_det_obj))
    return payload, num_tlvs, num_det_obj

def parse_tlvs(data, num_tlvs, num_det_obj):
    '''
    Parse individual Type Length Value structures.
    https://dev.ti.com/tirex/explore/content/radar_toolbox_2_30_00_12/software_docs/Understanding_UART_Data_Output_Format.html

    Inputs:
        data: Data read from serial.
        num_tlvs: Number of TLVs in the data.
        num_det_obj: Number of detected objects by mmWave sensor.

    Returns:
        x: X coordinates of detected objects.
        y: Y coordinates of detected objects.
        z: Z coordinates of detected objects.
        v: Velocities of detected objects.
    '''
    offset = 0
    x = []
    y = []
    z = []
    v = []

    for _ in range(num_tlvs):
        if offset + 8 > len(data):
            break
        tlv_type = getUint32(data[offset:offset+4])
        tlv_length = getUint32(data[offset+4:offset+8]) # does not include the length of the TLV header
        offset += 8

        if tlv_type == 1:  # Point cloud TLV
            tlv_offset = offset
            for obj in range(num_det_obj):
                x.append(struct.unpack('<f', codecs.decode(binascii.hexlify(data[tlv_offset:tlv_offset+4:1]),'hex'))[0])

                y.append(struct.unpack('<f', codecs.decode(binascii.hexlify(data[tlv_offset+4:tlv_offset+8:1]),'hex'))[0])

                z.append(struct.unpack('<f', codecs.decode(binascii.hexlify(data[tlv_offset+8:tlv_offset+12:1]),'hex'))[0])

                v.append(struct.unpack('<f', codecs.decode(binascii.hexlify(data[tlv_offset+12:tlv_offset+16:1]),'hex'))[0])
                tlv_offset += 16
            print(x,y,z)
        offset += tlv_length

    return x,y,z,v

def live_colored_2d_plot(ser, color_by='y'):
    """
    Live 2D plot from mmWave radar data.
    color_by: 'y' or 'v' — third dimension mapped to color.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    scatter = ax.scatter([], [], c=[], cmap='plasma', s=30, edgecolors='k')

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Z (meters)')
    ax.set_title(f"Live 2D Point Cloud (color = {color_by.upper()})")

    cbar = plt.colorbar(scatter, ax=ax, label=color_by.upper())

    def update(frame):
        frame = read_frame(ser)
        while frame is None:
            frame = read_frame(ser)

        payload, num_tlvs, num_det_obj = frame
        x, y, z, v = [], [], [], []
        try:
            x, y, z = parse_tlvs(payload, num_tlvs, num_det_obj)
        except Exception as e:
            print("Parse error:", e)
            return scatter,

        if len(x) == 0:
            return scatter,

        color_val = {'y': y, 'v': v}.get(color_by, y)
        scatter.set_offsets(list(zip(x, z)))
        scatter.set_array(np.array(color_val))
        scatter.set_clim(np.min(color_val), np.max(color_val))
        cbar.update_normal(scatter)

        ax.relim()
        ax.autoscale_view()

        return scatter,

    ani = FuncAnimation(fig, update, interval=100)
    plt.tight_layout()
    plt.show()

def main():
    send_config(CLI_PORT, CONFIG_FILE)
    print(f"Config sent to {CLI_PORT}, listening for point cloud on {DATA_PORT}...")

    ser = serial.Serial(DATA_PORT, DATA_BAUD, timeout=1)

    try:
        live_colored_2d_plot(ser, color_by='y')

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        ser.close()

if __name__ == '__main__':
    main()
