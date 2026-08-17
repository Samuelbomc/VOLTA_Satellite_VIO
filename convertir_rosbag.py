"""Convert ROS 2 bag into a EuRoC-style dataset for Basalt.

Only the configured camera and IMU topics are exported; all other topics are ignored.
"""

import os
import cv2
import numpy as np
import csv
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore

# ================= CONFIGURATION =================
BAG_PATH = 'dataset_caida'         
OUTPUT_FOLDER = 'mav0'             
IMAGE_TOPIC = '/front_cam/image_raw'
IMU_TOPIC = '/imu/data'
# =================================================

# Create output folders.
os.makedirs(f'{OUTPUT_FOLDER}/cam0/data', exist_ok=True)
os.makedirs(f'{OUTPUT_FOLDER}/imu0', exist_ok=True)

imu_data = []
cam_data = []

print(f"Opening ROSBAG: '{BAG_PATH}'...")

typestore = get_typestore(Stores.ROS2_HUMBLE)

with Reader(BAG_PATH) as reader:
    for connection, timestamp, rawdata in reader.messages():
        msg = typestore.deserialize_cdr(rawdata, connection.msgtype)

        # ================= CAMERA =================
        if connection.topic == IMAGE_TOPIC:
            try:
                width = msg.width
                height = msg.height
                img_array = np.frombuffer(msg.data, dtype=np.uint8)

                # Validate image size.
                if msg.encoding in ['rgb8', 'bgr8']:
                    if img_array.size != width * height * 3:
                        print("Corrupt RGB image, skipping")
                        continue
                    img = img_array.reshape((height, width, 3))

                    if msg.encoding == 'rgb8':
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                    # Convert to grayscale.
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                elif msg.encoding in ['mono8', '8UC1']:
                    if img_array.size != width * height:
                        print("Corrupt mono image, skipping")
                        continue
                    img = img_array.reshape((height, width))

                else:
                    print(f"Unsupported encoding: {msg.encoding}")
                    continue

                # Timestamp in nanoseconds.
                ts_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec

                filename = f"{ts_ns}.png"
                filepath = os.path.join(OUTPUT_FOLDER, 'cam0/data', filename)

                # Optimized save.
                cv2.imwrite(filepath, img, [cv2.IMWRITE_PNG_COMPRESSION, 3])

                cam_data.append([ts_ns, filename])

            except Exception as e:
                print(f"Error processing image: {e}")
                continue

        # ================= IMU =================
        elif connection.topic == IMU_TOPIC:
            try:
                ts_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec

                # Keep the EuRoC/Basalt ordering: angular velocity first, then linear acceleration.
                imu_data.append([
                    ts_ns,
                    msg.angular_velocity.x,
                    msg.angular_velocity.y,
                    msg.angular_velocity.z,
                    msg.linear_acceleration.x,
                    msg.linear_acceleration.y,
                    msg.linear_acceleration.z
                ])
            except Exception as e:
                print(f"Error processing IMU: {e}")
                continue

# ================= SORT DATA =================
print("Sorting data by timestamp...")
cam_data.sort(key=lambda x: x[0])
imu_data.sort(key=lambda x: x[0])

# ================= WRITE CSV =================
print("Writing camera data.csv...")
with open(f'{OUTPUT_FOLDER}/cam0/data.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['#timestamp [ns]', 'filename'])
    writer.writerows(cam_data)

print("Writing IMU data.csv...")
with open(f'{OUTPUT_FOLDER}/imu0/data.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        '#timestamp [ns]',
        'w_RS_S_x [rad s^-1]', 'w_RS_S_y [rad s^-1]', 'w_RS_S_z [rad s^-1]',
        'a_RS_S_x [m s^-2]', 'a_RS_S_y [m s^-2]', 'a_RS_S_z [m s^-2]'
    ])
    writer.writerows(imu_data)

print(f"\n✅ Conversion successful!")
print(f"Dataset ready at: '{OUTPUT_FOLDER}'")
