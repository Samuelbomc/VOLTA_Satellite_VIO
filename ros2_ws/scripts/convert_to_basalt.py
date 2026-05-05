import os
import cv2
import csv
import numpy as np
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore

class BNO055NoiseModel:
    def __init__(self):
        self.prev_ts = -1
        self.bg = np.zeros(3) # Bias del giroscopio
        self.ba = np.zeros(3) # Bias del acelerómetro
        
        # Parámetros aproximados para BNO055
        self.sigma_a = 0.0015  # Ruido blanco acelerómetro
        self.sigma_g = 0.0017  # Ruido blanco giroscopio
        self.sigma_ba = 1.0e-4 # Caminata aleatoria (Random Walk) acelerómetro
        self.sigma_bg = 1.0e-5 # Caminata aleatoria (Random Walk) giroscopio

    def process(self, ts_ns, a_x, a_y, a_z, g_x, g_y, g_z):
        if self.prev_ts < 0:
            self.prev_ts = ts_ns
            dt = 0.01 
        else:
            dt = (ts_ns - self.prev_ts) * 1e-9
            self.prev_ts = ts_ns
            
        if dt <= 0: dt = 0.01

        # Actualizar el Bias usando la raíz cuadrada de dt
        self.ba += np.random.randn(3) * self.sigma_ba * np.sqrt(dt)
        self.bg += np.random.randn(3) * self.sigma_bg * np.sqrt(dt)

        # Agregar ruido blanco discretizado y el bias
        a_noisy = np.array([a_x, a_y, a_z]) + self.ba + np.random.randn(3) * self.sigma_a / np.sqrt(dt)
        g_noisy = np.array([g_x, g_y, g_z]) + self.bg + np.random.randn(3) * self.sigma_g / np.sqrt(dt)
        
        return a_noisy, g_noisy

def setup_euroc_dirs(base_dir):
    os.makedirs(f"{base_dir}/cam0/data", exist_ok=True)
    os.makedirs(f"{base_dir}/imu0", exist_ok=True)
    os.makedirs(f"{base_dir}/groundtruth", exist_ok=True)
    
    with open(f"{base_dir}/cam0/data.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["#timestamp [ns]", "filename"])
        
    with open(f"{base_dir}/imu0/data.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["#timestamp [ns]", "w_RS_S_x [rad s^-1]", "w_RS_S_y [rad s^-1]", "w_RS_S_z [rad s^-1]", 
                         "a_RS_S_x [m s^-2]", "a_RS_S_y [m s^-2]", "a_RS_S_z [m s^-2]"])

    with open(f"{base_dir}/groundtruth/data.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["#timestamp [ns]", "p_RS_R_x [m]", "p_RS_R_y [m]", "p_RS_R_z [m]", 
                         "q_RS_w []", "q_RS_x []", "q_RS_y []", "q_RS_z []"])

def extract_bag(bag_path, out_dir):
    setup_euroc_dirs(out_dir)
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    imu_model = BNO055NoiseModel()
    
    with Reader(bag_path) as reader:
        for connection, timestamp, rawdata in reader.messages():
            
            # --- IMU (Con Inyección de Ruido Físico) ---
            if connection.topic == '/imu/raw':
                msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
                ts_ns = msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec
                if ts_ns == 0: continue 
                
                a_noisy, g_noisy = imu_model.process(
                    ts_ns, 
                    msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z,
                    msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z
                )
                
                with open(f"{out_dir}/imu0/data.csv", 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([ts_ns, g_noisy[0], g_noisy[1], g_noisy[2], a_noisy[0], a_noisy[1], a_noisy[2]])
            
            # --- GROUNDTRUTH (Sin ruido, para validación pura de Basalt) ---
            elif connection.topic == '/validation':
                msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
                ts_ns = msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec
                if ts_ns == 0: continue 
                
                pos = msg.pose.pose.position
                ori = msg.pose.pose.orientation
                with open(f"{out_dir}/groundtruth/data.csv", 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([ts_ns, pos.x, pos.y, pos.z, ori.w, ori.x, ori.y, ori.z])

            # --- CAMERA (Simulación de ROI y paso a Blanco y Negro) ---
            elif connection.topic == '/camera/raw':
                msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
                ts_ns = msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec
                if ts_ns == 0: continue 
                
                filename = f"{ts_ns}.png"
                
                if msg.encoding == 'rgb8':
                    # Decodificar el array de bytes a matriz de imagen RGB
                    img_rgb = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
                    
                    # Calcular el ROI cuadrado centrado
                    h, w = img_rgb.shape[:2]
                    size = min(h, w) # El lado más corto determina el tamaño del ROI
                    start_y = (h - size) // 2
                    start_x = (w - size) // 2
                    
                    # Aplicar el recorte (Crop) simulando la salida por hardware
                    img_roi = img_rgb[start_y:start_y+size, start_x:start_x+size]
                    
                    # Convertir a Blanco y Negro
                    img_gray = cv2.cvtColor(img_roi, cv2.COLOR_RGB2GRAY)
                    
                    # Guardar la imagen final
                    cv2.imwrite(f"{out_dir}/cam0/data/{filename}", img_gray)
                
                with open(f"{out_dir}/cam0/data.csv", 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([ts_ns, filename])

if __name__ == '__main__':
    # El punto '.' asume que el script se corre al lado del archivo .mcap
    extract_bag('.', 'basalt_dataset') 
    print("Extracción a formato Basalt/Euroc con ROI y Ruido IMU completada con éxito.")