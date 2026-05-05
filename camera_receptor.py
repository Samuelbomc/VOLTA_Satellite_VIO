import serial
import struct
import logging
import time
import threading
import queue

# --- LOGGING CONFIGURATION ---
logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')

# --- CONFIGURATION CONSTANTS ---
SERIAL_PORT = '/dev/serial0'
BAUD_RATE = 115200
SERIAL_TIMEOUT = 0.01  
READ_MAX_TIME = 0.1    

# --- PROTOCOL CONSTANTS ---
SYNC_1 = 0xAA
SYNC_2 = 0x55
ID_CAM_TP = 0x11
ID_INIT_CMD = 0xF0

LEN_CAM_TP = 28
LEN_INIT_CMD = 1
MAX_PAYLOAD = 64  


class CameraTelemetryNode:
    def __init__(self, port: str, baud_rate: int, timeout: float):
        self.port_name = port
        self.baud_rate = baud_rate
        self.timeout = timeout
        self.serial_conn = None
        
        # State tracking
        self.is_running = False
        self.last_init_time = 0.0
        
        # Concurrency & Metrics
        self.state_lock = threading.Lock()     # Protects node state shared across threads
        self.serial_lock = threading.Lock()    # Serial hardware access must be serialized
        
        self.last_imu_time = time.monotonic()
        self.dropped_packets = 0               
        
        # Thread-safe queue
        self.imu_queue = queue.Queue(maxsize=200)

    def connect(self):
        """Initializes the serial connection."""
        self.serial_conn = serial.Serial(
            port=self.port_name,
            baudrate=self.baud_rate,
            timeout=self.timeout
        )
        self.is_running = True
        with self.state_lock:
            self.last_imu_time = time.monotonic()
            
        logging.info("Flight system started on %s at %d baud", self.port_name, self.baud_rate)

    def close(self):
        """Closes the serial connection safely."""
        self.is_running = False
        with self.serial_lock:
            if self.serial_conn and self.serial_conn.is_open:
                self.serial_conn.close()

    @staticmethod
    def calculate_crc16(data: bytes) -> int:
        """Calculates the CRC-16-CCITT checksum."""
        crc = 0xFFFF
        for byte in data:
            crc ^= (byte << 8)
            for _ in range(8):
                if crc & 0x8000:
                    crc = ((crc << 1) ^ 0x1021)
                else:
                    crc = (crc << 1)
                crc &= 0xFFFF
        return crc

    def read_exact_bytes(self, n_bytes: int, max_time: float = READ_MAX_TIME) -> bytes:
        """Reads exactly n_bytes safely; returns None on timeout."""
        start_time = time.monotonic()
        data = b''
        
        while len(data) < n_bytes:
            if (time.monotonic() - start_time) > max_time:
                return None  
                
            # Protect serial hardware access across threads.
            with self.serial_lock:
                chunk = self.serial_conn.read(n_bytes - len(data))
                
            if chunk:
                data += chunk
                
        return data

    def wait_for_sync(self, max_attempts: int = 2000) -> bool:
        """State machine to find the [0xAA, 0x55] sync sequence."""
        state = 0
        attempts = 0
        
        while attempts < max_attempts:
            with self.serial_lock:
                chunk = self.serial_conn.read(1)
                
            if not chunk:
                return False
                
            b = chunk[0]
            attempts += 1
            
            if state == 0:
                if b == SYNC_1:
                    state = 1
            elif state == 1:
                if b == SYNC_2:
                    return True
                elif b == SYNC_1:
                    state = 1
                else:
                    state = 0
                    
        return False

    def handle_buffer_overflow(self):
        """Hybrid recovery for overflow and minor desyncs."""
        with self.serial_lock:
            in_waiting = self.serial_conn.in_waiting
            if in_waiting > 128:
                logging.warning("Buffer overflow (%d bytes). Resetting input buffer.", in_waiting)
                self.serial_conn.reset_input_buffer()
            elif in_waiting > 0:
                self.serial_conn.read(1)

    def send_init_response(self):
        """Builds and sends the package_init_to_Main."""
        payload = bytes([ID_CAM_TP])
        payload_len = len(payload)
        
        data_for_crc = bytes([ID_INIT_CMD, payload_len]) + payload
        crc_calculated = self.calculate_crc16(data_for_crc)
        crc_bytes = struct.pack('<H', crc_calculated)
        
        packet = bytes([SYNC_1, SYNC_2, ID_INIT_CMD, payload_len]) + payload + crc_bytes
        
        with self.serial_lock:
            self.serial_conn.write(packet)
            self.serial_conn.flush() 

    def listen_and_decode(self) -> str:
        """Main pipeline for UART reading."""
        if not self.wait_for_sync():
            return "SYNC_LOST"
            
        header = self.read_exact_bytes(2)
        if not header:
            return "TIMEOUT"
            
        packet_id = header[0]
        payload_len = header[1]
        
        if payload_len > MAX_PAYLOAD:
            self.handle_buffer_overflow()
            return "ERROR_LENGTH"
        
        is_valid_header = False
        if packet_id == ID_CAM_TP and payload_len == LEN_CAM_TP:
            is_valid_header = True
        elif packet_id == ID_INIT_CMD and payload_len == LEN_INIT_CMD:
            is_valid_header = True
            
        if not is_valid_header:
            flushed = self.read_exact_bytes(payload_len + 2)
            if not flushed:
                self.handle_buffer_overflow()
                return "ERROR_FLUSH"
            return "ERROR_HEADER"
            
        payload = self.read_exact_bytes(payload_len)
        if not payload:
            return "TIMEOUT"
            
        crc_bytes = self.read_exact_bytes(2)
        if not crc_bytes:
            return "TIMEOUT"
            
        received_crc = struct.unpack('<H', crc_bytes)[0]
        data_for_crc = bytes([packet_id, payload_len]) + payload
        
        if self.calculate_crc16(data_for_crc) != received_crc:
            return "ERROR_CRC"
            
        # --- SUCCESSFUL PACKET PROCESSING ---
        if packet_id == ID_CAM_TP:
            data = struct.unpack('<I 6f', payload)
            recv_time = time.monotonic()
            
            with self.state_lock:
                self.last_imu_time = recv_time
            
            # Diagnostic queue check (approximate but useful)
            q_size = self.imu_queue.qsize()
            if q_size > 150:
                logging.warning("Queue backlog growing critically: %d/200", q_size)
                
            try:
                self.imu_queue.put_nowait((recv_time, data))
            except queue.Full:
                self.dropped_packets += 1
                logging.warning("Queue Full! Dropped packets: %d", self.dropped_packets)
                
            return "IMU_OK"
            
        elif packet_id == ID_INIT_CMD:
            target_id = payload[0]
            current_time = time.monotonic()
            
            if current_time - self.last_init_time < 2.0:
                return "INIT_SPAM"
                
            self.last_init_time = current_time
            if target_id == ID_CAM_TP:
                self.send_init_response()
            return "INIT_OK"

        return "UNKNOWN_ID"


# --- THREAD WORKERS ---

def serial_worker(node: CameraTelemetryNode):
    logging.info("Serial UART worker started.")
    try:
        while node.is_running:
            node.listen_and_decode()
    except Exception as e:
        logging.critical("CRASH in Serial Thread: %s", e)
        node.is_running = False

def ros_processing_worker(node: CameraTelemetryNode):
    logging.info("ROS Processing worker started.")
    try:
        while node.is_running:
            try:
                recv_time, data = node.imu_queue.get(timeout=0.5)
                # --- ROS PUBLISH LOGIC GOES HERE ---
            except queue.Empty:
                pass 
    except Exception as e:
        logging.critical("CRASH in ROS Thread: %s", e)
        node.is_running = False


# --- MAIN LIFECYCLE (WITH ACTIVE WATCHDOG) ---

if __name__ == '__main__':
    telemetry_node = CameraTelemetryNode(SERIAL_PORT, BAUD_RATE, SERIAL_TIMEOUT)
    threads = []
    
    try:
        telemetry_node.connect()
        
        uart_thread = threading.Thread(target=serial_worker, args=(telemetry_node,))
        ros_thread = threading.Thread(target=ros_processing_worker, args=(telemetry_node,))
        
        threads.extend([uart_thread, ros_thread])
        for t in threads:
            t.start()
        
        logging.info("System operational. Active Watchdog running.")
        
        # --- ACTIVE WATCHDOG ---
        while telemetry_node.is_running:
            time.sleep(0.5)
            
            with telemetry_node.state_lock:
                time_since_last_imu = time.monotonic() - telemetry_node.last_imu_time
                
            if time_since_last_imu > 3600.0: # TEMPORARY: 1 HOUR FOR TESTING
                logging.error("WATCHDOG ALARM: No IMU data in %.1fs. Triggering hardware reset.", time_since_last_imu)
                
                # Safely reset UART buffers from the main thread.
                if telemetry_node.serial_conn and telemetry_node.serial_conn.is_open:
                    with telemetry_node.serial_lock:
                        telemetry_node.serial_conn.reset_input_buffer()
                        telemetry_node.serial_conn.reset_output_buffer()
                
                with telemetry_node.state_lock:
                    telemetry_node.last_imu_time = time.monotonic()
                
    except serial.SerialException as error:
        logging.critical("Critical serial port error: %s", error)
    except KeyboardInterrupt:
        logging.info("Manual shutdown triggered.")
    finally:
        logging.info("Initiating graceful shutdown...")
        telemetry_node.is_running = False 
        
        for t in threads:
            if t.is_alive():
                t.join(timeout=2.0)
                if t.is_alive():
                    logging.warning("Thread %s did not terminate cleanly.", t.name)
                
        telemetry_node.close()
        logging.info("System closed safely.")