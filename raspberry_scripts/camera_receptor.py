import serial
import serial
import struct
import logging
import time

from telemetry_pipeline import TelemetryProcessor

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
        self.processor = TelemetryProcessor()
        
        # State tracking
        self.is_running = False
        self.last_init_time = 0.0
        
        self.last_imu_time = time.monotonic()
        self.dropped_packets = 0               
        self.last_rate_time = time.monotonic() # TEMPORARY
        self.imu_packet_count = 0 # TEMPORARY
        self.last_payload_log_time = 0.0 # TEMPORARY
        self.init_received = False
        
        self.last_watchdog_check = time.monotonic()

    def connect(self):
        """Initializes the serial connection."""
        self.serial_conn = serial.Serial(
            port=self.port_name,
            baudrate=self.baud_rate,
            timeout=self.timeout
        )
        self.is_running = True
        self.last_imu_time = time.monotonic()
        logging.debug("Connected to serial port %s at %d baud", self.port_name, self.baud_rate)  # TEMPORARY
            
        logging.info("Flight system started on %s at %d baud", self.port_name, self.baud_rate)

    def close(self):
        """Closes the serial connection safely."""
        self.is_running = False
        if self.serial_conn and self.serial_conn.is_open:
            logging.debug("Closing serial port %s", self.port_name)  # TEMPORARY
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
                logging.debug("Read timeout after %.3fs while waiting for %d bytes (received %d)",
                              max_time, n_bytes, len(data))  # TEMPORARY
                return None  
                
            chunk = self.serial_conn.read(n_bytes - len(data))
                
            if chunk:
                data += chunk
            else:
                logging.debug("No data chunk available while reading %d bytes", n_bytes)  # TEMPORARY
                
        return data

    def wait_for_sync(self, max_attempts: int = 2000) -> bool:
        """State machine to find the [0xAA, 0x55] sync sequence."""
        state = 0
        attempts = 0
        
        while attempts < max_attempts:
            chunk = self.serial_conn.read(1)
                
            if not chunk:
                logging.debug("Sync wait failed: no data available on attempt %d", attempts)  # TEMPORARY
                return False
                
            b = chunk[0]
            attempts += 1
            
            if state == 0:
                if b == SYNC_1:
                    state = 1
            elif state == 1:
                if b == SYNC_2:
                    logging.debug("Sync sequence found after %d attempts", attempts)  # TEMPORARY
                    return True
                elif b == SYNC_1:
                    state = 1
                else:
                    state = 0
                    
        return False

    def handle_buffer_overflow(self):
        """Hybrid recovery for overflow and minor desyncs."""
        in_waiting = self.serial_conn.in_waiting
        if in_waiting > 128:
            logging.warning("Buffer overflow (%d bytes). Resetting input buffer.", in_waiting)
            self.serial_conn.reset_input_buffer()
        elif in_waiting > 0:
            logging.debug("Minor desync detected. Flushing 1 byte from buffer (%d bytes waiting).", in_waiting)  # TEMPORARY
            self.serial_conn.read(1)

    def send_init_response(self):
        """Builds and sends the package_init_to_Main."""
        payload = bytes([ID_CAM_TP])
        payload_len = len(payload)
        
        data_for_crc = bytes([SYNC_1, SYNC_2, ID_INIT_CMD, payload_len]) + payload
        crc_calculated = self.calculate_crc16(data_for_crc)
        crc_bytes = struct.pack('<H', crc_calculated)
        
        packet = bytes([SYNC_1, SYNC_2, ID_INIT_CMD, payload_len]) + payload + crc_bytes
        
        self.serial_conn.write(packet)
        self.serial_conn.flush() 
        logging.debug("Sent init response packet (len=%d, crc=0x%04X)", len(packet), crc_calculated)  # TEMPORARY

    def listen_and_decode(self) -> str:
        """Main pipeline for UART reading."""
        if not self.wait_for_sync():
            logging.debug("Sync lost while listening for packet")  # TEMPORARY
            return "SYNC_LOST"
            
        header = self.read_exact_bytes(2)
        if not header:
            logging.debug("Timeout while reading packet header")  # TEMPORARY
            return "TIMEOUT"
            
        packet_id = header[0]
        payload_len = header[1]
        
        if payload_len > MAX_PAYLOAD:
            logging.debug("Invalid payload length %d (max %d)", payload_len, MAX_PAYLOAD)  # TEMPORARY
            self.handle_buffer_overflow()
            return "ERROR_LENGTH"

        if not self.init_received and packet_id != ID_INIT_CMD:
            logging.debug("Waiting for INIT. Ignoring packet id=0x%02X len=%d", packet_id, payload_len)  # TEMPORARY
            flushed = self.read_exact_bytes(payload_len + 2)
            if not flushed:
                self.handle_buffer_overflow()
                return "ERROR_FLUSH"
            return "WAIT_INIT"
        
        is_valid_header = False
        if packet_id == ID_CAM_TP and payload_len == LEN_CAM_TP:
            is_valid_header = True
        elif packet_id == ID_INIT_CMD and payload_len == LEN_INIT_CMD:
            is_valid_header = True
            
        if not is_valid_header:
            logging.debug("Invalid header: id=0x%02X len=%d", packet_id, payload_len)  # TEMPORARY
            flushed = self.read_exact_bytes(payload_len + 2)
            if not flushed:
                self.handle_buffer_overflow()
                return "ERROR_FLUSH"
            return "ERROR_HEADER"
            
        payload = self.read_exact_bytes(payload_len)
        if not payload:
            logging.debug("Timeout while reading payload (len=%d)", payload_len)  # TEMPORARY
            return "TIMEOUT"
            
        crc_bytes = self.read_exact_bytes(2)
        if not crc_bytes:
            logging.debug("Timeout while reading CRC bytes")  # TEMPORARY
            return "TIMEOUT"
            
        received_crc = struct.unpack('<H', crc_bytes)[0]
        data_for_crc = bytes([SYNC_1, SYNC_2, packet_id, payload_len]) + payload
        
        if self.calculate_crc16(data_for_crc) != received_crc:
            logging.debug("CRC mismatch for packet id=0x%02X (expected=0x%04X received=0x%04X)",
                          packet_id, self.calculate_crc16(data_for_crc), received_crc)  # TEMPORARY
            return "ERROR_CRC"
            
        # --- PACKET ROUTING ---
        if packet_id == ID_CAM_TP:
            if not self.init_received:
                logging.debug("IMU packet ignored before INIT")  # TEMPORARY
                return "IMU_BEFORE_INIT"
            data = struct.unpack('<I 6f', payload)
            recv_time = time.monotonic()

            self.last_imu_time = recv_time
            self.processor.handle_imu_packet(recv_time, data)
            logging.debug("IMU packet processed at %.6f", recv_time)  # TEMPORARY

            self.imu_packet_count += 1 # TEMPORARY
            rate_window = recv_time - self.last_rate_time # TEMPORARY
            if rate_window >= 1.0: # TEMPORARY
                rate_hz = self.imu_packet_count / rate_window # TEMPORARY
                logging.debug("IMU receive rate: %.2f Hz (window %.2fs)", rate_hz, rate_window)  # TEMPORARY
                self.imu_packet_count = 0 # TEMPORARY
                self.last_rate_time = recv_time # TEMPORARY

            if recv_time - self.last_payload_log_time >= 2.0: # TEMPORARY
                logging.debug("IMU payload sample: %s", data)  # TEMPORARY
                self.last_payload_log_time = recv_time # TEMPORARY
                
            return "IMU_OK"
            
        elif packet_id == ID_INIT_CMD:
            target_id = payload[0]
            current_time = time.monotonic()
            
            if current_time - self.last_init_time < 2.0:
                logging.debug("INIT command ignored due to spam protection")  # TEMPORARY
                return "INIT_SPAM"
                
            self.last_init_time = current_time
            if target_id == ID_CAM_TP:
                self.send_init_response()
                self.init_received = True
            logging.debug("INIT command processed for target id=0x%02X", target_id)  # TEMPORARY
            return "INIT_OK"

        return "UNKNOWN_ID"


# --- MAIN LIFECYCLE (WITH ACTIVE WATCHDOG) ---

def main():
    telemetry_node = CameraTelemetryNode(SERIAL_PORT, BAUD_RATE, SERIAL_TIMEOUT)

    try:
        telemetry_node.connect()
        logging.info("System operational. Active Watchdog running.")

        while telemetry_node.is_running:
            status = telemetry_node.listen_and_decode()
            if status in {"SYNC_LOST", "TIMEOUT", "ERROR_LENGTH", "ERROR_FLUSH", "ERROR_HEADER", "ERROR_CRC"}:
                logging.debug("Serial decode status: %s", status)  # TEMPORARY

            now = time.monotonic()
            if now - telemetry_node.last_watchdog_check < 0.5:
                continue
            telemetry_node.last_watchdog_check = now

            time_since_last_imu = now - telemetry_node.last_imu_time

            if not telemetry_node.init_received:
                continue

            if time_since_last_imu > 1:
                logging.error("WATCHDOG ALARM: No IMU data in %.1fs. Triggering hardware reset.", time_since_last_imu)

                # Safely reset UART buffers from the main thread.
                if telemetry_node.serial_conn and telemetry_node.serial_conn.is_open:
                    telemetry_node.serial_conn.reset_input_buffer()
                    telemetry_node.serial_conn.reset_output_buffer()

                telemetry_node.last_imu_time = time.monotonic()

    except serial.SerialException as error:
        logging.critical("Critical serial port error: %s", error)
    except KeyboardInterrupt:
        logging.info("Manual shutdown triggered.")
    finally:
        logging.info("Initiating graceful shutdown...")
        telemetry_node.is_running = False

        telemetry_node.close()
        logging.info("System closed safely.")


if __name__ == '__main__':
    main()
