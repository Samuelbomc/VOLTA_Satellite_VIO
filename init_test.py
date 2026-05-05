import serial
import struct
import logging
import time

# --- CONFIGURACIÓN DE LOGS ---
# Usamos INFO para ver claramente los mensajes en la terminal durante la prueba
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

# --- CONSTANTES DE CONFIGURACIÓN ---
PUERTO_SERIAL = '/dev/serial0'
BAUD_RATE = 115200
TIMEOUT_SERIAL = 0.1  

# --- CONSTANTES DEL PROTOCOLO ---
SYNC_1 = 0xAA
SYNC_2 = 0x55
ID_CAM_TP = 0x11
ID_INIT_CMD = 0xF0

def calcular_crc16(data: bytes) -> int:
    """Calcula el CRC-16-CCITT."""
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

def leer_exacto(puerto: serial.Serial, n_bytes: int) -> bytes:
    """Lee exactamente n_bytes de forma segura para evitar lecturas parciales."""
    start_time = time.monotonic()
    data = b''
    while len(data) < n_bytes:
        if (time.monotonic() - start_time) > 0.5: # Timeout de prueba más largo (0.5s)
            return None  
        chunk = puerto.read(n_bytes - len(data))
        if chunk:
            data += chunk
    return data

def esperar_sync(puerto: serial.Serial) -> bool:
    """Busca la secuencia de sincronización [0xAA, 0x55]."""
    estado = 0
    while True:
        chunk = puerto.read(1)
        if not chunk:
            return False
            
        b = chunk[0]
        if estado == 0 and b == SYNC_1:
            estado = 1
        elif estado == 1:
            if b == SYNC_2:
                return True
            elif b == SYNC_1:
                estado = 1
            else:
                estado = 0

def responder_inicializacion(puerto: serial.Serial):
    """Construye y envía la respuesta de inicialización a la computadora central."""
    payload = bytes([ID_CAM_TP])
    longitud = len(payload)
    
    # Calcular CRC sobre ID, Longitud y Payload
    datos_para_crc = bytes([ID_INIT_CMD, longitud]) + payload
    crc_calculated = calcular_crc16(datos_para_crc)
    crc_bytes = struct.pack('<H', crc_calculated)
    
    # Ensamblar paquete completo
    paquete = bytes([SYNC_1, SYNC_2, ID_INIT_CMD, longitud]) + payload + crc_bytes
    
    puerto.write(paquete)
    puerto.flush()
    logging.info("--> RESPUESTA ENVIADA: Paquete INIT con ID de Cámara (0x11).")

def ejecutar_prueba():
    """Bucle principal exclusivo para la prueba de inicialización."""
    try:
        with serial.Serial(PUERTO_SERIAL, BAUD_RATE, timeout=TIMEOUT_SERIAL) as puerto:
            logging.info(f"Conexión abierta en {PUERTO_SERIAL} a {BAUD_RATE} baudios.")
            logging.info("ESPERANDO comando de inicialización (0xF0) del Main OBC...\n" + "-"*50)
            
            while True:
                if not esperar_sync(puerto):
                    continue
                    
                cabecera = leer_exacto(puerto, 2)
                if not cabecera:
                    continue
                    
                id_paquete = cabecera[0]
                longitud = cabecera[1]
                
                # Para esta prueba estricta, solo nos interesa el ID_INIT_CMD
                if id_paquete != ID_INIT_CMD:
                    # Descartar paquetes que no sean de inicialización para mantener el buffer limpio
                    puerto.read(puerto.in_waiting) 
                    continue
                
                payload = leer_exacto(puerto, longitud)
                crc_bytes = leer_exacto(puerto, 2)
                
                if not payload or not crc_bytes:
                    logging.warning("Paquete INIT incompleto. Ignorando...")
                    continue
                    
                # Verificar CRC
                crc_recibido = struct.unpack('<H', crc_bytes)[0]
                datos_para_crc = bytes([id_paquete, longitud]) + payload
                
                if calcular_crc16(datos_para_crc) != crc_recibido:
                    logging.error("Fallo de CRC en el paquete INIT. Ignorando...")
                    continue
                
                id_destino = payload[0]
                logging.info(f"<-- RECIBIDO: Petición INIT válida. Buscando ID: {hex(id_destino)}")
                
                # Si el sistema principal busca esta cámara, responder
                if id_destino == ID_CAM_TP:
                    responder_inicializacion(puerto)
                    logging.info("-" * 50)
                else:
                    logging.warning(f"La petición era para otro ID ({hex(id_destino)}). No se responde.")

    except serial.SerialException as error:
        logging.critical(f"Error abriendo el puerto serial: {error}")
    except KeyboardInterrupt:
        logging.info("\nPrueba finalizada por el usuario (Ctrl+C).")

if __name__ == '__main__':
    ejecutar_prueba()