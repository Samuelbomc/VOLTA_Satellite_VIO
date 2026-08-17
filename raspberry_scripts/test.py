import serial
import time

SERIAL_PORT = "/dev/ttyAMA0"
BAUD_RATE = 115200
TIMEOUT_S = 1.0

def main() -> None:
    try:
        with serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=TIMEOUT_S) as ser:
            print(f"Escuchando en {SERIAL_PORT} a {BAUD_RATE} baud...")
            while True:
                data = ser.read(64)
                if data:
                    print(f"RX ({len(data)} bytes): {data.hex()}")
                else:
                    print("ERROR: no se recibió ningún dato.")
    except serial.SerialException as exc:
        print(f"ERROR serial: {exc}")
    except KeyboardInterrupt:
        print("Salida solicitada por el usuario.")

if __name__ == "__main__":
    main()