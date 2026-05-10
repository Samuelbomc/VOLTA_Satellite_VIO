# Data Communication Standar
---
## General

Before each packet, two separate bytes [0xAA 0x55] (uint8_t) will be sent to identify its arrival.

Each packet will have a designated ID (uint8_t) at the beginning, followed by a set packet length (uint8_t) to confirm its termination, and a CRC (uint16_t) at the end to verify the packet's integrity upon receipt.

The length_of_package variable indicates the payload length (the exact number of bytes containing the data, excluding the Sync bytes, ID, Length byte, and CRC).

All multi-byte variables shall be transmitted in Little-Endian byte order.

---

## ID Package Assignment

* ID_CTR_TP: 0X10
* ID_CAM_TP: 0X11
* ID_FGH_SP: 0X20
* ID_INIT_CMD: 0XF0

---

## Flight Data Packages

* **control_telemetry_package (CTR_TP):** package sent to the control algorithm

    * **Frequency:** 100 Hz
    * **Weight:** 59 Bytes
    * **Content:** 

        | VARIABLE | TYPE | UNIT |
        | :--- | :---: | :--- |
        | 0xAA | uint8_t | N/A |
        | 0x55 | uint8_t | N/A |
        | ID_CTR_TP | uint8_t | N/A |
        | length_of_package | uint8_t | byte |
        | timestamp | uint32_t | ms |
        | altitude | float32 | m |
        | vertical_velocity | float32 | m/s |
        | ax, ay, az | float32 | m/s^2 |
        | gx, gy, gz | float32 | rad/s |
        | qw, qx, qy, qz | float32 | N/A |
        | flight_state | uint8_t | N/A |
        | CRC-16-CCITT | uint16_t | N/A |

* **camera_telemetry_package (CAM_TP):** package sent to the camera algorithm

    * **Frequency:** 100 Hz
    * **Weight:** 34 Bytes
    * **Content:** 

        | VARIABLE | TYPE | UNIT |
        | :--- | :---: | :--- |
        | 0xAA | uint8_t | N/A |
        | 0x55 | uint8_t | N/A |
        | ID_CAM_TP | uint8_t | N/A |
        | length_of_package | uint8_t | byte |
        | timestamp | uint32_t | ms |
        | ax, ay, az | float32 | m/s^2 |
        | gx, gy, gz | float32 | rad/s |
        | CRC-16-CCITT | uint16_t | N/A |

* **flight_state_package (FGH_SP):** package containing flight status

    * **Frequency:** N/A
    * **Weight:** 15 Bytes
    * **Content:** 

        | VARIABLE | TYPE | UNIT |
        | :--- | :---: | :--- |
        | 0xAA | uint8_t | N/A |
        | 0x55 | uint8_t | N/A |
        | ID_FGH_SP | uint8_t | N/A |
        | length_of_package | uint8_t | byte |
        | timestamp | uint32_t | ms |
        | altitude | float32 | m |
        | flight_state | uint8_t | N/A |
        | CRC-16-CCITT | uint16_t | N/A |

---

## Initialize Communications Packages

* **package_init_from_Main:** package sent from main OBC to verify communications, contains the receiver ID

    * **Frequency:** N/A
    * **Weight:** 7 Bytes
    * **Content:** 

        | VARIABLE | TYPE | UNIT |
        | :--- | :---: | :--- |
        | 0xAA | uint8_t | N/A |
        | 0x55 | uint8_t | N/A |
        | ID_INIT_CMD | uint8_t | N/A |
        | length_of_package | uint8_t | byte |
        | ID_CAM_TP / ID_CTR_TP | uint8_t | N/A |
        | CRC-16-CCITT | uint16_t | N/A |

* **package_init_to_Main:** package sent from main OBC to verify communications, contains the transmitter ID

    * **Frequency:** N/A
    * **Weight:** 7 Bytes
    * **Content:** 

        | VARIABLE | TYPE | UNIT |
        | :--- | :---: | :--- |
        | 0xAA | uint8_t | N/A |
        | 0x55 | uint8_t | N/A |
        | ID_INIT_CMD | uint8_t | N/A |
        | length_of_package | uint8_t | byte |
        | ID_CAM_TP / ID_CTR_TP | uint8_t | N/A |
        | CRC-16-CCITT | uint16_t | N/A |
