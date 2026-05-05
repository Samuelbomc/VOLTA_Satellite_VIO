import cv2
import numpy as np
import glob
import os


CHECKERBOARD = (6, 9) 


subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_CHECK_COND + cv2.fisheye.CALIB_FIX_SKEW

# Generar el modelo 3D perfecto del tablero (Corregido para Fisheye)
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 1, 3), np.float32)
objp[:, 0, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

objpoints = [] 
imgpoints = [] 

# --- 2. CARGAR IMÁGENES ---

ruta_imagenes = 'imagenes_calibracion/*.jpg'
images = glob.glob(ruta_imagenes)


_img_shape = None

print("Procesando imágenes, por favor espera...")
for fname in images:
    img = cv2.imread(fname)
    if _img_shape == None:
        _img_shape = img.shape[:2]
    else:
        assert _img_shape == img.shape[:2], "Error: Todas las fotos deben tener la misma resolución."

    # Convertir a blanco y negro
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Encontrar las esquinas (la "Realidad Deformada")
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

    # Si encuentra las esquinas, aplica precisión sub-píxel y las guarda
    if ret == True:
        objpoints.append(objp)
        cv2.cornerSubPix(gray, corners, (3,3), (-1,-1), subpix_criteria)
        imgpoints.append(corners)
        print(f"[OK] Esquinas detectadas en: {os.path.basename(fname)}")
    else:
        print(f"[FALLO] Esquinas NO detectadas en: {os.path.basename(fname)}")

# --- 3. CÁLCULO DE CALIBRACIÓN FISHEYE ---
print("\nFiltrando imágenes problemáticas y calculando parámetros...")

N_OK = len(objpoints)
if N_OK == 0:
    print("No se detectó el tablero en ninguna foto.")
    exit()

# 1. FORZAR FLOAT64 ESTRICTO
objpoints = [np.asarray(p, dtype=np.float64).reshape(1, -1, 3) for p in objpoints]
imgpoints = [np.asarray(p, dtype=np.float64).reshape(1, -1, 2) for p in imgpoints]

# 2. SUPOSICIÓN INICIAL DE MATRIZ K
w, h = gray.shape[::-1]
K_guess = np.zeros((3, 3), dtype=np.float64)
K_guess[0, 0] = K_guess[1, 1] = w * 0.5
K_guess[0, 2] = w * 0.5
K_guess[1, 2] = h * 0.5
K_guess[2, 2] = 1.0
D_guess = np.zeros((4, 1), dtype=np.float64)

calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_FIX_SKEW + cv2.fisheye.CALIB_USE_INTRINSIC_GUESS

# --- NUEVO: FILTRO AISLANTE ---
# Probamos cada imagen 1 por 1. Si colapsa la matemática, la descartamos.
objpoints_seguros = []
imgpoints_seguros = []

for i in range(len(objpoints)):
    try:
        cv2.fisheye.calibrate(
            [objpoints[i]], [imgpoints[i]], gray.shape[::-1],
            K_guess.copy(), D_guess.copy(), 
            [np.zeros((1, 1, 3), dtype=np.float64)], 
            [np.zeros((1, 1, 3), dtype=np.float64)], 
            calibration_flags,
            (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
        )
        # Si sobrevive sin error, es una imagen segura
        objpoints_seguros.append(objpoints[i])
        imgpoints_seguros.append(imgpoints[i])
    except Exception as e:
        print(f"[-] Descartando imagen en el índice {i} (Causaba colapso matemático)")

print(f"\nImágenes seguras para calibrar: {len(objpoints_seguros)} de {N_OK}")

if len(objpoints_seguros) < 3:
    print("Error: Quedaron muy pocas imágenes válidas. ¡Toma fotos con el tablero más inclinado!")
    exit()

# 3. CALIBRACIÓN FINAL SOLO CON IMÁGENES SEGURAS
rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(len(objpoints_seguros))]
tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(len(objpoints_seguros))]

rms, K, D, _, _ = cv2.fisheye.calibrate(
    objpoints_seguros, imgpoints_seguros, gray.shape[::-1],
    K_guess, D_guess, rvecs, tvecs, calibration_flags,
    (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
)

print("\n" + "="*50)
print("              ¡CALIBRACIÓN EXITOSA!")
print("="*50)
print(f"Error RMS de reproyección: {rms:.4f} (Debe ser menor a 1.0 para VIO)")
print("-" * 50)
print("Matriz Intrínseca (K):\n", K)
print("-" * 50)
print("Coeficientes de Distorsión Fisheye (D) [k1, k2, k3, k4]:\n", D.T) 
print("="*50)