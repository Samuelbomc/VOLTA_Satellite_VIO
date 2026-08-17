import cv2
import numpy as np
import glob
import os


CHECKERBOARD = (6, 9) 


subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_CHECK_COND + cv2.fisheye.CALIB_FIX_SKEW

# Generate the ideal 3D chessboard model (corrected for fisheye).
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 1, 3), np.float32)
objp[:, 0, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

objpoints = [] 
imgpoints = [] 

# --- 2. LOAD IMAGES ---

ruta_imagenes = 'imagenes_calibracion/*.jpg'
images = glob.glob(ruta_imagenes)


_img_shape = None

print("Processing images, please wait...")
for fname in images:
    img = cv2.imread(fname)
    if _img_shape == None:
        _img_shape = img.shape[:2]
    else:
        assert _img_shape == img.shape[:2], "Error: All photos must have the same resolution."

    # Convert to grayscale.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Find the corners (the "distorted reality").
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

    # If corners are found, refine them to sub-pixel accuracy and store them.
    if ret == True:
        objpoints.append(objp)
        cv2.cornerSubPix(gray, corners, (3,3), (-1,-1), subpix_criteria)
        imgpoints.append(corners)
        print(f"[OK] Corners detected in: {os.path.basename(fname)}")
    else:
        print(f"[FAIL] Corners NOT detected in: {os.path.basename(fname)}")

# --- 3. FISHEYE CALIBRATION ---
print("\nFiltering problematic images and computing parameters...")

N_OK = len(objpoints)
if N_OK == 0:
    print("The chessboard was not detected in any image.")
    exit()

# 1. Force strict float64.
objpoints = [np.asarray(p, dtype=np.float64).reshape(1, -1, 3) for p in objpoints]
imgpoints = [np.asarray(p, dtype=np.float64).reshape(1, -1, 2) for p in imgpoints]

# 2. Initial K matrix guess.
w, h = gray.shape[::-1]
K_guess = np.zeros((3, 3), dtype=np.float64)
K_guess[0, 0] = K_guess[1, 1] = w * 0.5
K_guess[0, 2] = w * 0.5
K_guess[1, 2] = h * 0.5
K_guess[2, 2] = 1.0
D_guess = np.zeros((4, 1), dtype=np.float64)

calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_FIX_SKEW + cv2.fisheye.CALIB_USE_INTRINSIC_GUESS

# --- NEW: ISOLATION FILTER ---
# Test each image one by one. If the math collapses, discard it.
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
        # If it survives without error, it is a safe image.
        objpoints_seguros.append(objpoints[i])
        imgpoints_seguros.append(imgpoints[i])
    except Exception as e:
        print(f"[-] Discarding image at index {i} (caused mathematical collapse)")

print(f"\nSafe images for calibration: {len(objpoints_seguros)} of {N_OK}")

if len(objpoints_seguros) < 3:
    print("Error: Too few valid images remain. Take photos with the board at a steeper angle!")
    exit()

# 3. Final calibration using safe images only.
rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(len(objpoints_seguros))]
tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(len(objpoints_seguros))]

rms, K, D, _, _ = cv2.fisheye.calibrate(
    objpoints_seguros, imgpoints_seguros, gray.shape[::-1],
    K_guess, D_guess, rvecs, tvecs, calibration_flags,
    (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
)

print("\n" + "="*50)
print("              CALIBRATION SUCCESSFUL!")
print("="*50)
print(f"Reprojection RMS error: {rms:.4f} (should be below 1.0 for VIO)")
print("-" * 50)
print("Intrinsic matrix (K):\n", K)
print("-" * 50)
print("Fisheye distortion coefficients (D) [k1, k2, k3, k4]:\n", D.T) 
print("="*50)
