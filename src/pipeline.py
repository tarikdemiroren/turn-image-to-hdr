import cv2
import numpy as np
import os
import sys

def simulate_hdr(input_path, output_path, exposure_boost=2.5, saturation_boost=1.4, bloom_strength=0.7):
    print(f"[1] Loading image from: {input_path}")
    img = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {input_path}")
    print(f"[2] Image loaded, shape={img.shape}, dtype={img.dtype}")
    img = img.astype(np.float32) / 255.0

    # --- Step 1: Exposure Boost ---
    print("[3] Applying exposure boost...")
    boosted = np.clip(img * exposure_boost, 0, 1)
    print(f"    → max after boost = {boosted.max():.3f}")

    # --- Step 2: Local Contrast (CLAHE) ---
    print("[4] Running CLAHE...")
    lab = cv2.cvtColor((boosted * 255).astype(np.uint8), cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge((l_clahe, a, b))
    boosted_contrast = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR).astype(np.float32) / 255.0
    print(f"    → min/max after CLAHE = {boosted_contrast.min():.3f}/{boosted_contrast.max():.3f}")

    # --- Step 3: Bloom Effect ---
    print("[5] Adding bloom effect...")
    bright = np.clip(boosted_contrast - 0.6, 0, 1)
    bright_blurred = cv2.GaussianBlur(bright, (0, 0), sigmaX=10, sigmaY=10)
    bloomed = np.clip(boosted_contrast + bright_blurred * bloom_strength, 0, 1)
    print(f"    → max after bloom = {bloomed.max():.3f}")

    # --- Step 4: Saturation Boost ---
    print("[6] Boosting saturation...")
    hsv = cv2.cvtColor((bloomed * 255).astype(np.uint8), cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = np.clip(s.astype(np.float32) * saturation_boost, 0, 255).astype(np.uint8)
    hsv_boosted = cv2.merge((h, s, v))
    vivid = cv2.cvtColor(hsv_boosted, cv2.COLOR_HSV2BGR).astype(np.float32) / 255.0
    print(f"    → max after saturation = {vivid.max():.3f}")

    # --- Step 5: Soft Highlight Curve ---
    print("[7] Applying soft highlight curve...")
    def soft_highlight_curve(x):
        return np.where(x < 0.8, x, 1.0 - 0.5 * np.exp(-(x - 0.8) * 10))
    final = soft_highlight_curve(vivid)
    print(f"    → max after tone curve = {final.max():.3f}")

    # --- Step 6: Save as 16-bit PNG ---
    print(f"[8] Saving to: {output_path}")
    output_16bit = np.clip(final * 65535, 0, 65535).astype(np.uint16)
    ok = cv2.imwrite(output_path, output_16bit)
    print(f"    → write success = {ok}")
    if not ok:
        raise IOError(f"Failed to write image to {output_path}")
    print("[9] Done!")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python simulate_hdr.py <input_image_path>")
        sys.exit(1)

    in_path = sys.argv[1]
    base, ext = os.path.splitext(in_path)
    out_path = base + "-hdr.png"
    simulate_hdr(in_path, out_path)
