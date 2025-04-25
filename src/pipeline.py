#!/usr/bin/env python3
"""
simulate_true_hdr_with_metadata.py

Generate a vibrant, HDR‑style 16‑bit PNG with embedded HDR metadata chunks.
Pipeline:
  1) Load & normalize
  2) Saturation boost
  3) Synthetic exposures for true HDR radiance map (Debevec)
  4) Tone‑mapping (Reinhard or Drago fallback)
  5) Bloom on luminance only
  6) Gamma lift
  7) Save as <input>-hdr.png with embedded PNG metadata (gAMA, cHRM, custom HDR info)
"""

import cv2
import numpy as np
import os
import sys
from PIL import Image, PngImagePlugin

def simulate_true_hdr_with_metadata(input_path):
    # Parameters
    saturation_boost = 1.8
    exposure_factors = [0.5, 1.0, 2.0]
    bloom_threshold = 0.6
    bloom_strength = 1.0
    tone_gamma = 2.2
    tone_intensity = 0.0
    tone_light_adapt = 1.0
    tone_color_adapt = 0.8
    final_gamma = 1.0 / 1.2

    # 1) Load & normalize
    img_bgr = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not load image: {input_path}")
    img = img_bgr.astype(np.float32) / 255.0

    # 2) Saturation boost in HSV
    hsv = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = cv2.split(hsv)
    s = np.clip(s * saturation_boost, 0, 255)
    img = cv2.cvtColor(cv2.merge((h, s, v)).astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32) / 255.0

    # 3) Synthetic exposures
    exposures = []
    for f in exposure_factors:
        ev = np.clip(img * f, 0, 1)
        exposures.append((ev * 255).astype(np.uint8))
    times = np.array(exposure_factors, dtype=np.float32)

    # 4) True HDR merge (Debevec)
    merge_debevec = cv2.createMergeDebevec()
    hdr = merge_debevec.process(exposures, times=times.copy())

    # 5) Tone‑mapping
    if hasattr(cv2, 'createTonemapReinhard'):
        tonemap = cv2.createTonemapReinhard(
            gamma=tone_gamma,
            intensity=tone_intensity,
            light_adapt=tone_light_adapt,
            color_adapt=tone_color_adapt
        )
        ldr = tonemap.process(hdr)
    elif hasattr(cv2, 'createTonemapDrago'):
        tonemap = cv2.createTonemapDrago(
            gamma=tone_gamma,
            saturation=tone_color_adapt,
            bias=0.85
        )
        ldr = tonemap.process(hdr)
    else:
        ldr = np.clip(hdr / (hdr + 1.0), 0, 1)

    # 6) Bloom on luminance only
    lab = cv2.cvtColor((ldr * 255).astype(np.uint8), cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    Lf = L.astype(np.float32) / 255.0
    bright = np.clip(Lf - bloom_threshold, 0, 1)
    blurred = cv2.GaussianBlur(bright, (0, 0), sigmaX=15, sigmaY=15)
    L2 = np.clip(Lf + blurred * bloom_strength, 0, 1)
    lab2 = cv2.merge((np.uint8(L2 * 255), A, B))
    ldr = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR).astype(np.float32) / 255.0

    # 7) Final gamma lift
    final = np.clip(ldr ** final_gamma, 0, 1)

    # 8) Convert to 16‑bit
    img16 = (final * 65535).astype(np.uint16)

    # 9) Prepare metadata
    pnginfo = PngImagePlugin.PngInfo()
    # Gamma chunk (1/2.2)
    pnginfo.add_text("gAMA", str(round(1.0 / tone_gamma, 6)))
    # Chromaticities (textual; e.g., BT.2020)
    pnginfo.add_text("cHRM", "Primaries=BT.2020")
    # Custom HDR metadata
    pnginfo.add_text("HDR_Method", "Debevec Merge + Reinhard ToneMap")
    pnginfo.add_text("Exposure_Factors", ",".join(map(str, exposure_factors)))
    pnginfo.add_text("Peak_Brightness_cd/m2", "1000")

    # 10) Save PNG
    base, _ = os.path.splitext(input_path)
    out_png = f"{base}-hdr.png"
    pil_img = Image.fromarray(img16, mode='RGB')
    pil_img.save(out_png, format='PNG', pnginfo=pnginfo)
    print(f"Saved HDR‑style PNG with metadata: {out_png}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python simulate_true_hdr_with_metadata.py <input.png>")
        sys.exit(1)
    simulate_true_hdr_with_metadata(sys.argv[1])
