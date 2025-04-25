import cv2
import numpy as np
import os
import sys

def make_hdr_radiance(img, exposure_factors=[0.1,1.0,1.0]):
    # simulate exposures
    exps = []
    for f in exposure_factors:
        ev = np.clip(img * f, 0, 1)
        exps.append((ev*255).astype(np.uint8))
    times = np.array(exposure_factors, dtype=np.float32)
    merge = cv2.createMergeDebevec()
    hdr = merge.process(exps, times=times.copy())
    return hdr

def tone_map_to_png(hdr, out_png,
                    method='drago',
                    gamma=2.2,
                    saturation=1.2,
                    bit_depth=16):
    """
    method: 'drago' or 'reinhard'
    bit_depth: 8 or 16
    """
    if method.lower()=='reinhard':
        tm = cv2.createTonemapReinhard(gamma=gamma, intensity=0, light_adapt=1, color_adapt=saturation)
    else:
        tm = cv2.createTonemapDrago(gamma=gamma, saturation=saturation, bias=0.85)
    ldr = tm.process(hdr)  # float32 in [0,1]
    
    # choose 8-bit or 16-bit
    if bit_depth==8:
        out = np.clip(ldr*255, 0, 255).astype(np.uint8)
    else:
        out = np.clip(ldr*(2**bit_depth-1), 0, 2**bit_depth-1).astype(np.uint16)
    
    if not cv2.imwrite(out_png, out):
        raise IOError(f"Failed to write tone-mapped PNG to {out_png}")
    print(f"Wrote tone-mapped PNG → {out_png}")

def convert(input_path):
    # load & normalize
    img = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Couldn’t load {input_path}")
    img = img.astype(np.float32)/255.0

    # 1) make HDR float map
    hdr = make_hdr_radiance(img)

    # 2a) save raw HDR & EXR if you want
    base,_ = os.path.splitext(input_path)
    hdr_path = base + ".hdr"
    exr_path = base + ".exr"
    cv2.imwrite(hdr_path, hdr)
    cv2.imwrite(exr_path, hdr)
    print(f"Saved float-HDR → {hdr_path}, {exr_path}")

    # 2b) tone-map and save as PNG
    out_png = base + "-hdr.png"
    tone_map_to_png(hdr, out_png,
                    method='drago',     # try 'reinhard' too
                    gamma=2.2,
                    saturation=1.4,
                    bit_depth=16)

if __name__=="__main__":
    if len(sys.argv)!=2:
        print("Usage: python hdr_png.py <input.png>")
        sys.exit(1)
    convert(sys.argv[1])
