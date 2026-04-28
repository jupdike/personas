"""Combine an RGB image with a face-parse classification mask into an RGBA PNG.

The mask is a single-channel image whose pixel values are class integers from a
face-parsing model. Edit ``combine()`` to map class -> alpha; right now it just
writes alpha=255 everywhere so you can verify the wiring.
"""
import argparse

from PIL import Image

CLASS_BACKGROUND = 0
CLASS_SKIN = 1
CLASS_L_BROW = 2
CLASS_R_BROW = 3
CLASS_L_EYE = 4
CLASS_R_EYE = 5
CLASS_GLASSES = 6
CLASS_L_EAR = 7
CLASS_R_EAR = 8
CLASS_EARRING = 9
CLASS_NOSE = 10
CLASS_MOUTH = 11
CLASS_U_LIP = 12
CLASS_L_LIP = 13
CLASS_NECK = 14
CLASS_NECKLACE = 15
CLASS_CLOTH = 16
CLASS_HAIR = 17
CLASS_HAT = 18

exclude = { CLASS_BACKGROUND, CLASS_CLOTH, CLASS_HAT, CLASS_NECKLACE, CLASS_EARRING, CLASS_GLASSES}

def combine(rgb: Image.Image, mask: Image.Image) -> Image.Image:
    if rgb.size != mask.size:
        raise SystemExit(f"size mismatch: rgb={rgb.size} mask={mask.size}")
    rgb = rgb.convert("RGB")
    mask = mask.convert("L")
    w, h = rgb.size
    rgb_px = rgb.load()
    mask_px = mask.load()
    out = Image.new("RGBA", (w, h))
    out_px = out.load()
    for y in range(h):
        for x in range(w):
            r, g, b = rgb_px[x, y]
            cls = mask_px[x, y]
            a = 0 if cls in exclude else 255
            out_px[x, y] = (r, g, b, a)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("rgb_path", help="RGB source PNG")
    p.add_argument("mask_path", help="Face-parse classification PNG (grayscale)")
    p.add_argument("out_path", help="Output RGBA PNG")
    args = p.parse_args()

    rgb = Image.open(args.rgb_path)
    mask = Image.open(args.mask_path)
    out = combine(rgb, mask)
    out.save(args.out_path)


if __name__ == "__main__":
    main()
