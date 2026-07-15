import os
import sys

import numpy as np
from PIL import Image
from rembg import remove, new_session
from scipy import ndimage

DEFAULT_FOLDER = "rgbd_photos/try10_rgb"
MODEL = "u2net"

# The pear sits on a turntable inside a photo light box. rembg's saliency
# model treats the whole box as foreground (it's salient against the black
# surroundings), so its raw mask isn't pear-specific. The pear is the only
# warm-toned object in the scene (box walls/floor are neutral gray/white),
# so we refine rembg's foreground mask down to the pear with a warm-color
# filter and keep only the largest resulting blob.
WARM_RB_DIFF = 15


def isolate_pear_mask(image, rembg_mask):
    arr = np.array(image.convert("RGB")).astype(int)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    warm = (r - b > WARM_RB_DIFF) & (r > g)

    candidate = warm & (rembg_mask > 128)
    candidate = ndimage.binary_closing(candidate, structure=np.ones((5, 5)))
    candidate = ndimage.binary_opening(candidate, structure=np.ones((3, 3)))

    labeled, num = ndimage.label(candidate)
    if num == 0:
        return np.zeros_like(candidate, dtype=np.uint8)

    sizes = ndimage.sum(candidate, labeled, range(1, num + 1))
    largest = np.argmax(sizes) + 1
    pear_mask = labeled == largest

    return (pear_mask * 255).astype(np.uint8)


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_FOLDER
    out_dir = os.path.join(folder, "rembg_out")
    os.makedirs(out_dir, exist_ok=True)

    files = sorted(
        f for f in os.listdir(folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )
    if not files:
        raise FileNotFoundError(f"No images found in {folder}")

    session = new_session(MODEL)

    for fname in files:
        src_path = os.path.join(folder, fname)
        img = Image.open(src_path)

        rembg_rgba = remove(img, session=session)
        rembg_mask = np.array(rembg_rgba.split()[-1])

        pear_mask = isolate_pear_mask(img, rembg_mask)
        ys, xs = np.where(pear_mask > 0)

        stem = os.path.splitext(fname)[0]

        mask_path = os.path.join(out_dir, f"{stem}_mask.png")
        Image.fromarray(pear_mask).save(mask_path)

        cutout = np.array(img.convert("RGB"))
        rgba = np.dstack([cutout, pear_mask])
        cutout_path = os.path.join(out_dir, f"{stem}_cutout.png")
        Image.fromarray(rgba, mode="RGBA").save(cutout_path)

        if len(xs):
            bbox = (xs.min(), ys.min(), xs.max(), ys.max())
            print(f"{fname}: bbox={bbox} -> {cutout_path}, {mask_path}")
        else:
            print(f"{fname}: no pear detected -> {cutout_path}, {mask_path}")


if __name__ == "__main__":
    main()
