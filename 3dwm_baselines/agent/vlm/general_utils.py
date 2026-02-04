import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import ast
import io
import re
import tempfile
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union

def _np_to_png_bytes(
    image: np.ndarray,
    *,
    assume_bgr: bool = True,
) -> bytes:
    """
    Convert a NumPy image to PNG bytes.

    Supports:
      - HxW (grayscale)
      - HxWx3 (RGB/BGR)
      - HxWx4 (RGBA/BGRA)
    Dtypes:
      - uint8 (0..255)
      - float (assumes 0..1 or 0..255; clipped)
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f"image must be np.ndarray, got {type(image)}")

    arr = image

    # Handle floats
    if np.issubdtype(arr.dtype, np.floating):
        # Heuristic: if max <= 1.5 assume 0..1, else 0..255
        maxv = float(np.nanmax(arr)) if arr.size else 0.0
        scale = 255.0 if maxv <= 1.5 else 1.0
        arr = np.clip(arr * scale, 0.0, 255.0).astype(np.uint8)

    # Handle ints wider than uint8
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    if arr.ndim == 2:
        pil = Image.fromarray(arr, mode="L")
    elif arr.ndim == 3 and arr.shape[2] in (3, 4):
        c = arr.shape[2]
        if c == 3:
            if assume_bgr:
                arr = arr[..., ::-1]  # BGR -> RGB
            pil = Image.fromarray(arr, mode="RGB")
        else:  # 4 channels
            if assume_bgr:
                # BGRA -> RGBA
                arr = arr[..., [2, 1, 0, 3]]
            pil = Image.fromarray(arr, mode="RGBA")
    else:
        raise ValueError(f"Unsupported image shape {arr.shape}. Expected HxW, HxWx3, or HxWx4.")

    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()

def parse_response_for_bbox(response: str):
    try:
        if "Output:" in response:
            response = response.split("Output:")[1]
        res = []
        coordinates = response.split(":")
        for coordinate in coordinates:
            coordinate = coordinate.strip()[1:-1].split(',')
            res.append((int(coordinate[0]), int(coordinate[1])))
        return res
    except Exception:
        pass
    return "llm_parse_error"

def draw_bbox(image: Image.Image, bbox: list):
    # bbox is in the format of top-left and bottom-right corners: [(x1, y1), (x2, y2)]
    bbox = [bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1]]
    draw = ImageDraw.Draw(image)
    draw.rectangle(bbox, outline="red", width=4)
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.axis("off")
    plt.title("Bounding Box Overlay")
    plt.show()

def deproject_2d_to_3d(pixel_xy, depth, camera_res, camera_fov, camera_extrinsics):
	pixel_x, pixel_y = pixel_xy
	f_x = camera_res[0] / (2.0 * np.tan(np.radians(camera_fov / 2.0)))
	f_y = camera_res[1] / (2.0 * np.tan(np.radians(camera_fov / 2.0)))
	intrinsic_K = np.array([[f_x, 0.0, camera_res[0] / 2.0],
							[0.0, f_y, camera_res[1] / 2.0],
							[0.0, 0.0, 1.0]])
	K_inv = np.linalg.inv(intrinsic_K)
	P_image = np.array([pixel_x, pixel_y, 1.0], dtype=np.float64)
	P_camera = K_inv @ P_image
	P_camera = P_camera * depth
	extrinsic_inv = np.linalg.inv(camera_extrinsics)
	P_camera_h = np.array([P_camera[0], P_camera[1], P_camera[2], 1.0])
	P_world = extrinsic_inv @ P_camera_h
	return P_world[:3]

def _as_uint8(img: np.ndarray) -> np.ndarray:
    """Convert float [0,1] or [0,255] / other dtypes to uint8 safely."""
    if img.dtype == np.uint8:
        return img
    arr = img
    if np.issubdtype(arr.dtype, np.floating):
        # Heuristic: if max <= 1.0, assume [0,1]; else assume [0,255]
        maxv = float(np.nanmax(arr)) if arr.size else 1.0
        scale = 255.0 if maxv <= 1.0 else 1.0
        arr = np.clip(arr * scale, 0, 255)
    else:
        arr = np.clip(arr, 0, 255)
    return arr.astype(np.uint8)

def _to_pil(img: np.ndarray) -> Image.Image:
    """Accept HxW, HxW1, HxWx3, HxWx4 and return a PIL Image."""
    if img.ndim == 2:
        return Image.fromarray(_as_uint8(img), mode="L")
    if img.ndim == 3:
        h, w, c = img.shape
        if c == 1:
            return Image.fromarray(_as_uint8(img[..., 0]), mode="L")
        if c == 3:
            return Image.fromarray(_as_uint8(img), mode="RGB")
        if c == 4:
            return Image.fromarray(_as_uint8(img), mode="RGBA")
    raise ValueError(f"Unsupported image shape {img.shape}; expected (H,W), (H,W,1|3|4)")

def _save_np_to_temp_png(img: np.ndarray) -> Path:
    """Write np array image to a temp .png and return its Path."""
    pil = _to_pil(img)
    tmp = tempfile.NamedTemporaryFile(prefix="vlm_", suffix=".png", delete=False)
    tmp_path = Path(tmp.name)
    tmp.close()
    pil.save(tmp_path, format="PNG")
    return tmp_path

def _ensure_list_imgs(
    images: Union[np.ndarray, Iterable[np.ndarray]]
) -> List[np.ndarray]:
    if isinstance(images, np.ndarray):
        return [images]
    if isinstance(images, (list, tuple)):
        out = []
        for i, im in enumerate(images):
            if not isinstance(im, np.ndarray):
                raise TypeError(f"Element {i} is {type(im)}; expected np.ndarray")
            out.append(im)
        if not out:
            raise ValueError("Empty image list.")
        return out
    raise TypeError(f"Unsupported type {type(images)}; pass a np.ndarray or list/tuple of them")