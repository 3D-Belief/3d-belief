import os
import re
import json
from typing import Tuple, Union

import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image
import supervision as sv

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if api_key is None:
    raise RuntimeError("GOOGLE_API_KEY env var not set")

client = genai.Client(api_key=api_key)

safety_settings = [
    types.SafetySetting(
        category="HARM_CATEGORY_DANGEROUS_CONTENT",
        threshold="BLOCK_ONLY_HIGH",
    ),
]

MODEL_NAME = "gemini-2.5-flash"
TEMPERATURE = 0.5

def _empty_detections() -> sv.Detections:
    return sv.Detections(
        xyxy=np.zeros((0, 4), dtype=np.float32),
        confidence=np.zeros((0,), dtype=np.float32),
        class_id=np.zeros((0,), dtype=np.int64),
    )


def _extract_json_list(text: str):
    """
    Try to extract a JSON list from Gemini text even if it includes extra prose/code fences.
    Returns Python object (list) or None.
    """
    if not text:
        return None

    # Strip markdown code fences if present
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        candidate = fenced.group(1).strip()
        try:
            obj = json.loads(candidate)
            return obj if isinstance(obj, list) else None
        except Exception:
            pass

    # Try to find the first [...] block (best-effort)
    m = re.search(r"(\[\s*{.*}\s*\])", text, flags=re.DOTALL)
    if m:
        candidate = m.group(1)
        try:
            obj = json.loads(candidate)
            return obj if isinstance(obj, list) else None
        except Exception:
            return None

    # Last resort: try full text
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, list) else None
    except Exception:
        return None


def _safe_detections_from_json(result_text: str, resolution_wh: Tuple[int, int]) -> sv.Detections:
    """
    Build detections from a JSON list of entries with key 'box_2d'.
    This is a fallback when sv.Detections.from_vlm fails.
    We ignore masks here (metrics often only need boxes); you can extend if you need masks.
    """
    obj = _extract_json_list(result_text)
    if not obj:
        return _empty_detections()

    w, h = resolution_wh
    xyxy_list = []
    conf_list = []
    class_id_list = []

    def clamp(v, lo, hi):
        return max(lo, min(hi, v))

    for entry in obj:
        if not isinstance(entry, dict):
            continue
        box = entry.get("box_2d", None)
        if box is None:
            continue

        # Try common formats:
        # 1) {"box_2d": [x1, y1, x2, y2]}
        # 2) {"box_2d": {"x1":..., "y1":..., "x2":..., "y2":...}}
        # 3) {"box_2d": {"xmin":..., "ymin":..., "xmax":..., "ymax":...}}
        x1 = y1 = x2 = y2 = None
        if isinstance(box, (list, tuple)) and len(box) == 4:
            x1, y1, x2, y2 = box
        elif isinstance(box, dict):
            for a, b in [("x1", "y1"), ("xmin", "ymin")]:
                if a in box and b in box:
                    x1, y1 = box[a], box[b]
                    break
            for a, b in [("x2", "y2"), ("xmax", "ymax")]:
                if a in box and b in box:
                    x2, y2 = box[a], box[b]
                    break

        if x1 is None or y1 is None or x2 is None or y2 is None:
            continue

        # Sometimes models return normalized coords [0,1]; detect + scale.
        vals = np.array([x1, y1, x2, y2], dtype=np.float32)
        if np.all(np.isfinite(vals)) and vals.max() <= 1.5:
            vals[0] *= w
            vals[2] *= w
            vals[1] *= h
            vals[3] *= h

        x1, y1, x2, y2 = [float(v) for v in vals]
        x1 = clamp(x1, 0.0, float(w))
        x2 = clamp(x2, 0.0, float(w))
        y1 = clamp(y1, 0.0, float(h))
        y2 = clamp(y2, 0.0, float(h))

        # Ensure proper ordering
        if x2 <= x1 or y2 <= y1:
            continue

        xyxy_list.append([x1, y1, x2, y2])

        # Confidence: may be missing -> default 1.0
        conf = entry.get("confidence", entry.get("score", None))
        try:
            conf = float(conf)
        except Exception:
            conf = 1.0
        conf_list.append(conf)

        # Optional: map label text to a single class id (0)
        class_id_list.append(0)

    if len(xyxy_list) == 0:
        return _empty_detections()

    return sv.Detections(
        xyxy=np.asarray(xyxy_list, dtype=np.float32),
        confidence=np.asarray(conf_list, dtype=np.float32),
        class_id=np.asarray(class_id_list, dtype=np.int64),
    )


def segment_label_with_gemini(
    image_np: np.ndarray,
    label: str,
    return_annotated: bool = False,
) -> Union[sv.Detections, Tuple[sv.Detections, Image.Image]]:
    """
    Run Gemini 2.5 instance segmentation for a given label.

    Args:
        image_np: HxWx3 uint8 numpy array (RGB) OR PIL Image.
        label: text label to segment, e.g. "toilet", "motorcycle".
        return_annotated: if True, also returns a PIL Image with overlays.

    Returns:
        detections or (detections, annotated_image)
    """
    # Ensure PIL image in RGB
    if isinstance(image_np, Image.Image):
        image = image_np.convert("RGB")
    else:
        image_np = np.asarray(image_np)
        if image_np.ndim == 3 and image_np.shape[-1] == 4:
            image_np = image_np[..., :3]
        if image_np.dtype != np.uint8:
            image_np = np.clip(image_np, 0, 255).astype(np.uint8)
        image = Image.fromarray(image_np, mode="RGB")

    width, height = image.size
    target_height = max(1, int(1024 * height / max(1, width)))
    resized_image = image.resize((1024, target_height), Image.Resampling.LANCZOS)

    prompt = (
        f"Give the segmentation masks for the {label}. "
        "Output a JSON list of segmentation masks where each entry contains "
        'the 2D bounding box in the key "box_2d", '
        'the segmentation mask in key "mask", '
        'and the text label in the key "label". Use descriptive labels.'
    )

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=[prompt, resized_image],
        config=types.GenerateContentConfig(
            temperature=TEMPERATURE,
            safety_settings=safety_settings,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )

    resolution_wh = image.size

    # Primary path: use supervision parser
    try:
        detections = sv.Detections.from_vlm(
            vlm=sv.VLM.GOOGLE_GEMINI_2_5,
            result=response.text,
            resolution_wh=resolution_wh,
        )

        # Guard against the exact crash you hit: N>0 but confidence empty
        n = len(detections)
        if n > 0:
            if detections.confidence is None or np.asarray(detections.confidence).shape != (n,):
                detections = sv.Detections(
                    xyxy=np.asarray(detections.xyxy, dtype=np.float32).reshape(-1, 4),
                    confidence=np.ones((n,), dtype=np.float32),
                    class_id=(
                        np.asarray(detections.class_id, dtype=np.int64).reshape(-1)
                        if detections.class_id is not None and np.asarray(detections.class_id).shape == (n,)
                        else np.zeros((n,), dtype=np.int64)
                    ),
                    mask=getattr(detections, "mask", None),
                )
    except Exception:
        # Fallback: parse JSON ourselves; if that fails return empty
        detections = _safe_detections_from_json(response.text, resolution_wh=resolution_wh)

    if not return_annotated:
        return detections

    thickness = sv.calculate_optimal_line_thickness(resolution_wh=resolution_wh)
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=resolution_wh) / 3

    box_annotator = sv.BoxAnnotator(thickness=thickness)
    label_annotator = sv.LabelAnnotator(
        smart_position=True,
        text_color=sv.Color.BLACK,
        text_scale=text_scale,
        text_position=sv.Position.CENTER,
    )
    masks_annotator = sv.MaskAnnotator()

    annotated = image
    for annotator in (box_annotator, label_annotator, masks_annotator):
        annotated = annotator.annotate(scene=annotated, detections=detections)

    if not isinstance(annotated, Image.Image):
        annotated = Image.fromarray(annotated)

    return detections, annotated