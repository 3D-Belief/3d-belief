from belief_baselines.agent.vlm.api import *
from belief_baselines.agent.vlm.openai_utils import *
from belief_baselines.agent.vlm.general_utils import _ensure_list_imgs, _save_np_to_temp_png, _np_to_png_bytes
from typing import Any, Dict, List, Optional, Tuple, Union, Iterable, Sequence

import ast, re, io
from PIL import Image
import numpy as np
from pathlib import Path
from dataclasses import dataclass

from google import genai
from google.genai import types

class VLM():
    def __init__(self, vlm_model_name: str):
        api_credentials = get_openai_api_credentials()
        self.vlm = OpenAIClient(*api_credentials, model_name=vlm_model_name)
        self._temp_paths: List[Path] = []  # track temp files to clean up

    def __del__(self):
        # best-effort cleanup of temp files
        for p in getattr(self, "_temp_paths", []):
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass

    def prompt_score_obj_image(self, image: np.ndarray, object_name: str) -> bool:
        """
        Accept a single NumPy image (H x W [x C]) and return True/False for presence.
        """
        img_path = _save_np_to_temp_png(image)
        self._temp_paths.append(img_path)

        user_prompt_parts: List[Union[str, Path]] = []

        intro_text = (
            "You are helping an agent to perform an object searching task in a household environment.\n"
            "You are given a target object name and an image that may or may not contain that object.\n"
            "Your job is to: \n"
            "determine whether the target object is present in the image, output 1 if you can see "
            "that object in the image, otherwise output 0\n"
            "Example: If the target object is \"refrigerator\", and one image shows a kitchen countertop with a refrigerator beside "
            "it, you should output 1 for presence. If another image shows a kitchen with countertop with a sink but "
            "no refrigerator, you should output 0. The third shows a bedroom with a bed, you should "
            "output 0 since there is no refrigerator.\n\n"
            "Example output: 1\n"
            "Do not include any extra commentary after the answer.\n\n"
            "Now the task begins.\n"
            f"The target object is: \"{object_name}\"\n\n"
            "The image is shown below:"
        )
        user_prompt_parts.append(intro_text)

        user_prompt_parts.append("\nImage:")
        user_prompt_parts.append(img_path)

        final_text = (
            "After reasoning, output your final answer as 0 or 1.\n"
            "Do not include any extra commentary after the answer."
        )
        user_prompt_parts.append(final_text)

        prompt = build_interleaved_prompt_openai(
            system_prompt=(
                "You are a visual reasoning assistant helping to do object recognition and "
                "object semantic inference given images.\n"
            ),
            user_prompt_parts=user_prompt_parts,
        )

        response = self.vlm.run_prompt(messages=prompt, max_tokens=1024)
        text_response = response["content"]

        try:
            res = int(str(text_response).strip())
            response["parsed"] = bool(res)
            return response
        except Exception:
            print(f"Failed to parse response '{text_response}', returning fallback 0")
            return {"parsed": False, "num_input_tokens": response.get("num_input_tokens", 0), "num_output_tokens": response.get("num_output_tokens", 0)}

    def prompt_score_obj_images(
        self,
        images: Union[np.ndarray, Iterable[np.ndarray]],
        object_name: str,
    ) -> List[Tuple[int, int]]:
        """
        Accept a NumPy image or a list/tuple of NumPy images.
        Returns a list of (presence, score) per image as integers.
        """
        imgs = _ensure_list_imgs(images)

        # Convert to temp files and keep order
        frame_files: List[Path] = []
        for im in imgs:
            p = _save_np_to_temp_png(im)
            self._temp_paths.append(p)
            frame_files.append(p)

        user_prompt_parts: List[Union[str, Path]] = []

        intro_text = (
            "You are helping an agent to perform an object searching task in a household environment.\n"
            "You are given a target object name and several images that may or may not contain that object.\n"
            "Your job is to: \n"
            "A. determine whether the target object is present in each image, output 1 if you can see "
            "that object in the image, otherwise output 0; \n"
            "B. give a score 1-5 based on how much the scene shown in the image is semantically related to the target object, "
            "5 means you can directly see the object; 4 means you see a scene that is highly related to the object, and a region that "
            "may be part of the object but you can't be sure; 3 means you can't directly see it, but the semantics strongly "
            "relate to the object so it may be nearby; 2 means fairly related; 1 means weakly related; 0 means the scene is "
            "not related to the object at all.\n"
            "Example: If the target object is \"refrigerator\", and one image shows a kitchen countertop with a refrigerator beside "
            "it, you should output 1 for presence and 5 for score. If another image shows a kitchen with countertop with a sink but "
            "no refrigerator, you should output 0 for presence and 3 for score. The third shows a bedroom with a bed, you should "
            "output 0 for presence and 0 for score.\n\n"
            "Write your answer as a list of tuples, each tuple for one image as (presence, score).\n"
            "Example: [(1, 5), (0, 3)]\n"
            "Do not include any extra commentary after the answer.\n\n"
            "Now the task begins.\n"
            f"The target object is: \"{object_name}\"\n\n"
            "The images are as follows:"
        )
        user_prompt_parts.append(intro_text)

        user_prompt_parts.append("\nImages:")
        user_prompt_parts.extend(frame_files)

        final_text = (
            "After reasoning, output your final answer for each image.\n"
            "Write your answer as a list of tuples, each tuple for one image as (presence, score).\n"
            "Do not include any extra commentary after the answer."
        )
        user_prompt_parts.append(final_text)

        prompt = build_interleaved_prompt_openai(
            system_prompt=(
                "You are a visual reasoning assistant helping to do object recognition and "
                "object semantic inference given images.\n"
            ),
            user_prompt_parts=user_prompt_parts,
        )

        response = self.vlm.run_prompt(messages=prompt, max_tokens=None)
        text_response = response["content"]

        try:
            tup_list = ast.literal_eval(str(text_response).strip())
            assert isinstance(tup_list, list) and all(isinstance(t, tuple) and len(t) == 2 for t in tup_list)
            # pad/truncate to match number of images
            if len(tup_list) < len(frame_files):
                tup_list.extend([(0, 0)] * (len(frame_files) - len(tup_list)))
            elif len(tup_list) > len(frame_files):
                tup_list = tup_list[: len(frame_files)]
            # ensure ints
            tup_list = [(int(p), int(s)) for (p, s) in tup_list]
            response["parsed"] = tup_list
            return response
        except Exception as e:
            print(f"Failed to parse response '{text_response}' ({e}), returning all zeros")
            return {"parsed": [(0, 0)] * len(frame_files), "num_input_tokens": response.get("num_input_tokens", 0), "num_output_tokens": response.get("num_output_tokens", 0)}
        
    def prompt_predict_actions(
        self,
        image: np.ndarray,
        text_prompt: str,
    ) -> Dict[str, Any]:
        """
        Accept a NumPy image or a list/tuple of NumPy images.
        Returns a dict with VLM response and parsed action.
        """
        img_path = _save_np_to_temp_png(image)
        self._temp_paths.append(img_path)

        user_prompt_parts: List[Union[str, Path]] = []

        user_prompt_parts.append(text_prompt)

        user_prompt_parts.append("\nCurrent observation:")
        user_prompt_parts.append(img_path)

        prompt = build_interleaved_prompt_openai(
            system_prompt=(
                "You are an embodied assistant, your goal is to find a set of next optimal actions "
                "for an agent performing an object searching task in a household environment.\n"
            ),
            user_prompt_parts=user_prompt_parts,
        )

        try:
            response = self.vlm.run_prompt(messages=prompt, max_tokens=None)
        except Exception as e:
            print(f"Failed to get response from VLM ({e}), returning empty action list")
            return {"parsed": [], "num_input_tokens": 0, "num_output_tokens": 0}
        text_response = response["content"]
        # parse a list of actions from the response text (e.g., ["turn_left", "move_forward"])
        try:
            actions = re.findall(r'\b(?:turn_left|turn_right|move_forward|move_back)\b', text_response)
            response["parsed"] = actions
            return response
        except Exception as e:
            print(f"Failed to parse response '{text_response}' ({e}), returning empty action list")
            return {"parsed": [], "num_input_tokens": response.get("num_input_tokens", 0), "num_output_tokens": response.get("num_output_tokens", 0)}
        
    def prompt_furniture_in_image(self, image: np.ndarray, object_list: List[str]) -> bool:
        """
        Accept a single NumPy image (H x W [x C]) and return True/False for presence.
        """
        img_path = _save_np_to_temp_png(image)
        self._temp_paths.append(img_path)

        user_prompt_parts: List[Union[str, Path]] = []

        objects = ", ".join(object_list)

        intro_text = (
            "You are helping an agent to recognize furniture/ appliance objects in a household environment.\n"
            "You are given a list of object names and an image that may contain some furniture/ appliance objects in the list.\n"
            "However, some objects in the list may not be furniture or appliances.\n"
            "Your job is to: \n"
            "1. Recognize which objects in the list are furniture/appliance objects, output a list of recognized furniture/appliance objects;\n"
            "2. Determine whether any of the recognized furniture/appliance objects are present in the image, output a list of present furniture/appliance objects;\n"
            "3. Recognize any other furniture/appliance objects in the image that are not in the given list, output a list of extra furniture/appliance objects (output empty list if none).\n"
            "Example: If the object list is [\"sofa\", \"table\", \"lamp\", \"book\", \"refrigerator\"], you should first recognize "
            "which of these are furniture/appliance objects, then determine which of those recognized objects are present in the image, then recognize "
            "any other furniture/appliance objects in the image that are not in the given list, in total as three lists.\n"
            "Example output: \n"
            "[\"sofa\", \"table\", \"lamp\", \"refrigerator\"]; [\"sofa\", \"lamp\"]; [\"countertop\"]\n"
            "Do not include any extra commentary after the answer.\n\n"
            "Now the task begins.\n"
            f"The target object is: \"[{objects}]\"\n\n"
            "The image is shown below:"
        )
        user_prompt_parts.append(intro_text)

        user_prompt_parts.append("\nImage:")
        user_prompt_parts.append(img_path)

        final_text = (
            "After reasoning, output your final answer as three lists, separated by a semicolon and a space.\n"
            "Do not include any extra commentary after the answer."
        )
        user_prompt_parts.append(final_text)

        prompt = build_interleaved_prompt_openai(
            system_prompt=(
                "You are a visual reasoning assistant helping to do object recognition and "
                "object semantic inference given images.\n"
            ),
            user_prompt_parts=user_prompt_parts,
        )

        response = self.vlm.run_prompt(messages=prompt, max_tokens=1024)
        text_response = response["content"]

        try:
            # parse three lists from the response text
            lists = re.findall(r'\[.*?\]', text_response)
            assert len(lists) >= 3, "Expected at least three lists in the response"
            furniture_list = ast.literal_eval(lists[0])
            present_list = ast.literal_eval(lists[1])
            extra_list = ast.literal_eval(lists[2])
            response["parsed"] = {"furniture": furniture_list, "present": present_list, "extra": extra_list}
            return response
        except Exception:
            print(f"Failed to parse response '{text_response}', returning fallback 0")
            return {"parsed": {"furniture": [], "present": [], "extra": []}, "num_input_tokens": response.get("num_input_tokens", 0), "num_output_tokens": response.get("num_output_tokens", 0)}

class GeminiVLM:
    """
    Minimal Gemini VLM wrapper.

    Auth:
      - Gemini Developer API: set env GEMINI_API_KEY or pass api_key=...
    """

    def __init__(
        self,
        vlm_model_name: str = "gemini-2.5-flash",
        api_key: Optional[str] = None,
    ) -> None:
        self.model_name = vlm_model_name
        self.client = genai.Client(api_key=api_key)
    
    def _usage(self, resp: Any) -> Tuple[int, int]:
        """Best-effort token accounting for python-genai responses."""
        num_in = 0
        num_out = 0
        um = getattr(resp, "usage_metadata", None)
        if um is not None:
            num_in = int(getattr(um, "prompt_token_count", 0) or 0)
            num_out = int(getattr(um, "candidates_token_count", 0) or 0)
        return num_in, num_out

    def prompt_predict_actions(
        self,
        image: Union[np.ndarray, Sequence[np.ndarray]],
        text_prompt: str,
    ) -> Dict[str, Any]:
        """
        Accept a NumPy image or a list/tuple of NumPy images.
        Returns a dict with VLM response and parsed action.
        """
        try:
            images: List[np.ndarray] = list(image) if isinstance(image, (list, tuple)) else [image]

            parts: List[Any] = []
            parts.append(text_prompt)
            parts.append("\nCurrent observation:")

            # Interleave all images (Gemini supports multiple image parts)
            for im in images:
                png_bytes = _np_to_png_bytes(im, assume_bgr=True)
                parts.append(types.Part.from_bytes(data=png_bytes, mime_type="image/png"))

            resp = self.client.models.generate_content(
                model=self.model_name,
                contents=parts
            )

            text_response = (resp.text or "").strip()

            # Token accounting
            num_in = 0
            num_out = 0
            um = getattr(resp, "usage_metadata", None)
            if um is not None:
                # python-genai uses snake_case fields
                num_in = int(getattr(um, "prompt_token_count", 0) or 0)
                num_out = int(getattr(um, "candidates_token_count", 0) or 0)

            # Parse a list of actions from the response text
            actions = re.findall(r"\b(?:turn_left|turn_right|move_forward|move_back)\b", text_response)

            return {
                "content": text_response,
                "parsed": actions,
                "num_input_tokens": num_in,
                "num_output_tokens": num_out,
            }

        except Exception as e:
            print(f"Failed to get response from VLM ({e}), returning empty action list")
            return {"parsed": [], "num_input_tokens": 0, "num_output_tokens": 0, "content": ""}

    def prompt_score_obj_image(self, image: np.ndarray, object_name: str) -> Dict[str, Any]:
        """
        Accept a single NumPy image (H x W x C) and return dict:
          - content: raw model text
          - parsed: bool (True if present)
          - num_input_tokens / num_output_tokens
        """
        intro_text = (
            "You are helping an agent to perform an object searching task in a household environment.\n"
            "You are given a target object name and an image that may or may not contain that object.\n"
            "Your job is to: \n"
            "determine whether the target object is present in the image, output 1 if you can see "
            "that object in the image, otherwise output 0\n"
            "Example: If the target object is \"refrigerator\", and one image shows a kitchen countertop with a refrigerator beside "
            "it, you should output 1 for presence. If another image shows a kitchen with countertop with a sink but "
            "no refrigerator, you should output 0. The third shows a bedroom with a bed, you should "
            "output 0 since there is no refrigerator.\n\n"
            "Example output: 1\n"
            "Do not include any extra commentary after the answer.\n\n"
            "Now the task begins.\n"
            f"The target object is: \"{object_name}\"\n\n"
            "The image is shown below:"
        )
        final_text = "After reasoning, output your final answer as 0 or 1. No extra text."

        try:
            png_bytes = _np_to_png_bytes(image, assume_bgr=True)
            parts: List[Any] = [
                intro_text,
                "\nImage:",
                types.Part.from_bytes(data=png_bytes, mime_type="image/png"),
                "\n",
                final_text,
            ]

            resp = self.client.models.generate_content(
                model=self.model_name,
                contents=parts,
            )
            text_response = (resp.text or "").strip()
            num_in, num_out = self._usage(resp)

            # Robust parse: look for first 0/1 token in the output
            m = re.search(r"\b([01])\b", text_response)
            if m is None:
                # fallback: accept things like "1." or "0," etc.
                m = re.search(r"([01])", text_response)

            parsed = bool(int(m.group(1))) if m else False

            return {
                "content": text_response,
                "parsed": parsed,
                "num_input_tokens": num_in,
                "num_output_tokens": num_out,
            }

        except Exception as e:
            print(f"Failed to get response from Gemini VLM ({e}), returning fallback 0")
            return {"content": "", "parsed": False, "num_input_tokens": 0, "num_output_tokens": 0}

    def prompt_score_obj_images(
        self,
        images: Union[np.ndarray, Iterable[np.ndarray]],
        object_name: str,
    ) -> Dict[str, Any]:
        """
        Accept a NumPy image or a list/iterable of NumPy images.
        Returns dict:
          - content: raw model text
          - parsed: List[Tuple[int,int]] (presence, score) per image
          - num_input_tokens / num_output_tokens

        Presence: 0/1
        Score: 0-5
        """
        imgs = _ensure_list_imgs(images)

        intro_text = (
            "You are helping an agent to perform an object searching task in a household environment.\n"
            "You are given a target object name and several images that may or may not contain that object.\n"
            "Your job is to: \n"
            "A. determine whether the target object is present in each image, output 1 if you can see "
            "that object in the image, otherwise output 0; \n"
            "B. give a score 1-5 based on how much the scene shown in the image is semantically related to the target object, "
            "5 means you can directly see the object; 4 means you see a scene that is highly related to the object, and a region that "
            "may be part of the object but you can't be sure; 3 means you can't directly see it, but the semantics strongly "
            "relate to the object so it may be nearby; 2 means fairly related; 1 means weakly related; 0 means the scene is "
            "not related to the object at all.\n"
            "Example: If the target object is \"refrigerator\", and one image shows a kitchen countertop with a refrigerator beside "
            "it, you should output 1 for presence and 5 for score. If another image shows a kitchen with countertop with a sink but "
            "no refrigerator, you should output 0 for presence and 3 for score. The third shows a bedroom with a bed, you should "
            "output 0 for presence and 0 for score.\n\n"
            "Write your answer as a list of tuples, each tuple for one image as (presence, score).\n"
            "Example: [(1, 5), (0, 3)]\n"
            "Do not include any extra commentary after the answer.\n\n"
            "Now the task begins.\n"
            f"The target object is: \"{object_name}\"\n\n"
            "The images are as follows:"
        )
        final_text = (
            "After reasoning, output your final answer for each image.\n"
            "Return ONLY a Python list of tuples (presence, score) with the same length as the number of images."
        )

        try:
            parts: List[Any] = [intro_text, "\nImages (in order):\n"]

            for idx, im in enumerate(imgs):
                png_bytes = _np_to_png_bytes(im, assume_bgr=True)
                parts.append(f"\nImage {idx}:\n")
                parts.append(types.Part.from_bytes(data=png_bytes, mime_type="image/png"))

            parts.append("\n")
            parts.append(final_text)

            resp = self.client.models.generate_content(
                model=self.model_name,
                contents=parts,
            )

            text_response = (resp.text or "").strip()
            num_in, num_out = self._usage(resp)

            # Parse list of tuples
            parsed_list: List[Tuple[int, int]]
            try:
                tup_list = ast.literal_eval(text_response)
                assert isinstance(tup_list, list)
                assert all(isinstance(t, tuple) and len(t) == 2 for t in tup_list)
                tup_list = [(int(p), int(s)) for (p, s) in tup_list]
            except Exception:
                # Fallback: try to extract tuples from messy outputs
                found = re.findall(r"\(\s*([01])\s*,\s*([0-5])\s*\)", text_response)
                tup_list = [(int(p), int(s)) for (p, s) in found]

            # Pad/truncate to match number of images
            if len(tup_list) < len(imgs):
                tup_list.extend([(0, 0)] * (len(imgs) - len(tup_list)))
            elif len(tup_list) > len(imgs):
                tup_list = tup_list[: len(imgs)]

            # Clamp to valid ranges just in case
            tup_list = [(1 if p else 0, max(0, min(5, s))) for (p, s) in tup_list]

            return {
                "content": text_response,
                "parsed": tup_list,
                "num_input_tokens": num_in,
                "num_output_tokens": num_out,
            }

        except Exception as e:
            print(f"Failed to get response from Gemini VLM ({e}), returning all zeros")
            return {
                "content": "",
                "parsed": [(0, 0)] * len(imgs),
                "num_input_tokens": 0,
                "num_output_tokens": 0,
            }