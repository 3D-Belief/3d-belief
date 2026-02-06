from belief_baselines.agent.vlm.api import *
from belief_baselines.agent.vlm.openai_utils import *
from belief_baselines.agent.vlm.general_utils import _ensure_list_imgs, _save_np_to_temp_png
from typing import Any, Dict, List, Optional, Tuple, Union, Iterable
import numpy as np
from pathlib import Path
import ast, re
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen3VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from qwen_vl_utils import process_vision_info

class VLM():
    def __init__(self, vlm_model_name: str, adapter_path: str):
        # base_model_path = 'Qwen/Qwen2.5-VL-7B-Instruct'
        base_model_path = 'Qwen/Qwen3-VL-8B-Instruct'

        print("Loading base model...")
        # model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        #     base_model_path, 
        #     device_map="auto"
        # )
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            base_model_path, 
            dtype="auto",
            device_map="auto"
        )

        # Load processor from base model (not adapter)
        self.processor = AutoProcessor.from_pretrained(base_model_path)

        # # Now load the LoRA adapter on top of the base model
        # print("Loading LoRA adapter...")
        # model = PeftModel.from_pretrained(model, adapter_path)
        self.model = model.eval()

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
        image_path = _save_np_to_temp_png(image)
        image_path = str(image_path)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": text_prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to("cuda")

        generated_ids = self.model.generate(**inputs, max_new_tokens=2048)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        text_response = output_text[0]
        print(text_response)
        response = {"num_input_tokens": len(text_prompt), "num_output_tokens": len(text_response)}
        actions = re.findall(r'(?:turn_left|turn_right|move_forward|move_back)', text_response)
        print(actions)
        response["parsed"] = actions
        return response
        
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