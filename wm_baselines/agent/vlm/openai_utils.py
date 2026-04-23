from typing import List, Union
from pathlib import Path
import base64
import json
import os

def get_openai_api_credentials():
    try:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Missing OpenAI API key in environment variables.")
        return [api_key]
    except Exception as e:
        raise Exception(f"An error occurred: {e}")

def get_azure_api_credentials():
    try:
        with open(".api.json", 'r') as file:
            credentials = json.load(file)
            api_key = os.environ.get("AZURE_OPENAI_API_KEY")
            api_version = credentials.get("api_version")
            azure_endpoint = credentials.get("azure_endpoint")
            if not all([api_key, api_version, azure_endpoint]):
                raise ValueError("Missing required API credentials.")
            return api_key, api_version, azure_endpoint
    except FileNotFoundError:
        raise FileNotFoundError("Credentials file not found. Please create '.api.json'.")
    except json.JSONDecodeError:
        raise ValueError("Error decoding JSON from the credentials file.")
    except Exception as e:
        raise Exception(f"An error occurred: {e}")

def encode_image_to_data_url(image_path: Path) -> str:
    image_path = str(image_path)
    mime = "image/png" if image_path.endswith(".png") else "image/jpeg"
    with open(image_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{data}"

def build_interleaved_prompt(system_prompt, user_prompt_parts: List[Union[str, Path]], image_detail="high") -> List[dict]:
    user_prompt = []
    for part in user_prompt_parts:
        if isinstance(part, Path):
            if part.suffix.lower() in (".jpg", ".jpeg", ".png"):
                url = encode_image_to_data_url(part)
                user_prompt.append({
                    "type": "image_url",
                    "image_url": {"url": url, "detail": image_detail}
                })
            else:
                raise ValueError(f"Unsupported image format: {part.suffix}")
        elif isinstance(part, str):
            user_prompt.append({
                "type": "text",
                "text": str(part)
            })
        else:
            raise ValueError(f"Unsupported prompt part: {part}")
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}] if system_prompt else [{"role": "user", "content": user_prompt}]

def build_interleaved_prompt_openai(system_prompt, user_prompt_parts: List[Union[str, Path]], image_detail="high", responses_api=True) -> List[dict]:
    user_prompt = []
    if responses_api:
        for part in user_prompt_parts:
            if isinstance(part, Path):
                assert part.exists(), f"Image file {part} does not exist."
                if part.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    url = encode_image_to_data_url(part)
                    user_prompt.append({
                        "type": "input_image",
                        "image_url": url
                    })
                else:
                    raise ValueError(f"Unsupported image format: {part.suffix}")
            elif isinstance(part, str):
                user_prompt.append({
                    "type": "input_text",
                    "text": str(part)
                })
            else:
                raise ValueError(f"Unsupported prompt part: {part}")
    else:
        for part in user_prompt_parts:
            if isinstance(part, Path):
                assert part.exists(), f"Image file {part} does not exist."
                if part.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    url = encode_image_to_data_url(part)
                    user_prompt.append({
                        "type": "image_url",
                        "image_url": {"url": url, "detail": image_detail}
                    })
                else:
                    raise ValueError(f"Unsupported image format: {part.suffix}")
            elif isinstance(part, str):
                user_prompt.append({
                    "type": "text",
                    "text": str(part)
                })
            else:
                raise ValueError(f"Unsupported prompt part: {part}")
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}] if system_prompt else [{"role": "user", "content": user_prompt}]
