from openai import OpenAI
from openai import AzureOpenAI
from typing import List, Dict, Any

class AzureOpenAIClient:
    def __init__(self, api_key: str, api_version: str, azure_endpoint: str, model_name: str):
        self.client = AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=azure_endpoint
        )
        self.model_name = model_name

    def run_prompt(self, messages: List[dict], max_tokens: int = 1024) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content

class OpenAIClient:
    def __init__(self, api_key: str, model_name: str):
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name

    def run_prompt(self, messages: List[dict], max_tokens: int = None, responses_api: bool = True, reasoning_effort: str = None, reasoning_summary: str = "detailed") -> str:
        if responses_api:
            reasoning_config = {"summary": reasoning_summary} if reasoning_summary is not None else {}
            if reasoning_effort is not None:
                reasoning_config["effort"] = reasoning_effort
            prompt_kwargs = dict(
                model=self.model_name,
                input=messages,
                reasoning=reasoning_config,
            )
            if max_tokens is not None:
                prompt_kwargs["max_output_tokens"] = max_tokens
            response = self.client.responses.create(**prompt_kwargs)
            response_ = {}
            response_["num_input_tokens"] = response.usage.input_tokens
            response_["num_output_tokens"] = response.usage.output_tokens
            reasoning_summary_items = [item for item in response.output if item.type == "reasoning"]
            if reasoning_summary_items and reasoning_summary_items[0].summary:
                response_["reasoning_summary"] = [summary.text for summary in reasoning_summary_items[0].summary]
            else:
                response_["reasoning_summary"] = None
            message_items = [item for item in response.output if item.type == "message"]
            assert message_items and message_items[0].content
            response_["content"] = message_items[0].content[0].text
        else:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_tokens
            )
            response_ = {}
            response_["num_input_tokens"] = response.usage.prompt_tokens
            response_["num_output_tokens"] = response.usage.completion_tokens
            response_["content"] = response.choices[0].message.content
            response_["reasoning_summary"] = None
        return response_