import json
import re
from typing import List, Dict, Any, Union, Optional
import io
import os
import base64
from PIL import Image
import mimetypes
import litellm
from litellm import completion, completion_cost
from dotenv import load_dotenv

load_dotenv()

class LiteLLMWrapper:
    """Wrapper for LiteLLM to support multiple models and logging"""
    
    def __init__(
        self,
        model_name: str = "gpt-4-vision-preview",
        temperature: float = 0.7,
        print_cost: bool = False,
        verbose: bool = False,
        use_langfuse: bool = True,
    ):
        """
        Initialize the LiteLLM wrapper
        
        Args:
            model_name: Name of the model to use (e.g. "azure/gpt-4", "vertex_ai/gemini-pro")
            temperature: Temperature for completion
            print_cost: Whether to print the cost of the completion
            verbose: Whether to print verbose output
            use_langfuse: Whether to enable Langfuse logging
        """
        self.model_name = model_name
        self.temperature = temperature
        self.print_cost = print_cost
        self.verbose = verbose
        self.accumulated_cost = 0

        if self.verbose:
            os.environ['LITELLM_LOG'] = 'DEBUG'
        
        # Set langfuse callback only if enabled
        if use_langfuse:
            litellm.success_callback = ["langfuse"]
            litellm.failure_callback = ["langfuse"]

    def _encode_file(self, file_path: Union[str, Image.Image]) -> str:
        """
        Encode local file or PIL Image to base64 string
        
        Args:
            file_path: Path to local file or PIL Image object
            
        Returns:
            Base64 encoded file string
        """
        if isinstance(file_path, Image.Image):
            buffered = io.BytesIO()
            file_path.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode("utf-8")
        else:
            with open(file_path, "rb") as file:
                return base64.b64encode(file.read()).decode("utf-8")

    def _get_mime_type(self, file_path: str) -> str:
        """
        Get the MIME type of a file based on its extension
        
        Args:
            file_path: Path to the file
            
        Returns:
            MIME type as a string (e.g., "image/jpeg", "audio/mp3")
        """
        mime_type, _ = mimetypes.guess_type(file_path)
        if mime_type is None:
            raise ValueError(f"Unsupported file type: {file_path}")
        return mime_type

    def __call__(self, messages: List[Dict[str, Any]], metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Process messages and return completion
        
        Args:
            messages: List of message dictionaries with 'type' and 'content' keys
            metadata: Optional metadata to pass to litellm completion, e.g. for Langfuse tracking
        
        Returns:
            Generated text response
        """
        if metadata is None:
            print("No metadata provided, using empty metadata")
            metadata = {}
        metadata["trace_name"] = f"litellm-completion-{self.model_name}"
        # Convert messages to LiteLLM format
        formatted_messages = []
        for msg in messages:
            if msg["type"] == "text":
                formatted_messages.append({
                    "role": "user",
                    "content": [{"type": "text", "text": msg["content"]}]
                })
            elif msg["type"] in ["image", "audio", "video"]:
                # Check if content is a local file path or PIL Image
                if isinstance(msg["content"], Image.Image) or os.path.isfile(msg["content"]):
                    try:
                        if isinstance(msg["content"], Image.Image):
                            mime_type = "image/png"
                        else:
                            mime_type = self._get_mime_type(msg["content"])
                        base64_data = self._encode_file(msg["content"])
                        data_url = f"data:{mime_type};base64,{base64_data}"
                    except ValueError as e:
                        print(f"Error processing file {msg['content']}: {e}")
                        continue
                else:
                    data_url = msg["content"]
                
                # Append the formatted message based on the model
                if "gemini" in self.model_name:
                    formatted_messages.append({
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": data_url
                            }
                        ]
                    })
                elif "gpt" in self.model_name:
                    # GPT and other models expect a different format
                    if msg["type"] == "image":
                        # Default format for images and videos in GPT
                        formatted_messages.append({
                            "role": "user",
                            "content": [
                                {
                                    "type": f"image_url", 
                                    f"{msg['type']}_url": {
                                        "url": data_url,
                                        "detail": "high"  
                                    }
                                }
                            ]
                        })
                    else:
                        raise ValueError("For GPT, only text and image inferencing are supported")
                else:
                    raise ValueError("Only support Gemini and Gpt for Multimodal capability now")

        try:
            # LiteLLM 1.60.x predates the MiniMax provider. Use its supported
            # OpenAI-compatible route for the current text models only.
            if self.model_name in {
                "minimax/MiniMax-M3.1-Flash", "minimax/MiniMax-M3"
            }:
                api_key = os.environ.get("MINIMAX_API_KEY")
                if not api_key:
                    raise ValueError("MINIMAX_API_KEY is required for MiniMax text models")
                response = completion(
                    model="openai/" + self.model_name.split("/", 1)[1],
                    api_base=os.environ.get("MINIMAX_API_BASE") or "https://api.minimax.io/v1",
                    api_key=api_key,
                    messages=formatted_messages,
                    # MiniMax requires temperature > 0; omit the evaluator's 0.0
                    # to use the provider default instead.
                    temperature=self.temperature if self.temperature > 0 else None,
                    metadata=metadata,
                    max_retries=99,
                )
            # if it's openai o series model, set temperature to None and reasoning_effort to "medium"
            elif (re.match(r"^o\d+.*$", self.model_name) or re.match(r"^openai/o.*$", self.model_name)):
                self.temperature = None
                self.reasoning_effort = "medium"
                response = completion(
                    model=self.model_name,
                    messages=formatted_messages,
                    temperature=self.temperature,
                    reasoning_effort=self.reasoning_effort,
                    metadata=metadata,
                    max_retries=99
                )
            else:
                response = completion(
                    model=self.model_name,
                    messages=formatted_messages,
                    temperature=self.temperature,
                    metadata=metadata,
                    max_retries=99
                )
            if self.print_cost:
                # pass your response from completion to completion_cost
                cost = completion_cost(completion_response=response)
                formatted_string = f"Cost: ${float(cost):.10f}"
                # print(formatted_string)
                self.accumulated_cost += cost
                print(f"Accumulated Cost: ${self.accumulated_cost:.10f}")
                
            content = response.choices[0].message.content
            if content is None:
                print(f"Got null response from model. Full response: {response}")
            return content
        
        except Exception as e:
            print(f"Error in model completion: {e}")
            return str(e)
        
if __name__ == "__main__":
    pass