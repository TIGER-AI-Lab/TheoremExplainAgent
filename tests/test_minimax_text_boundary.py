"""Offline tests using real LiteLLM/OpenAI serialization, never live inference.

Run with the versions from requirements.txt. Only HTTP transport is replaced
for the request tests; legacy passthrough is separately a wrapper unit test.
"""
import argparse
import ast
import json
import os
from pathlib import Path
import socket
import unittest
from unittest.mock import patch

os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
import httpx
import litellm

# Do not load developer credentials from .env during test imports.
with patch("dotenv.load_dotenv"):
    from mllm_tools.litellm import LiteLLMWrapper


class MiniMaxTextBoundaryTest(unittest.TestCase):
    def enterContext(self, context):
        value = context.__enter__()
        self.addCleanup(context.__exit__, None, None, None)
        return value

    def setUp(self):
        self.requests = []
        self.enterContext(patch.dict(os.environ, {
            "MINIMAX_API_KEY": "fixture-minimax-key",
            "OPENAI_API_KEY": "fixture-openai-key",
        }, clear=True))
        self.enterContext(patch.object(socket.socket, "connect", side_effect=AssertionError("Network forbidden")))
        self.enterContext(patch.object(httpx.HTTPTransport, "handle_request", self.capture))
        litellm.success_callback = []
        litellm.failure_callback = []
        litellm.telemetry = False

    def capture(self, request):
        self.requests.append(request)
        return httpx.Response(200, request=request, json={
            "id": "chatcmpl-fixture", "object": "chat.completion", "created": 1,
            "model": json.loads(request.content)["model"],
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        })

    def call(self, model, messages=None, temperature=0.0):
        wrapper = LiteLLMWrapper(model_name=model, temperature=temperature, use_langfuse=False)
        result = wrapper(messages or [{"type": "text", "content": "hello"}], metadata={})
        self.assertEqual(result, "ok")
        request = self.requests[-1]
        self.assertEqual(request.method, "POST")
        return request, json.loads(request.content)

    def test_catalog_order(self):
        catalog = json.loads((Path(__file__).resolve().parents[1] / "src/utils/allowed_models.json").read_text())
        self.assertEqual([m for m in catalog["allowed_models"] if m.startswith("minimax/")],
                         ["minimax/MiniMax-M3.1-Flash", "minimax/MiniMax-M3"])

    def test_evaluation_cli_default_reaches_http(self):
        # Execute the actual parser statements without importing media/ML stacks.
        root = Path(__file__).resolve().parents[1]
        tree = ast.parse((root / "evaluate.py").read_text())
        default = next(n for n in tree.body if isinstance(n, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == "DEFAULT_TEXT_MODEL" for t in n.targets))
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
        statements = [default]
        for node in main.body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "args" for t in node.targets):
                break
            statements.append(node)
        namespace = {"argparse": argparse, "ALLOWED_MODELS": json.loads((root / "src/utils/allowed_models.json").read_text())["allowed_models"]}
        exec(compile(ast.Module(body=statements, type_ignores=[]), "evaluate.py", "exec"), namespace)
        args = namespace["parser"].parse_args(["--file_path", "fixture.txt", "--output_folder", "unused", "--eval_type", "text"])
        _, body = self.call(args.model_text)
        self.assertEqual(body["model"], "MiniMax-M3.1-Flash")
        self.assertEqual(args.model_image, "azure/gpt-4o")
        self.assertEqual(args.model_video, "gemini/gemini-1.5-pro-002")

    def test_flash_actual_http_serialization(self):
        request, body = self.call("minimax/MiniMax-M3.1-Flash")
        self.assertEqual(str(request.url), "https://api.minimax.io/v1/chat/completions")
        self.assertEqual(request.headers["authorization"], "Bearer fixture-minimax-key")
        self.assertEqual(body["model"], "MiniMax-M3.1-Flash")
        self.assertEqual(body["messages"], [{"role": "user", "content": [{"type": "text", "text": "hello"}]}])
        for key in ("temperature", "thinking", "reasoning_effort", "service_tier", "reasoning_split"):
            self.assertNotIn(key, body)  # Provider defaults: adaptive thinking/max effort.

    def test_m3_preserves_explicit_model_and_temperature(self):
        _, body = self.call("minimax/MiniMax-M3", temperature=0.7)
        self.assertEqual(body["model"], "MiniMax-M3")
        self.assertEqual(body["temperature"], 0.7)

    def test_custom_and_regional_endpoint_overrides(self):
        for base in ("https://api.minimaxi.com/v1", "https://custom.example/compatible/v1"):
            with self.subTest(base=base), patch.dict(os.environ, {"MINIMAX_API_BASE": base}):
                request, body = self.call("minimax/MiniMax-M3.1-Flash")
                self.assertEqual(str(request.url), base + "/chat/completions")
                self.assertEqual(body["model"], "MiniMax-M3.1-Flash")

    def test_missing_minimax_key_never_falls_back_to_openai(self):
        os.environ.pop("MINIMAX_API_KEY")
        wrapper = LiteLLMWrapper(model_name="minimax/MiniMax-M3", use_langfuse=False)
        result = wrapper([{"type": "text", "content": "hello"}], metadata={})
        self.assertIn("MINIMAX_API_KEY is required", result)
        self.assertEqual(self.requests, [])

    def test_non_minimax_image_payload_is_preserved(self):
        request, body = self.call("gpt-4o", [{"type": "image", "content": "https://example.com/image.png"}])
        self.assertEqual(str(request.url), "https://api.openai.com/v1/chat/completions")
        self.assertEqual(body["model"], "gpt-4o")
        self.assertEqual(body["temperature"], 0.0)
        self.assertEqual(body["messages"][0]["content"], [{"type": "image_url", "image_url": {"url": "https://example.com/image.png", "detail": "high"}}])

    def test_legacy_and_custom_ids_are_passed_through_unit_only(self):
        for model in ("minimax/MiniMax-M2.7", "openai/custom-saved-model"):
            with self.subTest(model=model), patch("mllm_tools.litellm.completion") as completion:
                LiteLLMWrapper(model_name=model, use_langfuse=False)([{"type": "text", "content": "legacy"}], metadata={})
                self.assertEqual(completion.call_args.kwargs["model"], model)
                self.assertNotIn("api_base", completion.call_args.kwargs)

    def test_minimax_media_still_rejected_before_http(self):
        wrapper = LiteLLMWrapper(model_name="minimax/MiniMax-M3", use_langfuse=False)
        with self.assertRaisesRegex(ValueError, "Multimodal capability"):
            wrapper([{"type": "video", "content": "https://example.com/video.mp4"}], metadata={})
        self.assertEqual(self.requests, [])


if __name__ == "__main__":
    unittest.main()
