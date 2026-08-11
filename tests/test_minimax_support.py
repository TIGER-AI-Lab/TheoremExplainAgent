import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import Mock


fake_litellm = types.ModuleType("litellm")
fake_litellm.completion = Mock(
    return_value=SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))]
    )
)
fake_litellm.completion_cost = Mock(return_value=0)
fake_litellm.success_callback = []
fake_litellm.failure_callback = []
sys.modules["litellm"] = fake_litellm

fake_dotenv = types.ModuleType("dotenv")
fake_dotenv.load_dotenv = Mock()
sys.modules["dotenv"] = fake_dotenv

from mllm_tools.litellm import LiteLLMWrapper


class FakeGeminiWrapper:
    def __init__(self, model_name, temperature):
        self.model_name = model_name
        self.temperature = temperature


fake_moviepy = types.ModuleType("moviepy")
fake_moviepy.VideoFileClip = object
sys.modules["moviepy"] = fake_moviepy

fake_gemini = types.ModuleType("mllm_tools.gemini")
fake_gemini.GeminiWrapper = FakeGeminiWrapper
sys.modules["mllm_tools.gemini"] = fake_gemini

fake_eval_utils = types.ModuleType("eval_suite.utils")
fake_eval_utils.calculate_geometric_mean = Mock()
sys.modules["eval_suite.utils"] = fake_eval_utils

fake_text_utils = types.ModuleType("eval_suite.text_utils")
fake_text_utils.parse_srt_to_text = Mock()
fake_text_utils.fix_transcript = Mock()
fake_text_utils.evaluate_text = Mock()
sys.modules["eval_suite.text_utils"] = fake_text_utils

fake_video_utils = types.ModuleType("eval_suite.video_utils")
fake_video_utils.evaluate_video_chunk_new = Mock()
sys.modules["eval_suite.video_utils"] = fake_video_utils

fake_image_utils = types.ModuleType("eval_suite.image_utils")
fake_image_utils.evaluate_sampled_images = Mock()
sys.modules["eval_suite.image_utils"] = fake_image_utils

import evaluate


class MiniMaxSupportTest(unittest.TestCase):
    def setUp(self):
        fake_litellm.completion.reset_mock()

    def test_minimax_m3_formats_image_and_video_inputs(self):
        model = LiteLLMWrapper(
            model_name="minimax/MiniMax-M3",
            use_langfuse=False,
        )

        result = model([
            {"type": "image", "content": "https://example.com/image.png"},
            {"type": "video", "content": "https://example.com/video.mp4"},
        ])

        self.assertEqual(result, "ok")
        messages = fake_litellm.completion.call_args.kwargs["messages"]
        self.assertEqual(
            messages[0]["content"][0],
            {
                "type": "image_url",
                "image_url": {"url": "https://example.com/image.png"},
            },
        )
        self.assertEqual(
            messages[1]["content"][0],
            {
                "type": "video_url",
                "video_url": {"url": "https://example.com/video.mp4"},
            },
        )

    def test_minimax_m27_rejects_media_inputs(self):
        model = LiteLLMWrapper(
            model_name="minimax/MiniMax-M2.7",
            use_langfuse=False,
        )

        with self.assertRaisesRegex(ValueError, "only supports text input"):
            model([{"type": "image", "content": "https://example.com/image.png"}])

        fake_litellm.completion.assert_not_called()

    def test_minimax_m3_is_available_for_video_evaluation(self):
        self.assertIn("minimax/MiniMax-M3", evaluate.VIDEO_MODELS)
        self.assertNotIn("minimax/MiniMax-M2.7", evaluate.VIDEO_MODELS)
        model = evaluate._create_video_model("minimax/MiniMax-M3")

        self.assertIsInstance(model, LiteLLMWrapper)
        self.assertEqual(model.model_name, "minimax/MiniMax-M3")

    def test_existing_video_models_keep_their_wrapper(self):
        model = evaluate._create_video_model("gemini/gemini-1.5-pro-002")

        self.assertIsInstance(model, FakeGeminiWrapper)
        self.assertEqual(model.model_name, "gemini/gemini-1.5-pro-002")


if __name__ == "__main__":
    unittest.main()
