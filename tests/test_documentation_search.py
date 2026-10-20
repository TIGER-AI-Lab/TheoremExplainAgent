"""Scripted models and HTTP fixtures; no model inference or Manim rendering."""
import asyncio
import importlib
import json
import os
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import requests

from src.core.code_generator import CodeGenerator
from src.rag.parallel_search import ENDPOINT, USER_AGENT, SearchError


URL = "https://docs.manim.community/en/v0.18.1/reference/manim.mobject.geometry.arc.Circle.html"


class Response:
    def __init__(self, message, sse=False, status=200):
        self.message = message
        self.status_code = status
        self.headers = {"Content-Type": "text/event-stream" if sse else "application/json"}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def iter_content(self, size):
        yield json.dumps(self.message).encode()

    def iter_lines(self, decode_unicode=False):
        yield ": heartbeat"
        yield ""
        yield "event: message"
        yield "data: " + json.dumps(self.message)
        yield ""


class MCPFixture:
    def __init__(self, sse=False):
        self.requests = []
        self.sse = sse
        self.failure = None
        self.http_status = 200
        self.bad_response_id = False
        self.search_result = {"results": [
            {"url": "https://example.com/not-manim", "excerpts": ["UNRELATED"]},
            {"url": URL, "excerpts": ["SEARCH_EXCERPT"]},
        ]}
        self.fetch_result = {"results": [{"url": URL, "excerpts": ["Circle(radius=1): FETCHED_DOC"]}]}

    def send(self, http, request, **kwargs):
        payload = json.loads(request.body)
        self.requests.append((request.url, dict(request.headers), {**kwargs, "json": payload}))
        method = payload["method"]
        if method == "initialize":
            result = {"protocolVersion": "2025-03-26", "capabilities": {}, "serverInfo": {"name": "fixture", "version": "1"}}
        elif method == "notifications/initialized":
            return Response(None, status=202)
        elif method == "tools/list":
            result = {"tools": [{"name": "web_search"}, {"name": "web_fetch"}]}
        else:
            if self.http_status != 200:
                return Response({}, status=self.http_status)
            name = payload["params"]["name"]
            if self.failure:
                if isinstance(self.failure, BaseException):
                    raise self.failure
                return Response({"id": payload["id"], "result": self.failure}, self.sse)
            value = self.search_result if name == "web_search" else self.fetch_result
            result = {"content": [{"type": "text", "text": json.dumps(value)}]}
        response = Response({"jsonrpc": "2.0", "id": -1 if self.bad_response_id else payload["id"], "result": result}, self.sse)
        if method == "initialize":
            response.headers["Mcp-Session-Id"] = "transport-id-only"
        return response

    def calls(self):
        return [r[2]["json"]["params"] for r in self.requests if r[2]["json"]["method"] == "tools/call"]


def helper(inputs, **kwargs):
    return '```json\n[{"type":"manim-core","query":"Manim Circle radius usage example"}]\n```'


class DocumentationSearchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.fixture = MCPFixture()
        fixture = self.fixture
        self.send = patch.object(requests.Session, "send", lambda http, request, **kw: fixture.send(http, request, **kw))
        self.send.start()
        self.addCleanup(self.send.stop)
        self.model = Mock(return_value='```python\nprint("scene")\n```')

    def generator(self, **kwargs):
        return CodeGenerator(self.model, helper, output_dir=self.temp.name, use_langfuse=False, **kwargs)

    def generate(self, generator, **kwargs):
        return generator.generate_manim_code("Circles", "A circle", "Outline", "Draw a circle", 1, **kwargs)

    def test_default_dispatch_and_prompt_consumption_json_and_sse(self):
        for sse in (False, True):
            self.fixture.sse = sse
            self.fixture.requests.clear()
            with patch.dict("os.environ", {"PARALLEL_API_KEY": "must-not-be-used"}, clear=True):
                generator = self.generator()
                code, _ = self.generate(generator, additional_context="EXISTING_CONTEXT")
            self.assertIn('print("scene")', code)
            prompt = self.model.call_args.args[0][0]["content"]
            self.assertIn("FETCHED_DOC", prompt)
            self.assertIn(URL, prompt)
            self.assertIn("EXISTING_CONTEXT", prompt)
            self.assertNotIn("UNRELATED", prompt)
            calls = self.fixture.calls()
            self.assertEqual([x["name"] for x in calls], ["web_search", "web_fetch"])
            self.assertEqual(calls[0]["arguments"]["session_id"], calls[1]["arguments"]["session_id"])
            self.assertEqual(calls[1]["arguments"]["urls"], [URL])
            self.assertIn("site:docs.manim.community ", calls[0]["arguments"]["search_queries"][0])
            for url, headers, kwargs in self.fixture.requests:
                self.assertEqual(url, ENDPOINT)
                self.assertEqual(headers["User-Agent"], USER_AGENT)
                self.assertNotIn("Authorization", headers)
                self.assertEqual(kwargs["timeout"], (5, 20))
                self.assertFalse(kwargs["allow_redirects"])

    def test_netrc_credentials_are_not_used(self):
        with patch("requests.sessions.get_netrc_auth", side_effect=AssertionError("Anonymous request used netrc")):
            self.generate(self.generator())
        self.assertTrue(self.fixture.requests)

    def test_transport_session_is_separate_from_research_identity(self):
        self.generate(self.generator(), search_session_id="research-id")
        for _, headers, kwargs in self.fixture.requests:
            if kwargs["json"]["method"] != "initialize":
                self.assertEqual(headers["Mcp-Session-Id"], "transport-id-only")
                self.assertEqual(headers["MCP-Protocol-Version"], "2025-03-26")
        self.assertTrue(all(c["arguments"]["session_id"] == "research-id" for c in self.fixture.calls()))

    def test_http_errors_redirects_and_malformed_responses(self):
        generator = self.generator()
        for status in (302, 403, 429, 500):
            self.fixture.http_status = status
            with self.assertRaises(SearchError):
                self.generate(generator)
        self.fixture.http_status = 200
        self.fixture.bad_response_id = True
        with self.assertRaises(SearchError):
            self.generate(generator)
        self.model.assert_not_called()

    def test_disabled_search_makes_no_queries_or_requests(self):
        generator = self.generator(search_provider="none")
        generator.helper_model = Mock(side_effect=AssertionError("No helper call expected"))
        self.generate(generator)
        self.assertEqual(self.fixture.requests, [])

    def test_local_choice_preserves_embedding_settings(self):
        vector = Mock()
        vector.find_relevant_docs.return_value = "LOCAL_DOC"
        module = types.ModuleType("src.rag.vector_store")
        module.RAGVectorStore = Mock(return_value=vector)
        with patch.dict(sys.modules, {"src.rag.vector_store": module}):
            for choices in ({"use_rag": True}, {"search_provider": "local"}):
                generator = self.generator(embedding_model="saved-embedding", manim_docs_path="saved-docs", **choices)
                self.generate(generator)
                self.assertEqual(module.RAGVectorStore.call_args.kwargs["embedding_model"], "saved-embedding")
                self.assertEqual(module.RAGVectorStore.call_args.kwargs["manim_docs_path"], "saved-docs")
                self.assertIn("LOCAL_DOC", self.model.call_args.args[0][0]["content"])
        self.assertEqual(self.fixture.requests, [])

    def test_conflicting_and_invalid_settings_fail_before_network(self):
        for choices in ({"search_provider": "invalid"}, {"use_rag": True, "search_provider": "none"},
                        {"use_rag": True, "search_provider": "parallel"}):
            with self.assertRaises(ValueError):
                self.generator(**choices)
        self.assertEqual(self.fixture.requests, [])

    def test_errors_timeout_permission_quota_and_cancellation(self):
        generator = self.generator()
        for failure in (requests.Timeout(), requests.ConnectionError(),
                        {"isError": True, "content": [{"type": "text", "text": "quota exceeded"}]},
                        {"isError": True, "content": [{"type": "text", "text": "permission denied"}]}):
            self.fixture.failure = failure
            with self.assertRaises(SearchError):
                self.generate(generator, search_session_id="same-research-run")
            self.model.assert_not_called()
        self.fixture.failure = asyncio.CancelledError()
        with self.assertRaises(asyncio.CancelledError):
            self.generate(generator, search_session_id="same-research-run")
        self.assertEqual({x["arguments"]["session_id"] for x in self.fixture.calls()}, {"same-research-run"})

    def test_fetch_error_uses_search_excerpts_and_empty_results_are_valid(self):
        self.fixture.fetch_result = {"results": [], "errors": [{"url": URL}]}
        self.generate(self.generator())
        self.assertIn("SEARCH_EXCERPT", self.model.call_args.args[0][0]["content"])
        self.fixture.search_result = {"results": []}
        self.generate(self.generator())
        self.assertNotIn("SEARCH_EXCERPT", self.model.call_args.args[0][0]["content"])

    def test_untrusted_urls_are_not_fetched(self):
        self.fixture.search_result = {"results": [{"url": x} for x in (
            "file:///etc/passwd", "https://docs.manim.community.evil.com/", "https://user@docs.manim.community/",
            "http://docs.manim.community/", "https://docs.manim.community:invalid/")]}
        self.generate(self.generator())
        self.assertEqual([x["name"] for x in self.fixture.calls()], ["web_search"])

    def test_video_job_owns_identity_across_scenes_repairs_and_concurrent_jobs(self):
        # Isolate optional media dependencies; exercise the real VideoGenerator,
        # CodeGenerator, prompt formatting, query helper and MCP caller below.
        renderer_module = types.ModuleType("src.core.video_renderer")
        renderer_module.VideoRenderer = Mock()
        parse_module = types.ModuleType("src.core.parse_video")
        parse_module.get_images_from_video = Mock()
        parse_module.image_with_most_non_black_space = Mock()
        with patch.dict(sys.modules, {"src.core.video_renderer": renderer_module, "src.core.parse_video": parse_module}):
            video_module = importlib.import_module("generate_video")
        video = video_module.VideoGenerator(self.model, helper_model=helper, output_dir=self.temp.name, use_langfuse=False)
        self.assertEqual(video.code_generator.search_provider, "parallel")
        self.assertFalse(video.planner.use_rag)

        class Renderer:
            def __init__(self):
                self.seen = set()

            async def render_scene(self, **kwargs):
                await asyncio.sleep(0)  # Interleave both video jobs between generation and repair.
                key = kwargs["topic"], kwargs["curr_scene"]
                if key not in self.seen:
                    self.seen.add(key)
                    return kwargs["code"], "Fix Circle radius"
                return kwargs["code"], None

        video.video_renderer = Renderer()

        async def run_jobs():
            await asyncio.gather(*[
                video.render_video_fix_code(topic, "description", "outline", ["Draw a circle", "Draw another circle"], session_id="shared-trace")
                for topic in ("Job A", "Job B")])

        asyncio.run(run_jobs())
        calls = self.fixture.calls()
        ids = [c["arguments"]["session_id"] for c in calls if c["name"] == "web_search"]
        self.assertEqual(len(ids), 8)  # Two scenes and two repairs in each job.
        self.assertEqual(len(set(ids)), 2)
        self.assertTrue(all(ids.count(value) == 4 for value in set(ids)))
        for search, fetch in zip(calls[::2], calls[1::2]):
            self.assertEqual(search["arguments"]["session_id"], fetch["arguments"]["session_id"])
        self.assertTrue(all("FETCHED_DOC" in call.args[0][0]["content"] for call in self.model.call_args_list))
        first_job_id = ids[0]
        asyncio.run(video.render_video_fix_code("Job A", "description", "outline", ["Draw a circle"], session_id="shared-trace"))
        self.assertEqual(self.fixture.calls()[-1]["arguments"]["session_id"], first_job_id)

        # A failed lookup must preserve the persisted identity when retried,
        # including after a new VideoGenerator is constructed for the same output.
        self.fixture.failure = {"isError": True, "content": [{"type": "text", "text": "quota exceeded"}]}
        with self.assertRaises(SearchError):
            asyncio.run(video.render_video_fix_code("Job A", "description", "outline", ["Draw a circle"], session_id="shared-trace"))
        self.assertEqual(self.fixture.calls()[-1]["arguments"]["session_id"], first_job_id)
        self.fixture.failure = None
        resumed = video_module.VideoGenerator(self.model, helper_model=helper, output_dir=self.temp.name, use_langfuse=False)
        resumed.video_renderer = video.video_renderer
        asyncio.run(resumed.render_video_fix_code("Job A", "description", "outline", ["Draw a circle"], session_id="shared-trace"))
        self.assertEqual(self.fixture.calls()[-1]["arguments"]["session_id"], first_job_id)

        # A new output directory explicitly starts a new research job.
        new_job = video_module.VideoGenerator(self.model, helper_model=helper,
                                             output_dir=os.path.join(self.temp.name, "new-job"), use_langfuse=False)
        new_job.code_generator.parallel_search = video.code_generator.parallel_search
        new_job.video_renderer = video.video_renderer
        asyncio.run(new_job.render_video_fix_code("Job A", "description", "outline", ["Draw a circle"], session_id="shared-trace"))
        self.assertNotIn(self.fixture.calls()[-1]["arguments"]["session_id"], ids)


if __name__ == "__main__":
    unittest.main()
