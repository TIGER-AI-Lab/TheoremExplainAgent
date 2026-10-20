import os
import re
import json
from typing import Union, List, Dict
from PIL import Image
import glob
import uuid

from src.utils.utils import extract_json
from mllm_tools.utils import _prepare_text_inputs, _extract_code, _prepare_text_image_inputs
from mllm_tools.gemini import GeminiWrapper
from mllm_tools.vertex_ai import VertexAIWrapper
from task_generator import (
    get_prompt_code_generation,
    get_prompt_fix_error,
    get_prompt_visual_fix_error,
    get_banned_reasonings,
    get_prompt_rag_query_generation_fix_error,
    get_prompt_context_learning_code,
    get_prompt_rag_query_generation_code
)
from task_generator.prompts_raw import (
    _code_font_size,
    _code_disable,
    _code_limit,
    _prompt_manim_cheatsheet
)
from src.rag.parallel_search import ParallelSearch, resolve_search_provider

class CodeGenerator:
    """A class for generating and managing Manim code."""

    def __init__(self, scene_model, helper_model, output_dir="output", print_response=False, use_rag=False, use_context_learning=False, context_learning_path="data/context_learning", chroma_db_path="rag/chroma_db", manim_docs_path="rag/manim_docs", embedding_model="azure/text-embedding-3-large", use_visual_fix_code=False, use_langfuse=True, session_id=None, search_provider=None):
        """Initialize the CodeGenerator.

        Args:
            scene_model: The model used for scene generation
            helper_model: The model used for helper tasks
            output_dir (str, optional): Directory for output files. Defaults to "output".
            print_response (bool, optional): Whether to print model responses. Defaults to False.
            use_rag (bool, optional): Whether to use RAG. Defaults to False.
            use_context_learning (bool, optional): Whether to use context learning. Defaults to False.
            context_learning_path (str, optional): Path to context learning examples. Defaults to "data/context_learning".
            chroma_db_path (str, optional): Path to ChromaDB. Defaults to "rag/chroma_db".
            manim_docs_path (str, optional): Path to Manim docs. Defaults to "rag/manim_docs".
            embedding_model (str, optional): Name of embedding model. Defaults to "azure/text-embedding-3-large".
            use_visual_fix_code (bool, optional): Whether to use visual code fixing. Defaults to False.
            use_langfuse (bool, optional): Whether to use Langfuse logging. Defaults to True.
            session_id (str, optional): Session identifier. Defaults to None.
            search_provider (str, optional): parallel (default), local with use_rag, or none.
        """
        self.scene_model = scene_model
        self.helper_model = helper_model
        self.output_dir = output_dir
        self.print_response = print_response
        self.search_provider = resolve_search_provider(use_rag, search_provider)
        self.use_rag = self.search_provider == "local"
        self.use_context_learning = use_context_learning
        self.context_learning_path = context_learning_path
        self.context_examples = self._load_context_examples() if use_context_learning else None
        self.manim_docs_path = manim_docs_path

        self.use_visual_fix_code = use_visual_fix_code
        self.banned_reasonings = get_banned_reasonings()
        self.session_id = session_id # Use session_id passed from VideoGenerator

        self.parallel_search = ParallelSearch() if self.search_provider == "parallel" else None
        if self.use_rag:
            from src.rag.vector_store import RAGVectorStore
            self.vector_store = RAGVectorStore(
                chroma_db_path=chroma_db_path,
                manim_docs_path=manim_docs_path,
                embedding_model=embedding_model,
                session_id=self.session_id,
                use_langfuse=use_langfuse
            )
        else:
            self.vector_store = None

    def _load_context_examples(self) -> str:
        """Load all context learning examples from the specified directory.

        Returns:
            str: Formatted context learning examples, or None if no examples found.
        """
        examples = []
        for example_file in glob.glob(f"{self.context_learning_path}/**/*.py", recursive=True):
            with open(example_file, 'r') as f:
                examples.append(f"# Example from {os.path.basename(example_file)}\n{f.read()}\n")

        # Format examples using get_prompt_context_learning_code instead of _prompt_context_learning
        if examples:
            formatted_examples = get_prompt_context_learning_code(
                examples="\n".join(examples)
            )
            return formatted_examples
        return None

    def _generate_rag_queries_code(self, implementation: str, scene_trace_id: str = None, topic: str = None, scene_number: int = None, session_id: str = None, relevant_plugins: List[str] = []) -> List[str]:
        """Generate RAG queries from the implementation plan.

        Args:
            implementation (str): The implementation plan text
            scene_trace_id (str, optional): Trace ID for the scene. Defaults to None.
            topic (str, optional): Topic of the scene. Defaults to None.
            scene_number (int, optional): Scene number. Defaults to None.
            session_id (str, optional): Session identifier. Defaults to None.
            relevant_plugins (List[str], optional): List of relevant plugins. Defaults to empty list.

        Returns:
            List[str]: List of generated RAG queries
        """
        # Create a cache key for this scene
        cache_key = f"{topic}_scene{scene_number}"

        # Check if we already have a cache file for this scene
        cache_dir = os.path.join(self.output_dir, re.sub(r'[^a-z0-9_]+', '_', topic.lower()), f"scene{scene_number}", "rag_cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, "rag_queries_code.json")

        # If cache file exists, load and return cached queries
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                cached_queries = json.load(f)
                print(f"Using cached RAG queries for {cache_key}")
                return cached_queries

        # Generate new queries if not cached
        if relevant_plugins:
            prompt = get_prompt_rag_query_generation_code(implementation, ", ".join(relevant_plugins))
        else:
            prompt = get_prompt_rag_query_generation_code(implementation, "No plugins are relevant.")

        queries = self.helper_model(
            _prepare_text_inputs(prompt),
            metadata={"generation_name": "rag_query_generation", "trace_id": scene_trace_id, "tags": [topic, f"scene{scene_number}"], "session_id": session_id}
        )

        print(f"RAG queries: {queries}")
        # retreive json triple backticks
        
        try: # add try-except block to handle potential json decode errors
            queries = re.search(r'```json(.*)```', queries, re.DOTALL).group(1)
            queries = json.loads(queries)
        except json.JSONDecodeError as e:
            print(f"JSONDecodeError when parsing RAG queries for storyboard: {e}")
            print(f"Response text was: {queries}")
            return [] # Return empty list in case of parsing error

        # Cache the queries
        with open(cache_file, 'w') as f:
            json.dump(queries, f)

        return queries

    def _generate_rag_queries_error_fix(self, error: str, code: str, scene_trace_id: str = None, topic: str = None, scene_number: int = None, session_id: str = None, relevant_plugins: List[str] = []) -> List[str]:
        """Generate RAG queries for fixing code errors.

        Args:
            error (str): The error message to fix
            code (str): The code containing the error
            scene_trace_id (str, optional): Trace ID for the scene. Defaults to None.
            topic (str, optional): Topic of the scene. Defaults to None.
            scene_number (int, optional): Scene number. Defaults to None.
            session_id (str, optional): Session identifier. Defaults to None.
            relevant_plugins (List[str], optional): List of relevant plugins. Defaults to empty list.

        Returns:
            List[str]: List of generated RAG queries for error fixing
        """
        # Create a cache key for this scene and error
        cache_key = f"{topic}_scene{scene_number}_error_fix"

        # Check if we already have a cache file for error fix queries
        cache_dir = os.path.join(self.output_dir, re.sub(r'[^a-z0-9_]+', '_', topic.lower()), f"scene{scene_number}", "rag_cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, "rag_queries_error_fix.json")

        # If cache file exists, load and return cached queries
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                cached_queries = json.load(f)
                print(f"Using cached RAG queries for error fix in {cache_key}")
                return cached_queries

        # Generate new queries for error fix if not cached
        prompt = get_prompt_rag_query_generation_fix_error(
            error=error,
            code=code,
            relevant_plugins=", ".join(relevant_plugins) if relevant_plugins else "No plugins are relevant."
        )

        queries = self.helper_model(
            _prepare_text_inputs(prompt),
            metadata={"generation_name": "rag-query-generation-fix-error", "trace_id": scene_trace_id, "tags": [topic, f"scene{scene_number}"], "session_id": session_id}
        )

        # remove json triple backticks
        queries = queries.replace("```json", "").replace("```", "")
        try: # add try-except block to handle potential json decode errors
            queries = json.loads(queries)
        except json.JSONDecodeError as e:
            print(f"JSONDecodeError when parsing RAG queries for error fix: {e}")
            print(f"Response text was: {queries}")
            return [] # Return empty list in case of parsing error

        # Cache the queries
        with open(cache_file, 'w') as f:
            json.dump(queries, f)

        return queries

    def _extract_code_with_retries(self, response_text: str, pattern: str, generation_name: str = None, trace_id: str = None, session_id: str = None, max_retries: int = 10) -> str:
        """Extract code from response text with retry logic.

        Args:
            response_text (str): The text containing code to extract
            pattern (str): Regex pattern for extracting code
            generation_name (str, optional): Name of generation step. Defaults to None.
            trace_id (str, optional): Trace identifier. Defaults to None.
            session_id (str, optional): Session identifier. Defaults to None.
            max_retries (int, optional): Maximum number of retries. Defaults to 10.

        Returns:
            str: The extracted code

        Raises:
            ValueError: If code extraction fails after max retries
        """
        retry_prompt = """
        Please extract the Python code in the correct format using the pattern: {pattern}. 
        You MUST NOT include any other text or comments. 
        You MUST return the exact same code as in the previous response, NO CONTENT EDITING is allowed.
        Previous response: 
        {response_text}
        """

        for attempt in range(max_retries):
            code_match = re.search(pattern, response_text, re.DOTALL)
            if code_match:
                return code_match.group(1)
            
            if attempt < max_retries - 1:
                print(f"Attempt {attempt + 1}: Failed to extract code pattern. Retrying...")
                # Regenerate response with a more explicit prompt
                response_text = self.scene_model(
                    _prepare_text_inputs(retry_prompt.format(pattern=pattern, response_text=response_text)),
                    metadata={
                        "generation_name": f"{generation_name}_format_retry_{attempt + 1}",
                        "trace_id": trace_id,
                        "session_id": session_id
                    }
                )
        
        raise ValueError(f"Failed to extract code pattern after {max_retries} attempts. Pattern: {pattern}")

    def generate_manim_code(self,
                            topic: str,
                            description: str,                            
                            scene_outline: str,
                            scene_implementation: str,
                            scene_number: int,
                            additional_context: Union[str, List[str]] = None,
                            scene_trace_id: str = None,
                            session_id: str = None,
                            rag_queries_cache: Dict = None,
                            search_session_id: str = None) -> str:
        """Generate Manim code from video plan.

        Args:
            topic (str): Topic of the scene
            description (str): Description of the scene
            scene_outline (str): Outline of the scene
            scene_implementation (str): Implementation details
            scene_number (int): Scene number
            additional_context (Union[str, List[str]], optional): Additional context. Defaults to None.
            scene_trace_id (str, optional): Trace identifier. Defaults to None.
            session_id (str, optional): Session identifier. Defaults to None.
            rag_queries_cache (Dict, optional): Cache for RAG queries. Defaults to None.
            search_session_id (str, optional): Opaque ID shared by searches for one video job.
                Omit only for a standalone research call.

        Returns:
            Tuple[str, str]: Generated code and response text
        """
        if self.use_context_learning:
            # Add context examples to additional_context
            if additional_context is None:
                additional_context = []
            elif isinstance(additional_context, str):
                additional_context = [additional_context]
            
            # Now using the properly formatted code examples
            if self.context_examples:
                additional_context.append(self.context_examples)

        if self.search_provider != "none":
            # Generate RAG queries (will use cache if available)
            rag_queries = self._generate_rag_queries_code(
                implementation=scene_implementation,
                scene_trace_id=scene_trace_id,
                topic=topic,
                scene_number=scene_number,
                session_id=session_id
            )

            if self.parallel_search:
                retrieved_docs = self.parallel_search.get_context(
                    rag_queries, search_session_id or str(uuid.uuid4()))
            else:
                retrieved_docs = self.vector_store.find_relevant_docs(
                    queries=rag_queries, k=2, trace_id=scene_trace_id,
                    topic=topic, scene_number=scene_number)

            # Format the retrieved documents into a string
            if additional_context is None:
                additional_context = []
            elif isinstance(additional_context, str):
                additional_context = [additional_context]
            else:
                additional_context = list(additional_context)
            additional_context.append(retrieved_docs)

        # Format code generation prompt with plan and retrieved context
        prompt = get_prompt_code_generation(
            scene_outline=scene_outline,
            scene_implementation=scene_implementation,
            topic=topic,
            description=description,
            scene_number=scene_number,
            additional_context=additional_context
        )

        # Generate code using model
        response_text = self.scene_model(
            _prepare_text_inputs(prompt),
            metadata={"generation_name": "code_generation", "trace_id": scene_trace_id, "tags": [topic, f"scene{scene_number}"], "session_id": session_id}
        )

        # Extract code with retries
        code = self._extract_code_with_retries(
            response_text,
            r"```python(.*)```",
            generation_name="code_generation",
            trace_id=scene_trace_id,
            session_id=session_id
        )
        return code, response_text

    def fix_code_errors(self, implementation_plan: str, code: str, error: str, scene_trace_id: str, topic: str, scene_number: int, session_id: str, rag_queries_cache: Dict = None, search_session_id: str = None) -> str:
        """Fix errors in generated Manim code.

        Args:
            implementation_plan (str): Original implementation plan
            code (str): Code containing errors
            error (str): Error message to fix
            scene_trace_id (str): Trace identifier
            topic (str): Topic of the scene
            scene_number (int): Scene number
            session_id (str): Session identifier
            rag_queries_cache (Dict, optional): Cache for RAG queries. Defaults to None.
            search_session_id (str, optional): Opaque ID shared by searches for one video job.
                Omit only for a standalone research call.

        Returns:
            Tuple[str, str]: Fixed code and response text
        """
        # Format error fix prompt
        prompt = get_prompt_fix_error(implementation_plan=implementation_plan, manim_code=code, error=error)

        if self.search_provider != "none":
            # Generate RAG queries for error fixing
            rag_queries = self._generate_rag_queries_error_fix(
                error=error,
                code=code,
                scene_trace_id=scene_trace_id,
                topic=topic,
                scene_number=scene_number,
                session_id=session_id
            )
            if self.parallel_search:
                retrieved_docs = self.parallel_search.get_context(
                    rag_queries, search_session_id or str(uuid.uuid4()))
            else:
                retrieved_docs = self.vector_store.find_relevant_docs(
                    queries=rag_queries, k=2, trace_id=scene_trace_id,
                    topic=topic, scene_number=scene_number)

            # Format the retrieved documents into a string
            prompt = get_prompt_fix_error(implementation_plan=implementation_plan, manim_code=code, error=error, additional_context=retrieved_docs)

        # Get fixed code from model
        response_text = self.scene_model(
            _prepare_text_inputs(prompt),
            metadata={"generation_name": "code_fix_error", "trace_id": scene_trace_id, "tags": [topic, f"scene{scene_number}"], "session_id": session_id}
        )

        # Extract fixed code with retries
        fixed_code = self._extract_code_with_retries(
            response_text,
            r"```python(.*)```",
            generation_name="code_fix_error",
            trace_id=scene_trace_id,
            session_id=session_id
        )
        return fixed_code, response_text

    def visual_self_reflection(self, code: str, media_path: Union[str, Image.Image], scene_trace_id: str, topic: str, scene_number: int, session_id: str) -> str:
        """Use snapshot image or mp4 video to fix code.

        Args:
            code (str): Code to fix
            media_path (Union[str, Image.Image]): Path to media file or PIL Image
            scene_trace_id (str): Trace identifier
            topic (str): Topic of the scene
            scene_number (int): Scene number
            session_id (str): Session identifier

        Returns:
            Tuple[str, str]: Fixed code and response text
        """
        
        # Determine if we're dealing with video or image
        is_video = isinstance(media_path, str) and media_path.endswith('.mp4')
        
        # Load prompt template
        with open('task_generator/prompts_raw/prompt_visual_self_reflection.txt', 'r') as f:
            prompt_template = f.read()
        
        # Format prompt
        prompt = prompt_template.format(code=code)
        
        # Prepare input based on media type
        if is_video and isinstance(self.scene_model, (GeminiWrapper, VertexAIWrapper)):
            # For video with Gemini models
            messages = [
                {"type": "text", "content": prompt},
                {"type": "video", "content": media_path}
            ]
        else:
            # For images or non-Gemini models
            if isinstance(media_path, str):
                media = Image.open(media_path)
            else:
                media = media_path
            messages = [
                {"type": "text", "content": prompt},
                {"type": "image", "content": media}
            ]
        
        # Get model response
        response_text = self.scene_model(
            messages,
            metadata={
                "generation_name": "visual_self_reflection",
                "trace_id": scene_trace_id,
                "tags": [topic, f"scene{scene_number}"],
                "session_id": session_id
            }
        )
        
        # Extract code with retries
        fixed_code = self._extract_code_with_retries(
            response_text,
            r"```python(.*)```",
            generation_name="visual_self_reflection",
            trace_id=scene_trace_id,
            session_id=session_id
        )
        return fixed_code, response_text