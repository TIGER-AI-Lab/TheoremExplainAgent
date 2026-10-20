import os
import json
import random
from typing import Union, List, Dict, Optional
import subprocess
import argparse
import glob
from PIL import Image
import re
from dotenv import load_dotenv
import asyncio
import uuid # Import uuid for generating trace_id

from mllm_tools.litellm import LiteLLMWrapper
from mllm_tools.utils import _prepare_text_inputs # Keep _prepare_text_inputs if still used directly in main

# Import new modules
from src.core.video_planner import VideoPlanner
from src.core.code_generator import CodeGenerator
from src.core.video_renderer import VideoRenderer
from src.utils.utils import _print_response, _extract_code, extract_xml # Import utility functions
from src.config.config import Config # Import Config class
from src.rag.parallel_search import resolve_search_provider

# Video parsing
from src.core.parse_video import (
    get_images_from_video,
    image_with_most_non_black_space
)
from task_generator import get_banned_reasonings
from task_generator.prompts_raw import (_code_font_size, _code_disable, _code_limit, _prompt_manim_cheatsheet)

# Load allowed models list from JSON file
allowed_models_path = os.path.join(os.path.dirname(__file__), 'src', 'utils', 'allowed_models.json')
with open(allowed_models_path, 'r') as f:
    allowed_models = json.load(f).get("allowed_models", [])

load_dotenv(override=True)

class VideoGenerator:
    """
    A class for generating manim videos using AI models.

    This class coordinates the video generation pipeline by managing scene planning,
    code generation, and video rendering. It supports concurrent scene processing,
    visual code fixing, and RAG (Retrieval Augmented Generation).

    Args:
        planner_model: Model used for scene planning and high-level decisions
        scene_model: Model used specifically for scene generation (defaults to planner_model)
        helper_model: Helper model for additional tasks (defaults to planner_model)
        output_dir (str): Directory to store generated files and videos
        verbose (bool): Whether to print detailed output
        use_rag (bool): Whether to use Retrieval Augmented Generation
        use_context_learning (bool): Whether to use context learning with example code
        context_learning_path (str): Path to context learning examples
        chroma_db_path (str): Path to ChromaDB for RAG
        manim_docs_path (str): Path to Manim documentation for RAG
        embedding_model (str): Model to use for embeddings
        use_visual_fix_code (bool): Whether to use visual feedback for code fixing
        use_langfuse (bool): Whether to enable Langfuse logging
        trace_id (str, optional): Trace ID for logging
        max_scene_concurrency (int): Maximum number of scenes to process concurrently

    Attributes:
        output_dir (str): Directory for output files
        verbose (bool): Verbosity flag
        use_visual_fix_code (bool): Visual code fixing flag
        session_id (str): Unique session identifier
        scene_semaphore (asyncio.Semaphore): Controls concurrent scene processing
        banned_reasonings (list): List of banned reasoning patterns
        planner (VideoPlanner): Handles scene planning
        code_generator (CodeGenerator): Handles code generation
        video_renderer (VideoRenderer): Handles video rendering
    """

    def __init__(self,
                 planner_model,
                 scene_model=None,
                 helper_model=None,
                 output_dir="output",
                 verbose=False,
                 use_rag=False,
                 use_context_learning=False,
                 context_learning_path="data/context_learning",
                 chroma_db_path="data/rag/chroma_db",
                 manim_docs_path="data/rag/manim_docs",
                 embedding_model="azure/text-embedding-3-large",
                 use_visual_fix_code=False,
                 use_langfuse=True,
                 trace_id=None,
                 max_scene_concurrency: int = 5,
                 search_provider: str = None):
        search_provider = resolve_search_provider(use_rag, search_provider)
        use_rag = search_provider == "local"
        self.output_dir = output_dir
        self.verbose = verbose
        self.use_visual_fix_code = use_visual_fix_code
        self.session_id = self._load_or_create_session_id()  # Modified to load existing or create new
        self.scene_semaphore = asyncio.Semaphore(max_scene_concurrency)
        self.banned_reasonings = get_banned_reasonings()

        # Initialize separate modules
        self.planner = VideoPlanner(
            planner_model=planner_model,
            helper_model=helper_model,
            output_dir=output_dir,
            print_response=verbose,
            use_context_learning=use_context_learning,
            context_learning_path=context_learning_path,
            use_rag=use_rag,
            session_id=self.session_id,
            chroma_db_path=chroma_db_path,
            manim_docs_path=manim_docs_path,
            embedding_model=embedding_model,
            use_langfuse=use_langfuse
        )
        self.code_generator = CodeGenerator(
            scene_model=scene_model if scene_model is not None else planner_model,
            helper_model=helper_model if helper_model is not None else planner_model,
            output_dir=output_dir,
            print_response=verbose,
            use_rag=use_rag,
            use_context_learning=use_context_learning,
            context_learning_path=context_learning_path,
            chroma_db_path=chroma_db_path,
            manim_docs_path=manim_docs_path,
            embedding_model=embedding_model,
            use_visual_fix_code=use_visual_fix_code,
            use_langfuse=use_langfuse,
            session_id=self.session_id,
            search_provider=search_provider
        )
        self.video_renderer = VideoRenderer(
            output_dir=output_dir,
            print_response=verbose,
            use_visual_fix_code=use_visual_fix_code
        )

    def _load_or_create_session_id(self) -> str:
        """
        Load existing session ID from file or create a new one.

        Returns:
            str: The session ID either loaded from file or newly created.
        """
        session_file = os.path.join(self.output_dir, "session_id.txt")

        if os.path.exists(session_file):
            with open(session_file, 'r') as f:
                session_id = f.read().strip()
                print(f"Loaded existing session ID: {session_id}")
                return session_id

        # Create new session ID if none exists
        session_id = str(uuid.uuid4())
        os.makedirs(self.output_dir, exist_ok=True)
        with open(session_file, 'w') as f:
            f.write(session_id)
        print(f"Created new session ID: {session_id}")
        return session_id

    def _save_topic_session_id(self, topic: str, session_id: str) -> None:
        """
        Save session ID for a specific topic.

        Args:
            topic (str): The topic to save the session ID for
            session_id (str): The session ID to save
        """
        file_prefix = topic.lower()
        file_prefix = re.sub(r'[^a-z0-9_]+', '_', file_prefix)
        topic_dir = os.path.join(self.output_dir, file_prefix)
        os.makedirs(topic_dir, exist_ok=True)

        session_file = os.path.join(topic_dir, "session_id.txt")
        with open(session_file, 'w') as f:
            f.write(session_id)

    def _load_topic_session_id(self, topic: str) -> Optional[str]:
        """
        Load session ID for a specific topic if it exists.

        Args:
            topic (str): The topic to load the session ID for

        Returns:
            Optional[str]: The session ID if found, None otherwise
        """
        file_prefix = topic.lower()
        file_prefix = re.sub(r'[^a-z0-9_]+', '_', file_prefix)
        session_file = os.path.join(self.output_dir, file_prefix, "session_id.txt")

        if os.path.exists(session_file):
            with open(session_file, 'r') as f:
                return f.read().strip()
        return None

    def _load_or_create_search_session_id(self, file_prefix: str) -> str:
        """Resume one documentation research run for a topic's output directory.

        Publish a fully written UUID atomically so concurrent/restarted workers
        cannot replace a job's identity after a failed lookup.
        """
        topic_dir = os.path.join(self.output_dir, file_prefix)
        os.makedirs(topic_dir, exist_ok=True)
        session_path = os.path.join(topic_dir, "search_session_id.txt")
        if not os.path.exists(session_path):
            candidate = str(uuid.uuid4())
            temporary_path = session_path + "." + candidate + ".tmp"
            try:
                with open(temporary_path, "x") as f:
                    f.write(candidate)
                try:
                    os.link(temporary_path, session_path)
                except FileExistsError:
                    pass  # Another worker already published the job's identity.
            finally:
                if os.path.exists(temporary_path):
                    os.unlink(temporary_path)
        with open(session_path) as f:
            # Reject corrupted state instead of silently changing quota identity.
            return str(uuid.UUID(f.read().strip()))

    def generate_scene_outline(self,
                            topic: str,
                            description: str,
                            session_id: str) -> str:
        """
        Generate scene outline using VideoPlanner.

        Args:
            topic (str): The topic of the video
            description (str): Description of the video content
            session_id (str): Session identifier for tracking

        Returns:
            str: Generated scene outline
        """
        return self.planner.generate_scene_outline(topic, description, session_id)

    async def generate_scene_implementation(self,
                                      topic: str,
                                      description: str,
                                      plan: str,
                                      session_id: str) -> List[str]:
        """
        Generate scene implementations using VideoPlanner.

        Args:
            topic (str): The topic of the video
            description (str): Description of the video content
            plan (str): The scene plan to implement
            session_id (str): Session identifier for tracking

        Returns:
            List[str]: List of generated scene implementations
        """
        return await self.planner.generate_scene_implementation(topic, description, plan, session_id)

    async def generate_scene_implementation_concurrently(self,
                                              topic: str,
                                              description: str,
                                              plan: str,
                                              session_id: str) -> List[str]:
        """
        Generate scene implementations concurrently using VideoPlanner.

        Args:
            topic (str): The topic of the video
            description (str): Description of the video content
            plan (str): The scene plan to implement
            session_id (str): Session identifier for tracking

        Returns:
            List[str]: List of generated scene implementations
        """
        return await self.planner.generate_scene_implementation_concurrently(topic, description, plan, session_id, self.scene_semaphore) # Pass semaphore

    def load_implementation_plans(self, topic: str) -> Dict[int, Optional[str]]:
        """
        Load implementation plans for each scene.

        Args:
            topic (str): The topic to load implementation plans for

        Returns:
            Dict[int, Optional[str]]: Dictionary mapping scene numbers to their plans.
                                    If a scene's plan is missing, its value will be None.
        """
        file_prefix = topic.lower()
        file_prefix = re.sub(r'[^a-z0-9_]+', '_', file_prefix)

        # Load scene outline from file
        scene_outline_path = os.path.join(self.output_dir, file_prefix, f"{file_prefix}_scene_outline.txt")
        if not os.path.exists(scene_outline_path):
            return {}
        
        with open(scene_outline_path, "r") as f:
            scene_outline = f.read()

        # Extract scene outline to get number of scenes
        scene_outline_content = extract_xml(scene_outline)
        scene_number = len(re.findall(r'<SCENE_(\d+)>[^<]', scene_outline_content))
        print(f"Number of scenes: {scene_number}")

        implementation_plans = {}

        # Check each scene's implementation plan
        for i in range(1, scene_number + 1):
            plan_path = os.path.join(self.output_dir, file_prefix, f"scene{i}", f"{file_prefix}_scene{i}_implementation_plan.txt")
            if os.path.exists(plan_path):
                with open(plan_path, "r") as f:
                    implementation_plans[i] = f.read()
                print(f"Found existing implementation plan for scene {i}")
            else:
                implementation_plans[i] = None
                print(f"Missing implementation plan for scene {i}")

        return implementation_plans

    async def render_video_fix_code(self,
                              topic: str,
                              description: str,
                              scene_outline: str,
                              implementation_plans: List,
                              max_retries=3,
                              session_id: str = None) -> None:
        """
        Render the video for all scenes with code fixing capability.

        Args:
            topic (str): The topic of the video
            description (str): Description of the video content
            scene_outline (str): The overall scene outline
            implementation_plans (List): List of implementation plans for each scene
            max_retries (int, optional): Maximum number of code fix attempts. Defaults to 3.
            session_id (str, optional): Session identifier for tracking
        """
        file_prefix = topic.lower()
        file_prefix = re.sub(r'[^a-z0-9_]+', '_', file_prefix)

        # Resume the research run across failed lookups and worker restarts.
        search_session_id = (self._load_or_create_search_session_id(file_prefix)
                             if self.code_generator.search_provider == "parallel" else None)

        # Create tasks for each scene
        tasks = []
        for i, implementation_plan in enumerate(implementation_plans):
            # Try to load scene trace id, or generate new one if it doesn't exist
            scene_dir = os.path.join(self.output_dir, file_prefix, f"scene{i+1}")
            subplan_dir = os.path.join(scene_dir, "subplans")
            os.makedirs(subplan_dir, exist_ok=True)  # Create directories if they don't exist
            
            scene_trace_id_path = os.path.join(subplan_dir, "scene_trace_id.txt")
            try:
                with open(scene_trace_id_path, 'r') as f:
                    scene_trace_id = f.read().strip()
            except FileNotFoundError:
                scene_trace_id = str(uuid.uuid4())
                with open(scene_trace_id_path, 'w') as f:
                    f.write(scene_trace_id)

            task = self.process_scene(i, scene_outline, implementation_plan, topic, description, max_retries, file_prefix, session_id, scene_trace_id, search_session_id)
            tasks.append(task)

        # Execute all tasks concurrently
        await asyncio.gather(*tasks)

    async def process_scene(self, i: int, scene_outline: str, scene_implementation: str, topic: str, description: str, max_retries: int, file_prefix: str, session_id: str, scene_trace_id: str, search_session_id: str = None):
        """
        Process a single scene using CodeGenerator and VideoRenderer.

        Args:
            i (int): Scene index
            scene_outline (str): Overall scene outline
            scene_implementation (str): Implementation plan for this scene
            topic (str): The topic of the video
            description (str): Description of the video content
            max_retries (int): Maximum number of code fix attempts
            file_prefix (str): Prefix for file naming
            session_id (str): Session identifier for tracking
            scene_trace_id (str): Trace identifier for this scene
        """
        search_session_id = search_session_id or str(uuid.uuid4())
        curr_scene = i + 1
        curr_version = 0
        # scene_trace_id = str(uuid.uuid4()) # Remove uuid generation
        rag_queries_cache = {}  # Initialize RAG queries cache

        # Create necessary directories
        code_dir = os.path.join(self.output_dir, file_prefix, f"scene{curr_scene}", "code")
        os.makedirs(code_dir, exist_ok=True)
        media_dir = os.path.join(self.output_dir, file_prefix, "media") # Define media_dir here

        async with self.scene_semaphore:
            # Step 3A: Generate initial manim code
            code, log = self.code_generator.generate_manim_code(
                topic=topic,
                description=description,
                scene_outline=scene_outline,
                scene_implementation=scene_implementation,
                scene_number=curr_scene,
                additional_context=[_prompt_manim_cheatsheet, _code_font_size, _code_limit, _code_disable],
                scene_trace_id=scene_trace_id, # Use passed scene_trace_id
                session_id=session_id,
                search_session_id=search_session_id,
                rag_queries_cache=rag_queries_cache  # Pass the cache
            )

            # Save initial code and log (file operations can be offloaded if needed)
            with open(os.path.join(code_dir, f"{file_prefix}_scene{curr_scene}_v{curr_version}_init_log.txt"), "w") as f:
                f.write(log)
            with open(os.path.join(code_dir, f"{file_prefix}_scene{curr_scene}_v{curr_version}.py"), "w") as f:
                f.write(code)
            print(f"Code saved to {code_dir}/{file_prefix}_scene{curr_scene}_v{curr_version}.py")

            # Step 3B: Compile and fix code if needed
            error_message = None
            while True: # Retry loop controlled by break statements
                code, error_message = await self.video_renderer.render_scene(
                    code=code,
                    file_prefix=file_prefix,
                    curr_scene=curr_scene,
                    curr_version=curr_version,
                    code_dir=code_dir,
                    media_dir=media_dir,
                    max_retries=max_retries, # Pass max_retries here if needed in render_scene
                    use_visual_fix_code=self.use_visual_fix_code,
                    visual_self_reflection_func=self.code_generator.visual_self_reflection, # Pass visual_self_reflection function
                    banned_reasonings=self.banned_reasonings, # Pass banned reasonings
                    scene_trace_id=scene_trace_id,
                    topic=topic,
                    session_id=session_id
                )
                if error_message is None: # Render success if error_message is None
                    break

                if curr_version >= max_retries: # Max retries reached
                    print(f"Max retries reached for scene {curr_scene}, error: {error_message}")
                    break # Exit retry loop

                curr_version += 1
                # if program runs this, it means that the code is not rendered successfully
                code, log = self.code_generator.fix_code_errors(
                    implementation_plan=scene_implementation,
                    code=code,
                    error=error_message,
                    scene_trace_id=scene_trace_id,
                    topic=topic,
                    scene_number=curr_scene,
                    session_id=session_id,
                    search_session_id=search_session_id,
                    rag_queries_cache=rag_queries_cache
                )

                with open(os.path.join(code_dir, f"{file_prefix}_scene{curr_scene}_v{curr_version}_fix_log.txt"), "w") as f:
                    f.write(log)
                with open(os.path.join(code_dir, f"{file_prefix}_scene{curr_scene}_v{curr_version}.py"), "w") as f:
                    f.write(code)

                print(f"Code saved to {code_dir}/{file_prefix}_scene{curr_scene}_v{curr_version}.py")

    def run_manim_process(self,
                          topic: str):
        """
        Run manim on all generated manim code for a specific topic using VideoRenderer.

        Args:
            topic (str): The topic to render videos for
        """
        return self.video_renderer.run_manim_process(topic)

    def create_snapshot_scene(self, topic: str, scene_number: int, version_number: int, return_type: str = "image"):
        """
        Create a snapshot of the video for a specific topic and scene using VideoRenderer.

        Args:
            topic (str): The topic of the video
            scene_number (int): Scene number to snapshot
            version_number (int): Version number to snapshot
            return_type (str, optional): Type of snapshot to return. Defaults to "image".

        Returns:
            The snapshot in the specified format
        """
        return self.video_renderer.create_snapshot_scene(topic, scene_number, version_number, return_type)

    def combine_videos(self, topic: str):
        """
        Combine all videos and subtitle files for a specific topic using VideoRenderer.

        Args:
            topic (str): The topic to combine videos for
        """
        self.video_renderer.combine_videos(topic)

    async def _generate_scene_implementation_single(self, topic: str, description: str, scene_outline_i: str, i: int, file_prefix: str, session_id: str, scene_trace_id: str) -> str:
        """
        Generate detailed implementation plan for a single scene using VideoPlanner.

        Args:
            topic (str): The topic of the video
            description (str): Description of the video content
            scene_outline_i (str): Outline for this specific scene
            i (int): Scene index
            file_prefix (str): Prefix for file naming
            session_id (str): Session identifier for tracking
            scene_trace_id (str): Trace identifier for this scene

        Returns:
            str: Generated implementation plan
        """
        return await self.planner._generate_scene_implementation_single(topic, description, scene_outline_i, i, file_prefix, session_id, scene_trace_id)

    async def generate_video_pipeline(self, topic: str, description: str, max_retries: int, only_plan: bool = False, specific_scenes: List[int] = None):
        """
        Modified pipeline to handle partial scene completions and option to only generate plans for specific scenes.

        Args:
            topic (str): The topic of the video
            description (str): Description of the video content
            max_retries (int): Maximum number of code fix attempts
            only_plan (bool, optional): Whether to only generate plans without rendering. Defaults to False.
            specific_scenes (List[int], optional): List of specific scenes to process. Defaults to None.
        """
        session_id = self._load_or_create_session_id()
        self._save_topic_session_id(topic, session_id)
        
        file_prefix = topic.lower()
        file_prefix = re.sub(r'[^a-z0-9_]+', '_', file_prefix)
        
        # Load or generate scene outline
        scene_outline_path = os.path.join(self.output_dir, file_prefix, f"{file_prefix}_scene_outline.txt")
        if os.path.exists(scene_outline_path):
            with open(scene_outline_path, "r") as f:
                scene_outline = f.read()
            print(f"Loaded existing scene outline for topic: {topic}")
            if self.planner.use_rag:
                self.planner.relevant_plugins = self.planner.rag_integration.detect_relevant_plugins(topic, description) or []
                self.planner.rag_integration.set_relevant_plugins(self.planner.relevant_plugins)
                print(f"Detected relevant plugins: {self.planner.relevant_plugins}")
        else:
            print(f"Generating new scene outline for topic: {topic}")
            scene_outline = self.planner.generate_scene_outline(topic, description, session_id)
            os.makedirs(os.path.join(self.output_dir, file_prefix), exist_ok=True)
            with open(scene_outline_path, "w") as f:
                f.write(scene_outline)

        # Load or generate implementation plans
        implementation_plans_dict = self.load_implementation_plans(topic)
        if not implementation_plans_dict:
            scene_outline_content = extract_xml(scene_outline)
            scene_numbers = len(re.findall(r'<SCENE_(\d+)>[^<]', scene_outline_content))
            implementation_plans_dict = {i: None for i in range(1, scene_numbers + 1)}

        # Generate missing implementation plans for specified scenes or all missing scenes
        missing_scenes = []
        for scene_num, plan in implementation_plans_dict.items():
            if plan is None and (specific_scenes is None or scene_num in specific_scenes):
                missing_scenes.append(scene_num)

        if missing_scenes:
            print(f"Generating implementation plans for missing scenes: {missing_scenes}")
            for scene_num in missing_scenes:
                scene_outline_content = extract_xml(scene_outline)
                scene_match = re.search(f'<SCENE_{scene_num}>(.*?)</SCENE_{scene_num}>', scene_outline_content, re.DOTALL)
                if scene_match:
                    scene_outline_i = scene_match.group(1)
                    scene_trace_id = str(uuid.uuid4())
                    implementation_plan = await self._generate_scene_implementation_single(
                        topic, description, scene_outline_i, scene_num, file_prefix, session_id, scene_trace_id)
                    implementation_plans_dict[scene_num] = implementation_plan

        if only_plan:
            print(f"Only generating plans - skipping code generation and video rendering for topic: {topic}")
            return

        # Convert dictionary to list maintaining scene order
        sorted_scene_numbers = sorted(implementation_plans_dict.keys())
        implementation_plans = [implementation_plans_dict[i] for i in sorted_scene_numbers]
        
        # Render scenes
        print(f"Starting video rendering for topic: {topic}")
        
        # Check which scenes need processing
        scenes_to_process = []
        for i, implementation_plan in enumerate(implementation_plans):
            scene_dir = os.path.join(self.output_dir, file_prefix, f"scene{i+1}")
            code_dir = os.path.join(scene_dir, "code")
            
            # Check if scene has any code files
            has_code = False
            if os.path.exists(code_dir):
                if any(f.endswith('.py') for f in os.listdir(code_dir)):
                    has_code = True
            
            # For only_render mode, only process scenes without code
            if args.only_render:
                if not has_code:
                    scenes_to_process.append((i+1, implementation_plan))
                    print(f"Scene {i+1} has no code, will process")
                else:
                    print(f"Scene {i+1} already has code, skipping")
            # For normal mode, process scenes that haven't been successfully rendered
            elif not os.path.exists(os.path.join(scene_dir, "succ_rendered.txt")):
                scenes_to_process.append((i+1, implementation_plan))
        
        if not scenes_to_process:
            print(f"No scenes need processing for topic '{topic}'.")
        else:
            print(f"Rendering {len(scenes_to_process)} scenes that need processing...")
            # Create a list of tuples with scene numbers and plans
            scene_plans = [(scene_num, plan) for scene_num, plan in scenes_to_process]
            # Sort by scene number to ensure correct order
            scene_plans.sort(key=lambda x: x[0])
            # Extract just the plans in the correct order
            filtered_implementation_plans = [plan for _, plan in scene_plans]
            await self.render_video_fix_code(topic, description, scene_outline, filtered_implementation_plans,
                                           max_retries=max_retries, session_id=session_id)
        
        if not args.only_render:  # Skip video combination in only_render mode
            print(f"Video rendering completed for topic '{topic}'.")

    def check_theorem_status(self, theorem: Dict) -> Dict[str, bool]:
        """
        Check if a theorem has its plan, code files, and rendered videos with detailed scene status.

        Args:
            theorem (Dict): Dictionary containing theorem information

        Returns:
            Dict[str, bool]: Dictionary containing status information for the theorem
        """
        topic = theorem['theorem']
        file_prefix = topic.lower()
        file_prefix = re.sub(r'[^a-z0-9_]+', '_', file_prefix)
        
        # Check scene outline
        scene_outline_path = os.path.join(self.output_dir, file_prefix, f"{file_prefix}_scene_outline.txt")
        has_scene_outline = os.path.exists(scene_outline_path)
        
        # Get number of scenes if outline exists
        num_scenes = 0
        if has_scene_outline:
            with open(scene_outline_path, "r") as f:
                scene_outline = f.read()
            scene_outline_content = extract_xml(scene_outline)
            num_scenes = len(re.findall(r'<SCENE_(\d+)>[^<]', scene_outline_content))
        
        # Check implementation plans, code files, and rendered videos
        implementation_plans = 0
        code_files = 0
        rendered_scenes = 0
        
        # Track status of individual scenes
        scene_status = []
        for i in range(1, num_scenes + 1):
            scene_dir = os.path.join(self.output_dir, file_prefix, f"scene{i}")
            
            # Check implementation plan
            plan_path = os.path.join(scene_dir, f"{file_prefix}_scene{i}_implementation_plan.txt")
            has_plan = os.path.exists(plan_path)
            if has_plan:
                implementation_plans += 1
            
            # Check code files
            code_dir = os.path.join(scene_dir, "code")
            has_code = False
            if os.path.exists(code_dir):
                if any(f.endswith('.py') for f in os.listdir(code_dir)):
                    has_code = True
                    code_files += 1
            
            # Check rendered scene video
            has_render = False
            if os.path.exists(scene_dir):
                succ_rendered_path = os.path.join(scene_dir, "succ_rendered.txt")
                if os.path.exists(succ_rendered_path):
                    has_render = True
                    rendered_scenes += 1
            
            scene_status.append({
                'scene_number': i,
                'has_plan': has_plan,
                'has_code': has_code,
                'has_render': has_render
            })

        # Check combined video
        combined_video_path = os.path.join(self.output_dir, file_prefix, f"{file_prefix}_combined.mp4")
        has_combined_video = os.path.exists(combined_video_path)
        
        return {
            'topic': topic,
            'has_scene_outline': has_scene_outline,
            'total_scenes': num_scenes,
            'implementation_plans': implementation_plans,
            'code_files': code_files,
            'rendered_scenes': rendered_scenes,
            'has_combined_video': has_combined_video,
            'scene_status': scene_status
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate Manim videos using AI')
    parser.add_argument('--model', type=str, choices=allowed_models,
                      default='gemini/gemini-1.5-pro-002', help='Select the AI model to use')
    parser.add_argument('--topic', type=str, default=None, help='Topic to generate videos for')
    parser.add_argument('--context', type=str, default=None, help='Context of the topic')
    parser.add_argument('--helper_model', type=str, choices=allowed_models,
                      default=None, help='Select the helper model to use')
    parser.add_argument('--only_gen_vid', action='store_true', help='Only generate videos to existing plans')
    parser.add_argument('--only_combine', action='store_true', help='Only combine videos')
    parser.add_argument('--peek_existing_videos', '--peek', action='store_true', help='Peek at existing videos')
    parser.add_argument('--output_dir', type=str, default=Config.OUTPUT_DIR, help='Output directory') # Use Config
    parser.add_argument('--theorems_path', type=str, default=None, help='Path to theorems json file')
    parser.add_argument('--sample_size', '--sample', type=int, default=None, help='Number of theorems to sample')
    parser.add_argument('--verbose', action='store_true', help='Print verbose output')
    parser.add_argument('--max_retries', type=int, default=5, help='Maximum number of retries for code generation')
    parser.add_argument('--use_rag', '--rag', action='store_true', help='Use local Retrieval Augmented Generation')
    parser.add_argument('--search_provider', choices=['parallel', 'local', 'none'], default=None,
                        help='Documentation retrieval: parallel by default, local with --use_rag, or none to disable')
    parser.add_argument('--use_visual_fix_code','--visual_fix_code', action='store_true', help='Use VLM to fix code with rendered visuals')
    parser.add_argument('--chroma_db_path', type=str, default=Config.CHROMA_DB_PATH, help="Path to Chroma DB") # Use Config
    parser.add_argument('--manim_docs_path', type=str, default=Config.MANIM_DOCS_PATH, help="Path to manim docs") # Use Config
    parser.add_argument('--embedding_model', type=str,
                       default=Config.EMBEDDING_MODEL, # Use Config
                       choices=["azure/text-embedding-3-large", "vertex_ai/text-embedding-005"],
                       help='Select the embedding model to use')
    parser.add_argument('--use_context_learning', action='store_true',
                       help='Use context learning with example Manim code')
    parser.add_argument('--context_learning_path', type=str,
                       default=Config.CONTEXT_LEARNING_PATH, # Use Config
                       help='Path to context learning examples')
    parser.add_argument('--use_langfuse', action='store_true',
                       help='Enable Langfuse logging')
    parser.add_argument('--max_scene_concurrency', type=int, default=1, help='Maximum number of scenes to process concurrently')
    parser.add_argument('--max_topic_concurrency', type=int, default=1,
                       help='Maximum number of topics to process concurrently')
    parser.add_argument('--debug_combine_topic', type=str, help='Debug combine videos', default=None)
    parser.add_argument('--only_plan', action='store_true', help='Only generate scene outline and implementation plans')
    parser.add_argument('--check_status', action='store_true', 
                       help='Check planning and code status for all theorems')
    parser.add_argument('--only_render', action='store_true', help='Only render scenes without combining videos')
    parser.add_argument('--scenes', nargs='+', type=int, help='Specific scenes to process (if theorems_path is provided)')
    args = parser.parse_args()

    # Initialize planner model using LiteLLM
    if args.verbose:
        verbose = True
    else:
        verbose = False
    planner_model = LiteLLMWrapper(
        model_name=args.model,
        temperature=0.7,
        print_cost=True,
        verbose=verbose,
        use_langfuse=args.use_langfuse
    )
    helper_model = LiteLLMWrapper(
        model_name=args.helper_model if args.helper_model else args.model, # Use helper_model if provided, else planner_model
        temperature=0.7,
        print_cost=True,
        verbose=verbose,
        use_langfuse=args.use_langfuse
    )
    scene_model = LiteLLMWrapper( # Initialize scene_model separately
        model_name=args.model,
        temperature=0.7,
        print_cost=True,
        verbose=verbose,
        use_langfuse=args.use_langfuse
    )
    print(f"Planner model: {args.model}, Helper model: {args.helper_model if args.helper_model else args.model}, Scene model: {args.model}") # Print all models


    if args.theorems_path:
        # Load the sample theorems
        with open(args.theorems_path, "r") as f:
            theorems = json.load(f)

        if args.sample_size:
            theorems = theorems[:args.sample_size]

        if args.peek_existing_videos:
            print(f"Here's the results of checking whether videos are rendered successfully in {args.output_dir}:")
            # in output_dir, find all combined.mp4 files and print number of successful rendered videos out of total number of folders
            successful_rendered_videos = 0
            total_folders = 0
            for item in os.listdir(args.output_dir):
                if os.path.isdir(os.path.join(args.output_dir, item)):
                    total_folders += 1
                    if os.path.exists(os.path.join(args.output_dir, item, f"{item}_combined.mp4")):
                        successful_rendered_videos += 1
            print(f"Number of successful rendered videos: {successful_rendered_videos}/{total_folders}")

            # also check whether any succ_rendered.txt in scene{i} folder, and then add up the number of successful rendered videos
            successful_rendered_videos = 0
            total_scenes = 0
            for item in os.listdir(args.output_dir):
                if os.path.isdir(os.path.join(args.output_dir, item)):
                    for scene_folder in os.listdir(os.path.join(args.output_dir, item)):
                        if "scene" in scene_folder and os.path.isdir(os.path.join(args.output_dir, item, scene_folder)):
                            total_scenes += 1
                            if os.path.exists(os.path.join(args.output_dir, item, scene_folder, "succ_rendered.txt")):
                                successful_rendered_videos += 1
            print(f"Number of successful rendered scenes: {successful_rendered_videos}/{total_scenes}")
            exit()

        video_generator = VideoGenerator(
            planner_model=planner_model,
            scene_model=scene_model, # Pass scene_model
            helper_model=helper_model, # Pass helper_model
            output_dir=args.output_dir,
            verbose=args.verbose,
            use_rag=args.use_rag,
            search_provider=args.search_provider,
            use_context_learning=args.use_context_learning,
            context_learning_path=args.context_learning_path,
            chroma_db_path=args.chroma_db_path,
            manim_docs_path=args.manim_docs_path,
            embedding_model=args.embedding_model,
            use_visual_fix_code=args.use_visual_fix_code,
            use_langfuse=args.use_langfuse,
            max_scene_concurrency=args.max_scene_concurrency
        )

        if args.debug_combine_topic is not None:
            video_generator.combine_videos(args.debug_combine_topic)
            exit()

        if args.only_gen_vid:
            # Generate videos for existing plans
            print("Generating videos for existing plans...")

            async def process_theorem(theorem, topic_semaphore):
                async with topic_semaphore:
                    topic = theorem['theorem']
                    print(f"Processing topic: {topic}")
                    await video_generator.render_video_fix_code(topic, theorem['description'], max_retries=args.max_retries)

            async def main():
                # Use the command-line argument for topic concurrency
                topic_semaphore = asyncio.Semaphore(args.max_topic_concurrency)
                tasks = [process_theorem(theorem, topic_semaphore) for theorem in theorems]
                await asyncio.gather(*tasks)

            asyncio.run(main())

        elif args.check_status:
            print("\nChecking theorem status...")
            video_generator = VideoGenerator(
                planner_model=planner_model,
                scene_model=scene_model,
                helper_model=helper_model,
                output_dir=args.output_dir,
                verbose=args.verbose,
                use_rag=args.use_rag,
                search_provider=args.search_provider,
                use_context_learning=args.use_context_learning,
                context_learning_path=args.context_learning_path,
                chroma_db_path=args.chroma_db_path,
                manim_docs_path=args.manim_docs_path,
                embedding_model=args.embedding_model,
                use_visual_fix_code=args.use_visual_fix_code,
                use_langfuse=args.use_langfuse,
                max_scene_concurrency=args.max_scene_concurrency
            )
            
            all_statuses = [video_generator.check_theorem_status(theorem) for theorem in theorems]
            
            # Print combined status table
            print("\nTheorem Status:")
            print("-" * 160)
            print(f"{'Topic':<40} {'Outline':<8} {'Total':<8} {'Status (Plan/Code/Render)':<50} {'Combined':<10} {'Missing Components':<40}")
            print("-" * 160)
            for status in all_statuses:
                # Create status string showing plan/code/render completion for each scene
                scene_status_str = ""
                for scene in status['scene_status']:
                    scene_str = (
                        ("P" if scene['has_plan'] else "-") +
                        ("C" if scene['has_code'] else "-") +
                        ("R" if scene['has_render'] else "-") + " "
                    )
                    scene_status_str += scene_str
                
                # Collect missing components
                missing_plans = []
                missing_code = []
                missing_renders = []
                for scene in status['scene_status']:
                    if not scene['has_plan']:
                        missing_plans.append(str(scene['scene_number']))
                    if not scene['has_code']:
                        missing_code.append(str(scene['scene_number']))
                    if not scene['has_render']:
                        missing_renders.append(str(scene['scene_number']))
                
                # Format missing components string
                missing_str = []
                if missing_plans:
                    missing_str.append(f"P:{','.join(missing_plans)}")
                if missing_code:
                    missing_str.append(f"C:{','.join(missing_code)}")
                if missing_renders:
                    missing_str.append(f"R:{','.join(missing_renders)}")
                missing_str = ' '.join(missing_str)
                
                print(f"{status['topic'][:37]+'...' if len(status['topic'])>37 else status['topic']:<40} "
                    f"{'✓' if status['has_scene_outline'] else '✗':<8} "
                    f"{status['total_scenes']:<8} "
                    f"{scene_status_str[:47]+'...' if len(scene_status_str)>47 else scene_status_str:<50} "
                    f"{'✓' if status['has_combined_video'] else '✗':<10} "
                    f"{missing_str[:37]+'...' if len(missing_str)>37 else missing_str:<40}")

            # Print summary
            print("\nSummary:")
            print(f"Total theorems: {len(theorems)}")
            print(f"Total scenes: {sum(status['total_scenes'] for status in all_statuses)}")
            print(f"Scene completion status:")
            print(f"  Plans: {sum(status['implementation_plans'] for status in all_statuses)} scenes")
            print(f"  Code: {sum(status['code_files'] for status in all_statuses)} scenes")
            print(f"  Renders: {sum(status['rendered_scenes'] for status in all_statuses)} scenes")
            print(f"Combined videos: {sum(1 for status in all_statuses if status['has_combined_video'])}/{len(theorems)}")
            exit()

        else:
            # Generate video pipeline from scratch
            print("Generating video pipeline from scratch...")

            async def process_theorem(theorem, topic_semaphore):
                async with topic_semaphore:
                    topic = theorem['theorem']
                    description = theorem['description']
                    print(f"Processing topic: {topic}")
                    if args.only_combine:
                        video_generator.combine_videos(topic)
                    else:
                        await video_generator.generate_video_pipeline(
                            topic, 
                            description, 
                            max_retries=args.max_retries,
                            only_plan=args.only_plan,
                            specific_scenes=args.scenes
                        )
                        if not args.only_plan and not args.only_render:  # Add condition for only_render
                            video_generator.combine_videos(topic)

            async def main():
                # Use the command-line argument for topic concurrency
                topic_semaphore = asyncio.Semaphore(args.max_topic_concurrency)
                tasks = [process_theorem(theorem, topic_semaphore) for theorem in theorems]
                await asyncio.gather(*tasks)

            asyncio.run(main())

    elif args.topic and args.context:
        video_generator = VideoGenerator(
            planner_model=planner_model,
            scene_model=scene_model, # Pass scene_model
            helper_model=helper_model, # Pass helper_model
            output_dir=args.output_dir,
            verbose=args.verbose,
            use_rag=args.use_rag,
            search_provider=args.search_provider,
            use_context_learning=args.use_context_learning,
            context_learning_path=args.context_learning_path,
            chroma_db_path=args.chroma_db_path,
            manim_docs_path=args.manim_docs_path,
            embedding_model=args.embedding_model,
            use_visual_fix_code=args.use_visual_fix_code,
            use_langfuse=args.use_langfuse,
            max_scene_concurrency=args.max_scene_concurrency
        )
        # Process single topic with context
        print(f"Processing topic: {args.topic}")

        if args.only_gen_vid:
            video_generator.render_video_fix_code(args.topic, args.context, max_retries=args.max_retries)
            exit()

        if args.only_combine:
            video_generator.combine_videos(args.topic)
        else:
            asyncio.run(video_generator.generate_video_pipeline(
                args.topic,
                args.context,
                max_retries=args.max_retries,
                only_plan=args.only_plan,
            ))
            if not args.only_plan and not args.only_render:
                video_generator.combine_videos(args.topic)
    else:
        print("Please provide either (--theorems_path) or (--topic and --context)")
        exit()
