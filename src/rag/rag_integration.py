import os
import re
import json
from typing import List, Dict

from mllm_tools.utils import _prepare_text_inputs
from task_generator import (
    get_prompt_rag_query_generation_fix_error,
    get_prompt_detect_plugins,
    get_prompt_rag_query_generation_technical,
    get_prompt_rag_query_generation_vision_storyboard,
    get_prompt_rag_query_generation_narration,
    get_prompt_rag_query_generation_code
)

class RAGIntegration:
    """Class for integrating RAG (Retrieval Augmented Generation) functionality.

    This class handles RAG integration including plugin detection, query generation,
    and document retrieval.

    Args:
        helper_model: Model used for generating queries and processing text
        output_dir (str): Directory for output files
        chroma_db_path (str): Path to ChromaDB
        manim_docs_path (str): Path to Manim documentation
        embedding_model (str): Name of embedding model to use
        use_langfuse (bool, optional): Whether to use Langfuse logging. Defaults to True
        session_id (str, optional): Session identifier. Defaults to None
    """

    def __init__(self, helper_model, output_dir, chroma_db_path, manim_docs_path, embedding_model, use_langfuse=True, session_id=None):
        self.helper_model = helper_model
        self.output_dir = output_dir
        self.manim_docs_path = manim_docs_path
        self.session_id = session_id
        self.relevant_plugins = None

        from src.rag.vector_store import RAGVectorStore
        self.vector_store = RAGVectorStore(
            chroma_db_path=chroma_db_path,
            manim_docs_path=manim_docs_path,
            embedding_model=embedding_model,
            session_id=self.session_id,
            use_langfuse=use_langfuse,
            helper_model=helper_model
        )

    def set_relevant_plugins(self, plugins: List[str]) -> None:
        """Set the relevant plugins for the current video.

        Args:
            plugins (List[str]): List of plugin names to set as relevant
        """
        self.relevant_plugins = plugins

    def detect_relevant_plugins(self, topic: str, description: str) -> List[str]:
        """Detect which plugins might be relevant based on topic and description.

        Args:
            topic (str): Topic of the video
            description (str): Description of the video content

        Returns:
            List[str]: List of detected relevant plugin names
        """
        # Load plugin descriptions
        plugins = self._load_plugin_descriptions()
        if not plugins:
            return []

        # Get formatted prompt using the task_generator function
        prompt = get_prompt_detect_plugins(
            topic=topic,
            description=description,
            plugin_descriptions=json.dumps([{'name': p['name'], 'description': p['description']} for p in plugins], indent=2)
        )

        try:
            response = self.helper_model(
                _prepare_text_inputs(prompt),
                metadata={"generation_name": "detect-relevant-plugins", "tags": [topic, "plugin-detection"], "session_id": self.session_id}
            )
            # Clean the response to ensure it only contains the JSON array
            response = re.search(r'```json(.*)```', response, re.DOTALL).group(1)
            try:
                relevant_plugins = json.loads(response)
            except json.JSONDecodeError as e:
                print(f"JSONDecodeError when parsing relevant plugins: {e}")
                print(f"Response text was: {response}")
                return []

            print(f"LLM detected relevant plugins: {relevant_plugins}")
            return relevant_plugins
        except Exception as e:
            print(f"Error detecting plugins with LLM: {e}")
            return []

    def _load_plugin_descriptions(self) -> list:
        """Load plugin descriptions from JSON file.

        Returns:
            list: List of plugin descriptions, empty list if loading fails
        """
        try:
            plugin_config_path = os.path.join(
                self.manim_docs_path,
                "plugin_docs",
                "plugins.json"
            )
            if os.path.exists(plugin_config_path):
                with open(plugin_config_path, "r") as f:
                    return json.load(f)
            else:
                print(f"Plugin descriptions file not found at {plugin_config_path}")
                return []
        except Exception as e:
            print(f"Error loading plugin descriptions: {e}")
            return []

    def _generate_rag_queries_storyboard(self, scene_plan: str, scene_trace_id: str = None, topic: str = None, scene_number: int = None, session_id: str = None, relevant_plugins: List[str] = []) -> List[str]:
        """Generate RAG queries from the scene plan to help create storyboard.

        Args:
            scene_plan (str): Scene plan text to generate queries from
            scene_trace_id (str, optional): Trace identifier for the scene. Defaults to None
            topic (str, optional): Topic name. Defaults to None
            scene_number (int, optional): Scene number. Defaults to None
            session_id (str, optional): Session identifier. Defaults to None
            relevant_plugins (List[str], optional): List of relevant plugins. Defaults to empty list

        Returns:
            List[str]: List of generated RAG queries
        """
        cache_key = f"{topic}_scene{scene_number}_storyboard_rag"
        cache_dir = os.path.join(self.output_dir, re.sub(r'[^a-z0-9_]+', '_', topic.lower()), f"scene{scene_number}", "rag_cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, "rag_queries_storyboard.json")

        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                return json.load(f)

        # Format relevant plugins as a string
        plugins_str = ", ".join(relevant_plugins) if relevant_plugins else "No plugins are relevant."
        
        # Generate the prompt with only the required arguments
        prompt = get_prompt_rag_query_generation_vision_storyboard(
            scene_plan=scene_plan,
            relevant_plugins=plugins_str
        )
        
        queries = self.helper_model(
            _prepare_text_inputs(prompt),
            metadata={"generation_name": "rag_query_generation_storyboard", "trace_id": scene_trace_id, "tags": [topic, f"scene{scene_number}"], "session_id": session_id}
        )

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

    def _generate_rag_queries_technical(self, storyboard: str, scene_trace_id: str = None, topic: str = None, scene_number: int = None, session_id: str = None, relevant_plugins: List[str] = []) -> List[str]:
        """Generate RAG queries from the storyboard to help create technical implementation.

        Args:
            storyboard (str): Storyboard text to generate queries from
            scene_trace_id (str, optional): Trace identifier for the scene. Defaults to None
            topic (str, optional): Topic name. Defaults to None
            scene_number (int, optional): Scene number. Defaults to None
            session_id (str, optional): Session identifier. Defaults to None
            relevant_plugins (List[str], optional): List of relevant plugins. Defaults to empty list

        Returns:
            List[str]: List of generated RAG queries
        """
        cache_key = f"{topic}_scene{scene_number}_technical_rag"
        cache_dir = os.path.join(self.output_dir, re.sub(r'[^a-z0-9_]+', '_', topic.lower()), f"scene{scene_number}", "rag_cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, "rag_queries_technical.json")

        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                return json.load(f)

        prompt = get_prompt_rag_query_generation_technical(
            storyboard=storyboard,
            relevant_plugins=", ".join(relevant_plugins) if relevant_plugins else "No plugins are relevant."
        )
        
        queries = self.helper_model(
            _prepare_text_inputs(prompt),
            metadata={"generation_name": "rag_query_generation_technical", "trace_id": scene_trace_id, "tags": [topic, f"scene{scene_number}"], "session_id": session_id}
        )

        try: # add try-except block to handle potential json decode errors
            queries = re.search(r'```json(.*)```', queries, re.DOTALL).group(1)
            queries = json.loads(queries)
        except json.JSONDecodeError as e:
            print(f"JSONDecodeError when parsing RAG queries for technical implementation: {e}")
            print(f"Response text was: {queries}")
            return [] # Return empty list in case of parsing error

        # Cache the queries
        with open(cache_file, 'w') as f:
            json.dump(queries, f)

        return queries

    def _generate_rag_queries_narration(self, storyboard: str, scene_trace_id: str = None, topic: str = None, scene_number: int = None, session_id: str = None, relevant_plugins: List[str] = []) -> List[str]:
        """Generate RAG queries from the storyboard to help create narration plan.

        Args:
            storyboard (str): Storyboard text to generate queries from
            scene_trace_id (str, optional): Trace identifier for the scene. Defaults to None
            topic (str, optional): Topic name. Defaults to None
            scene_number (int, optional): Scene number. Defaults to None
            session_id (str, optional): Session identifier. Defaults to None
            relevant_plugins (List[str], optional): List of relevant plugins. Defaults to empty list

        Returns:
            List[str]: List of generated RAG queries
        """
        cache_key = f"{topic}_scene{scene_number}_narration_rag"
        cache_dir = os.path.join(self.output_dir, re.sub(r'[^a-z0-9_]+', '_', topic.lower()), f"scene{scene_number}", "rag_cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, "rag_queries_narration.json")

        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                return json.load(f)

        prompt = get_prompt_rag_query_generation_narration(
            storyboard=storyboard,
            relevant_plugins=", ".join(relevant_plugins) if relevant_plugins else "No plugins are relevant."
        )
        
        queries = self.helper_model(
            _prepare_text_inputs(prompt),
            metadata={"generation_name": "rag_query_generation_narration", "trace_id": scene_trace_id, "tags": [topic, f"scene{scene_number}"], "session_id": session_id}
        )

        try: # add try-except block to handle potential json decode errors
            queries = re.search(r'```json(.*)```', queries, re.DOTALL).group(1)
            queries = json.loads(queries)
        except json.JSONDecodeError as e:
            print(f"JSONDecodeError when parsing narration RAG queries: {e}")
            print(f"Response text was: {queries}")
            return [] # Return empty list in case of parsing error

        # Cache the queries
        with open(cache_file, 'w') as f:
            json.dump(queries, f)

        return queries

    def get_relevant_docs(self, rag_queries: List[Dict], scene_trace_id: str, topic: str, scene_number: int) -> List[str]:
        """Get relevant documentation using the vector store.

        Args:
            rag_queries (List[Dict]): List of RAG queries to search for
            scene_trace_id (str): Trace identifier for the scene
            topic (str): Topic name
            scene_number (int): Scene number

        Returns:
            List[str]: List of relevant documentation snippets
        """
        return self.vector_store.find_relevant_docs(
            queries=rag_queries,
            k=2,
            trace_id=scene_trace_id,
            topic=topic,
            scene_number=scene_number
        )
    
    def _generate_rag_queries_code(self, implementation_plan: str, scene_trace_id: str = None, topic: str = None, scene_number: int = None, relevant_plugins: List[str] = None) -> List[str]:
        """Generate RAG queries from implementation plan.

        Args:
            implementation_plan (str): Implementation plan text to generate queries from
            scene_trace_id (str, optional): Trace identifier for the scene. Defaults to None
            topic (str, optional): Topic name. Defaults to None
            scene_number (int, optional): Scene number. Defaults to None
            relevant_plugins (List[str], optional): List of relevant plugins. Defaults to None

        Returns:
            List[str]: List of generated RAG queries
        """
        cache_key = f"{topic}_scene{scene_number}"
        cache_dir = os.path.join(self.output_dir, re.sub(r'[^a-z0-9_]+', '_', topic.lower()), f"scene{scene_number}", "rag_cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, "rag_queries_code.json")

        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                return json.load(f)

        prompt = get_prompt_rag_query_generation_code(
            implementation_plan=implementation_plan,
            relevant_plugins=", ".join(relevant_plugins) if relevant_plugins else "No plugins are relevant."
        )

        try:
            response = self.helper_model(
                _prepare_text_inputs(prompt),
                metadata={"generation_name": "rag_query_generation_code", "trace_id": scene_trace_id, "tags": [topic, f"scene{scene_number}"], "session_id": self.session_id}
            )
            
            # Clean and parse response
            response = re.search(r'```json(.*)```', response, re.DOTALL).group(1)
            queries = json.loads(response)

            # Cache the queries
            with open(cache_file, 'w') as f:
                json.dump(queries, f)

            return queries
        except Exception as e:
            print(f"Error generating RAG queries: {e}")
            return []

    def _generate_rag_queries_error_fix(self, error: str, code: str, scene_trace_id: str = None, topic: str = None, scene_number: int = None, session_id: str = None) -> List[str]:
        """Generate RAG queries for fixing code errors.

        Args:
            error (str): Error message to generate queries from
            code (str): Code containing the error
            scene_trace_id (str, optional): Trace identifier for the scene. Defaults to None
            topic (str, optional): Topic name. Defaults to None
            scene_number (int, optional): Scene number. Defaults to None
            session_id (str, optional): Session identifier. Defaults to None

        Returns:
            List[str]: List of generated RAG queries
        """
        if self.relevant_plugins is None:
            print("Warning: No plugins have been detected yet")
            plugins_str = "No plugins are relevant."
        else:
            plugins_str = ", ".join(self.relevant_plugins) if self.relevant_plugins else "No plugins are relevant."

        cache_key = f"{topic}_scene{scene_number}_error_fix"
        cache_dir = os.path.join(self.output_dir, re.sub(r'[^a-z0-9_]+', '_', topic.lower()), f"scene{scene_number}", "rag_cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, "rag_queries_error_fix.json")

        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                cached_queries = json.load(f)
                print(f"Using cached RAG queries for error fix in {cache_key}")
                return cached_queries

        prompt = get_prompt_rag_query_generation_fix_error(
            error=error, 
            code=code, 
            relevant_plugins=plugins_str
        )

        queries = self.helper_model(
            _prepare_text_inputs(prompt),
            metadata={"generation_name": "rag-query-generation-fix-error", "trace_id": scene_trace_id, "tags": [topic, f"scene{scene_number}"], "session_id": session_id}
        )


        try:  
            # retrieve json triple backticks
            queries = re.search(r'```json(.*)```', queries, re.DOTALL).group(1)
            queries = json.loads(queries)
        except json.JSONDecodeError as e:
            print(f"JSONDecodeError when parsing RAG queries for error fix: {e}")
            print(f"Response text was: {queries}")
            return []

        # Cache the queries
        with open(cache_file, 'w') as f:
            json.dump(queries, f)

        return queries