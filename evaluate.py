import os
import json
import argparse
import tempfile
from typing import Dict, List, Union
from datetime import datetime

from dotenv import load_dotenv
from moviepy import VideoFileClip

from mllm_tools.litellm import LiteLLMWrapper
from mllm_tools.gemini import GeminiWrapper
from eval_suite.utils import calculate_geometric_mean
from eval_suite.text_utils import parse_srt_to_text, fix_transcript, evaluate_text
from eval_suite.video_utils import evaluate_video_chunk_new
from eval_suite.image_utils import evaluate_sampled_images

load_dotenv()

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", "utils", "allowed_models.json")) as f:
    ALLOWED_MODELS = json.load(f)["allowed_models"]

DEFAULT_TEXT_MODEL = "minimax/MiniMax-M3.1-Flash"


def combine_results(output_folder: str, combined_file: str, results: Dict[str, Dict]) -> None:
    """
    Combine all evaluation results into a single file.

    Args:
        output_folder (str): Directory to store the combined file.
        combined_file (str): Name of the combined file.
        results (Dict[str, Dict]): Dictionary of evaluation results with file names as keys.

    Returns:
        None
    """
    combined_path = os.path.join(output_folder, combined_file)
    with open(combined_path, 'w') as output_file:
        json.dump(results, output_file, indent=4)


def save_individual_result(output_folder: str, file_name: str, result: Dict) -> None:
    """
    Save individual evaluation result to a file.

    Args:
        output_folder (str): Directory to store the evaluation file.
        file_name (str): Name of the file.
        result (Dict): Evaluation result.

    Returns:
        None
    """
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = f"evaluation_{file_name}_{current_time}.json"
    os.makedirs(output_folder, exist_ok=True)
    result_path = os.path.join(output_folder, result_file)
    with open(result_path, 'w') as output_file:
        json.dump(result, output_file, indent=4)


def evaluate_text_file(model, transcript_path, retry_limit):
    """
    Evaluate a text file using the provided model.

    Args:
        model: The model to use for evaluation.
        transcript_path (str): Path to the transcript file (.srt or .txt).
        retry_limit (int): Number of retry attempts for evaluation.

    Returns:
        Dict or None: Evaluation results if successful, None if file format unsupported.
    """
    if not transcript_path.endswith(('.srt', '.txt')):
        print(f"Skipping {transcript_path}: Unsupported file format for text evaluation.")
        return None

    if transcript_path.endswith(".srt"):
        transcript = parse_srt_to_text(transcript_path)
    elif transcript_path.endswith(".txt"):
        with open(transcript_path) as f:
            transcript = f.read().strip()
    else:
        raise ValueError("Unrecognized transcript file format.")

    capital_letter_proportion = sum(1 for c in transcript if c.isupper()) / sum(1 for c in transcript if c.isalpha())
    if capital_letter_proportion < 0.01:
        transcript = fix_transcript(model, transcript)

    print(f"Performing text evaluation: {os.path.basename(transcript_path)}")
    result = evaluate_text(model, transcript, retry_limit)
    return result


def evaluate_video_file(model, video_path, transcript_path, description_path, target_fps=None, output_folder=None):
    """
    Evaluate a video file using the provided model.

    Args:
        model: The model to use for evaluation.
        video_path (str): Path to the video file.
        transcript_path (str): Path to the transcript file.
        description_path (str): Path to the description file.
        target_fps (int, optional): Target frames per second for video processing.
        output_folder (str, optional): Directory to store output files.

    Returns:
        Dict or None: Evaluation results if successful, None if file format unsupported.
    """
    if not video_path.endswith(('.mp4', '.mkv')):
        print(f"Skipping {video_path}: Unsupported file format for video evaluation.")
        return None

    moviepy_temp_dir = os.path.join(output_folder, "moviepy_temp")

    # Chunking
    num_chunks = 10
    with VideoFileClip(video_path) as clip:
        duration = clip.duration
        chunk_duration = duration / num_chunks
        results = []
        
        # Create a temporary directory in the output_folder
        temp_dir_parent = output_folder or os.getcwd()
        with tempfile.TemporaryDirectory(dir=temp_dir_parent) as temp_dir:
            for i in range(10):
                start = i * chunk_duration
                end = min(start + chunk_duration, duration)
                chunk = clip.subclipped(start, end)
                chunk_path = os.path.join(temp_dir, f"chunk_{i+1}.mp4")
                # Explicitly set the temp_audiofile path with matching codec
                temp_audiofile = os.path.join(moviepy_temp_dir, f"temp_audio_chunk_{i+1}.m4a")
                chunk.write_videofile(
                    chunk_path,
                    codec="libx264",
                    audio_codec="aac",
                    temp_audiofile=temp_audiofile,
                    audio_bitrate="192k",
                    preset="ultrafast",  # Speed up encoding
                    logger=None
                )
                # Create processed videos folder inside output_folder
                processed_videos_dir = os.path.join(output_folder, "processed_videos")
                save_path = os.path.join(processed_videos_dir, f"processed_chunk_{i+1}.mp4")
                result = evaluate_video_chunk_new(
                    model,
                    chunk_path,
                    transcript_path,
                    description_path,
                    target_fps=target_fps,
                    save_processed_video=save_path
                )
                results.append(result)

    score_dict = {}
    for key in results[0]["evaluation"].keys():
        score_dict[key] = []
        for result in results:
            score_dict[key].append(result["evaluation"][key]["score"])

    evaluation = {}
    for key, scores in score_dict.items():
        evaluation[key] = {"score": calculate_geometric_mean(scores)}

    result_json = {
        "evaluation": evaluation,
        "video_chunks": results
    }
    return result_json


def extract_scores(data: Union[Dict, List]) -> List[int]:
    """
    Extract all score values from a nested dictionary or list structure.

    Args:
        data (Union[Dict, List]): The data structure to extract scores from.

    Returns:
        List[int]: List of extracted score values.
    """
    scores = []
    if isinstance(data, dict):
        for key, value in data.items():
            if "chunks" in key:
                continue
            elif isinstance(value, dict) or isinstance(value, list):
                scores.extend(extract_scores(value))
            elif key == 'score':
                scores.append(value)
    elif isinstance(data, list):
        for item in data:
            scores.extend(extract_scores(item))
    return scores


def calculate_overall_score(result: Dict) -> float:
    """
    Calculate the overall score from evaluation results.

    Args:
        result (Dict): Dictionary containing evaluation results.

    Returns:
        float: The calculated overall score.
    """
    scores = extract_scores(result)
    overall_score = calculate_geometric_mean(scores)
    return overall_score


def process_topic_name(topic_name: str) -> str:
    """
    Process a topic name by capitalizing words and handling special characters.

    Args:
        topic_name (str): The topic name to process.

    Returns:
        str: The processed topic name.
    """
    words = topic_name.replace("_s_", "'s_").split("_")
    return " ".join([word.capitalize() for word in words])


def merge_dicts(dict1: dict, dict2: dict) -> dict:
    """
    Recursively merge two dictionaries.

    Args:
        dict1 (dict): First dictionary.
        dict2 (dict): Second dictionary.

    Returns:
        dict: Merged dictionary.
    """
    merged = dict1.copy()
    for key, value in dict2.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def process_theorem(models, file_path: str, eval_type: str, retry_limit: int,
                    target_fps: int = None, use_parent_folder_as_topic: bool = False,
                    output_folder: str = None) -> tuple[str, dict]:
    """
    Process a theorem file or directory for evaluation.

    Args:
        models: Dictionary of models for different evaluation types.
        file_path (str): Path to the file or directory to evaluate.
        eval_type (str): Type of evaluation to perform.
        retry_limit (int): Number of retry attempts.
        target_fps (int, optional): Target frames per second for video processing.
        use_parent_folder_as_topic (bool, optional): Use parent folder name as topic.
        output_folder (str, optional): Directory to store output files.

    Returns:
        tuple[str, dict]: Tuple of file name and evaluation results.
    """
    ext_map = {
        'text': ('.txt', '.srt'),
        'video': ('.mp4', '.mkv')
    }

    # Handle single file evaluation
    if os.path.isfile(file_path):
        file_ext = os.path.splitext(file_path)[1].lower()
        file_name = os.path.basename(file_path)

        if eval_type == "text" and file_ext in ext_map['text']:
            return file_name, evaluate_text_file(models['text'], file_path, retry_limit)
        elif eval_type == "video" and file_ext in ext_map['video']:
            if use_parent_folder_as_topic:
                topic_name = os.path.basename(os.path.dirname(file_path))
            else:
                topic_name = None
            topic_name = process_topic_name(topic_name)
            return file_name, evaluate_video_file(models['video'], file_path, None, topic_name, target_fps, output_folder)
        elif eval_type == "image" and file_ext in ext_map['video']:
            if use_parent_folder_as_topic:
                topic_name = os.path.basename(os.path.dirname(file_path))
            else:
                topic_name = None
            topic_name = process_topic_name(topic_name)
            return file_name, evaluate_sampled_images(models['image'], file_path, topic_name, num_chunks=10, output_folder=output_folder)
        elif eval_type == "all":
            raise ValueError("Evaluation type 'all' is not supported for a single file. Try passing a folder with both a video and a subtitle file.")
        else:
            raise ValueError(f"File type of {file_path} does not match evaluation type {eval_type!r}")

    # Handle directory evaluation
    theorem_dir = file_path
    all_files = os.listdir(theorem_dir)

    # Look for transcript files, prioritizing .srt over .txt if both exist
    transcript_file_candidates = [f for f in all_files if f.endswith(ext_map['text']) and not f.endswith('_scene_outline.txt')]
    srt_files = [f for f in transcript_file_candidates if f.endswith('.srt')]
    txt_files = [f for f in transcript_file_candidates if f.endswith('.txt')]

    transcript_path = None
    if srt_files:
        transcript_path = os.path.join(theorem_dir, srt_files[0])
    elif txt_files:
        transcript_path = os.path.join(theorem_dir, txt_files[0])

    video_file_candidates = [f for f in all_files if f.endswith(ext_map['video'])]
    video_path = os.path.join(theorem_dir, video_file_candidates[0]) if len(video_file_candidates) == 1 else None

    topic_name = os.path.basename(theorem_dir)
    topic_name = process_topic_name(topic_name)

    if not video_path:
        print(f"Skipping {theorem_dir}: No video file found")
        return None, None

    text_result = video_result = image_result = None
    if eval_type == "text" or eval_type == "all":
        if transcript_path is None:
            print(f"Warning: No suitable transcript file found in {theorem_dir}")
        else:
            text_result = evaluate_text_file(models['text'], transcript_path, retry_limit)
    if eval_type == "video" or eval_type == "all":
        assert video_path is not None, f"Expected 1 video file, got {len(video_file_candidates)} for {theorem_dir}"
        video_result = evaluate_video_file(models['video'], video_path, transcript_path, topic_name, target_fps, output_folder)
    if eval_type == "image" or eval_type == "all":
        assert video_path is not None, f"Expected 1 video file, got {len(video_file_candidates)} for {theorem_dir}"
        image_result = evaluate_sampled_images(models['image'], video_path, topic_name, num_chunks=10, output_folder=output_folder)
    
    if eval_type == "all":
        result = {}
        if text_result:
            result = merge_dicts(result, text_result)
        if video_result:
            result = merge_dicts(result, video_result)
        if image_result:
            result = merge_dicts(result, image_result)
        if result:
            result["evaluation"]["overall_score"] = calculate_overall_score(result)
    else:
        result = text_result if eval_type == "text" else video_result if eval_type == "video" else image_result if eval_type == "image" else None

    file_name = os.path.basename(theorem_dir)
    return file_name, result


def main():
    """
    Main function to run the evaluation script.

    Parses command line arguments and orchestrates the evaluation process
    for text, video, and image content using specified AI models.
    """
    parser = argparse.ArgumentParser(description='Automatic evaluation of theorem explanation videos with LLMs')
    parser.add_argument('--model_text', type=str, 
                       choices=ALLOWED_MODELS,
                       default=DEFAULT_TEXT_MODEL,
                       help='Select the AI model to use for text evaluation')
    parser.add_argument('--model_video', type=str,
                       choices=['gemini/gemini-1.5-pro-002',
                                'gemini/gemini-2.0-flash-exp',
                                'gemini/gemini-2.0-pro-exp-02-05'],
                       default='gemini/gemini-1.5-pro-002',
                       help='Select the AI model to use for video evaluation')
    parser.add_argument('--model_image', type=str,
                       choices=ALLOWED_MODELS,
                       default='azure/gpt-4o',
                       help='Select the AI model to use for image evaluation')
    parser.add_argument('--eval_type', type=str, choices=['text', 'video', 'image', 'all'], default='all', help='Type of evaluation to perform')
    parser.add_argument('--file_path', type=str, help='Path to a file or a theorem folder', required=True)
    parser.add_argument('--output_folder', type=str, help='Directory to store the evaluation files', required=True)
    parser.add_argument('--retry_limit', type=int, default=3, help='Number of retry attempts for each inference')
    parser.add_argument('--combine', action='store_true', help='Combine all results into a single JSON file')
    parser.add_argument('--bulk_evaluate', action='store_true', help='Evaluate a folder of theorems together', default=False)
    parser.add_argument('--target_fps', type=int, help='Target FPS for video processing. If not set, original video FPS will be used', required=False)
    parser.add_argument('--use_parent_folder_as_topic', action='store_true', help='Use parent folder name as topic name for single file evaluation', default=True)
    parser.add_argument('--max_workers', type=int, default=4, help='Maximum number of concurrent workers for parallel processing')

    args = parser.parse_args()

    # Initialize separate models
    text_model = LiteLLMWrapper(
        model_name=args.model_text,
        temperature=0.0,
    )
    video_model = GeminiWrapper(
        model_name=args.model_video,
        temperature=0.0,
    )
    image_model = LiteLLMWrapper(
        model_name=args.model_image,
        temperature=0.0,
    )

    models = {
        'text': text_model,
        'video': video_model,
        'image': image_model
    }

    theorem_dirs = []
    if args.bulk_evaluate:
        assert os.path.isdir(args.file_path), "File path must be a folder for --bulk_evaluate"
        for root, dirnames, _ in os.walk(args.file_path):
            if not any(f.endswith(".mp4") for f in os.listdir(root)):
                continue

            theorem_dirs.append(root)
    elif os.path.isdir(args.file_path):
        assert any(f.endswith(".mp4") for f in os.listdir(args.file_path)), "The provided folder must contain a video file"

        theorem_dirs.append(args.file_path)

    # Create output directory and its temp subdirectories if it doesn't exist
    os.makedirs(args.output_folder, exist_ok=True)
    moviepy_temp_dir = os.path.join(args.output_folder, "moviepy_temp")
    os.makedirs(moviepy_temp_dir, exist_ok=True)
    VideoFileClip.DEFAULT_TEMP_DIR = moviepy_temp_dir

    processed_videos_dir = os.path.join(args.output_folder, "processed_videos")
    os.makedirs(processed_videos_dir, exist_ok=True)

    results = {}
    if theorem_dirs:
        for theorem_dir in theorem_dirs:
            file_name, result = process_theorem(
                models,
                theorem_dir,
                args.eval_type,
                args.retry_limit,
                args.target_fps,
                args.use_parent_folder_as_topic,
                args.output_folder
            )

            if result is not None:
                results[file_name] = result

                if not args.combine:
                    save_individual_result(args.output_folder, file_name, result)
    else:
        file_name, result = process_theorem(
            models, 
            args.file_path, 
            args.eval_type, 
            args.retry_limit,
            args.target_fps,
            args.use_parent_folder_as_topic,
            args.output_folder
        )
        
        if result is not None:
            results[file_name] = result

            if not args.combine:
                save_individual_result(args.output_folder, file_name, result)

    if args.combine:
        if len(results) > 1:
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            combined_file = f"evaluation_{current_time}.json"
            combine_results(args.output_folder, combined_file, results)
            print("Combining results completed.")
        else:
            for file_name, result in results.items():
                save_individual_result(args.output_folder, file_name, result)
    
    os.rmdir(moviepy_temp_dir)


if __name__ == "__main__":
    main()
