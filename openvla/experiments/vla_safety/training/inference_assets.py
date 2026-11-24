"""Copy small local inference assets and check custom Hugging Face entry points."""

import json
from pathlib import Path
import shutil


CONFIGS = ("config.json", "processor_config.json", "preprocessor_config.json", "tokenizer_config.json")
PATTERNS = ("*token*", "*processor*.json", "special_tokens_map.json", "added_tokens.json",
            "dataset_statistics.json", "*.model", "*.py")


def validate_inference_sources(directory):
    """Fail before GPU loading if a local auto_map points to missing Python code."""
    directory = Path(directory).resolve()
    for name in CONFIGS:
        config = directory / name
        if not config.is_file():
            continue
        for value in json.loads(config.read_text()).get("auto_map", {}).values():
            for reference in value if isinstance(value, (tuple, list)) else (value,):
                if reference is None:
                    continue
                # Some HF configs qualify a module with a repository name.
                module, _, class_name = reference.split("--")[-1].rpartition(".")
                source = directory / (module.replace(".", "/") + ".py")
                if not module or not class_name or not source.is_relative_to(directory):
                    raise ValueError(f"Invalid local auto_map entry in {config}: {reference}")
                if not source.is_file() or source.stat().st_size == 0:
                    raise ValueError(f"Missing inference source {source.name}, referenced by {config.name}")


def copy_inference_assets(checkpoint, output):
    """Include processor code as well as tokenizer/config assets; never copy weights."""
    checkpoint, output = Path(checkpoint).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    copied = []
    if checkpoint != output:
        files = {source for pattern in PATTERNS for source in checkpoint.glob(pattern) if source.is_file()}
        for source in sorted(files):
            destination = output / source.name
            if not destination.is_file() or destination.read_bytes() != source.read_bytes():
                shutil.copy2(source, destination)
                copied.append(source.name)
    validate_inference_sources(output)
    return copied
