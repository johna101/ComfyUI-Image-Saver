# ComfyUI-Image-Saver — Metadata Compiler Node
# Assembles generation metadata into a structured JSON dict for embedding in image files.

import os
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import folder_paths
from nodes import MAX_RESOLUTION

from ..services.hashing import get_sha256
from ..services.file_utils import full_checkpoint_path_for, parse_checkpoint_name_without_extension
from ..services.civitai import get_civitai_sampler_name, get_civitai_metadata, MAX_HASH_LENGTH
from ..services.prompt_parser import PromptMetadataExtractor


@dataclass
class Metadata:
    model_name: str
    positive: str
    negative: str
    width: int
    height: int
    seed: int
    steps: int
    cfg: float
    sampler_name: str
    scheduler_name: str
    denoise: float
    clip_skip: int
    additional_hashes: str
    ckpt_path: str
    gallery_metadata: dict = field(default_factory=dict)
    final_hashes: str = ''


class MetadataCompiler:
    """Compiles generation metadata into structured JSON for gallery consumption."""

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "optional": {
                "model_name":            ("STRING",  {"default": '', "multiline": False,                           "tooltip": "model name (can be multiple, separated by commas)"}),
                "positive":              ("STRING",  {"default": '', "multiline": True, "placeholder": "positive prompt",  "tooltip": "positive prompt"}),
                "negative":              ("STRING",  {"default": '', "multiline": True, "placeholder": "negative prompt", "tooltip": "negative prompt"}),
                "width":                 ("INT",     {"default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 8,  "tooltip": "image width"}),
                "height":                ("INT",     {"default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 8,  "tooltip": "image height"}),
                "seed_value":            ("INT",     {"default": 0, "min": 0, "max": 0xffffffffffffffff,           "tooltip": "seed"}),
                "steps":                 ("INT",     {"default": 20, "min": 1, "max": 10000,                       "tooltip": "number of steps"}),
                "cfg":                   ("FLOAT",   {"default": 7.0, "min": 0.0, "max": 100.0,                    "tooltip": "CFG value"}),
                "sampler_name":          ("STRING",  {"default": '', "multiline": False,                           "tooltip": "sampler name (as string)"}),
                "scheduler_name":        ("STRING",  {"default": 'normal', "multiline": False,                     "tooltip": "scheduler name (as string)"}),
                "denoise":               ("FLOAT",   {"default": 1.0, "min": 0.0, "max": 1.0,                      "tooltip": "denoise value"}),
                "clip_skip":             ("INT",     {"default": 0, "min": -24, "max": 24,                         "tooltip": "skip last CLIP layers (positive or negative value, 0 for no skip)"}),
                "additional_hashes":     ("STRING",  {"default": "", "multiline": False,                           "tooltip": "hashes separated by commas, optionally with names. 'Name:HASH' (e.g., 'MyLoRA:FF735FF83F98')\nWith download_civitai_data set to true, weights can be added as well. (e.g., 'HASH:Weight', 'Name:HASH:Weight')"}),
                "download_civitai_data": ("BOOLEAN", {"default": True,                                             "tooltip": "Download and cache data from civitai.com to save correct metadata. Allows LoRA weights to be saved to the metadata."}),
            },
        }

    RETURN_TYPES = ("METADATA", "STRING", "STRING")
    RETURN_NAMES = ("metadata", "hashes", "gallery_metadata_json")
    OUTPUT_TOOLTIPS = ("metadata for Image Saver", "Comma-separated list of hashes", "Gallery metadata as JSON string")
    FUNCTION = "get_metadata"
    CATEGORY = "ImageSaver"
    DESCRIPTION = "Prepare metadata for Image Saver"

    def get_metadata(
        self,
        model_name: str = "",
        positive: str = "",
        negative: str = "",
        width: int = 512,
        height: int = 512,
        seed_value: int = 0,
        steps: int = 20,
        cfg: float = 7.0,
        sampler_name: str = "",
        scheduler_name: str = "normal",
        denoise: float = 1.0,
        clip_skip: int = 0,
        additional_hashes: str = "",
        download_civitai_data: bool = True,
    ) -> tuple[Metadata, str, str]:
        metadata = MetadataCompiler.make_metadata(
            model_name, positive, negative, width, height, seed_value, steps, cfg,
            sampler_name, scheduler_name, denoise, clip_skip,
            additional_hashes, download_civitai_data
        )
        return (metadata, metadata.final_hashes, json.dumps(metadata.gallery_metadata))

    @staticmethod
    def make_metadata(model_name: str, positive: str, negative: str, width: int, height: int,
                      seed_value: int, steps: int, cfg: float, sampler_name: str,
                      scheduler_name: str, denoise: float, clip_skip: int,
                      additional_hashes: str, download_civitai_data: bool) -> Metadata:
        model_name, additional_hashes = get_multiple_models(model_name, additional_hashes)

        ckpt_path = full_checkpoint_path_for(model_name)
        if ckpt_path:
            modelhash = get_sha256(ckpt_path)[:10]
        else:
            modelhash = ""

        metadata_extractor = PromptMetadataExtractor([positive, negative])
        embeddings = metadata_extractor.get_embeddings()
        loras = metadata_extractor.get_loras()
        basemodelname = parse_checkpoint_name_without_extension(model_name)

        # Get existing hashes from model, loras, and embeddings
        existing_hashes = {modelhash.lower()} | {t[2].lower() for t in loras.values()} | {t[2].lower() for t in embeddings.values()}
        # Parse manual hashes
        manual_entries = parse_manual_hashes(additional_hashes, existing_hashes, download_civitai_data)
        # Get Civitai metadata
        civitai_resources, hashes, add_model_hash = get_civitai_metadata(
            model_name, ckpt_path, modelhash, loras, embeddings, manual_entries, download_civitai_data
        )

        # Build structured gallery metadata
        gallery_metadata = {
            "model": basemodelname,
            "model_path": model_name,
            "sampler": sampler_name,
            "scheduler": scheduler_name,
            "cfg": cfg,
            "steps": steps,
            "seed": seed_value,
            "size": [width, height],
            "denoise": denoise,
            "positive": positive.strip(),
            "negative": negative.strip(),
            "version": "ComfyUI",
        }

        # Optional fields — only include when present
        if clip_skip != 0:
            gallery_metadata["clip_skip"] = abs(clip_skip)

        if add_model_hash:
            gallery_metadata["model_hash"] = add_model_hash

        if hashes:
            gallery_metadata["hashes"] = hashes

        if civitai_resources:
            gallery_metadata["civitai_resources"] = civitai_resources

        # Build final hash string for chaining
        all_resources = {model_name: (ckpt_path, None, modelhash)} | loras | embeddings | manual_entries

        hash_parts = []
        for name, (_, weight, hash_value) in all_resources.items():
            if not hash_value:
                continue

            if name:
                filename = name.split(':')[-1]
                name_without_ext, ext = os.path.splitext(filename)
                supported_extensions = folder_paths.supported_pt_extensions | {".gguf"}
                clean_name = name_without_ext if ext.lower() in supported_extensions else filename
                name_part = f"{clean_name}:"
            else:
                name_part = ""

            weight_part = f":{weight}" if weight is not None and download_civitai_data else ""
            hash_parts.append(f"{name_part}{hash_value}{weight_part}")

        final_hashes = ",".join(hash_parts)

        metadata = Metadata(
            model_name=model_name, positive=positive, negative=negative,
            width=width, height=height, seed=seed_value, steps=steps, cfg=cfg,
            sampler_name=sampler_name, scheduler_name=scheduler_name,
            denoise=denoise, clip_skip=clip_skip,
            additional_hashes=additional_hashes, ckpt_path=ckpt_path,
            gallery_metadata=gallery_metadata, final_hashes=final_hashes
        )
        return metadata


# --- Helper functions ---

re_manual_hash = re.compile(r'^\s*([^:]+?)(?:\s*:\s*([^\s:][^:]*?))?\s*$')
re_manual_hash_weights = re.compile(r'^\s*([^:]+?)(?:\s*:\s*([^\s:][^:]*?))?(?:\s*:\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)))?\s*$')


def get_multiple_models(model_name: str, additional_hashes: str) -> tuple[str, str]:
    """Parse comma-separated model names. First becomes primary, rest added to additional_hashes."""
    model_names = [m.strip() for m in model_name.split(',')]
    model_name = model_names[0]

    for additional_model in model_names[1:]:
        additional_ckpt_path = full_checkpoint_path_for(additional_model)
        if additional_ckpt_path:
            additional_modelhash = get_sha256(additional_ckpt_path)[:10]
            if additional_hashes:
                additional_hashes += ","
            additional_hashes += f"{additional_model}:{additional_modelhash}"
    return model_name, additional_hashes


def parse_manual_hashes(additional_hashes: str, existing_hashes: set[str], download_civitai_data: bool) -> dict[str, tuple[str | None, float | None, str]]:
    """Process additional_hashes input string into normalized dict."""
    manual_entries: dict[str, tuple[str | None, float | None, str]] = {}
    unnamed_count = 0

    additional_hash_split = additional_hashes.replace("\n", ",").split(",") if additional_hashes else []
    for entry in additional_hash_split:
        match = (re_manual_hash_weights if download_civitai_data else re_manual_hash).search(entry)
        if match is None:
            print(f"ComfyUI-Image-Saver: Invalid additional hash string: '{entry}'")
            continue

        groups = tuple(group for group in match.groups() if group)

        weight = None
        if download_civitai_data and len(groups) > 1:
            try:
                weight = float(groups[-1])
                groups = groups[:-1]
            except (ValueError, TypeError):
                pass

        name, hash = groups if len(groups) > 1 else (None, groups[0])

        if len(hash) > MAX_HASH_LENGTH:
            print(f"ComfyUI-Image-Saver: Skipping hash. Length exceeds maximum of {MAX_HASH_LENGTH} characters: {hash}")
            continue

        if any(hash.lower() == existing_hash.lower() for _, _, existing_hash in manual_entries.values()):
            print(f"ComfyUI-Image-Saver: Skipping duplicate hash: {hash}")
            continue

        if hash.lower() in existing_hashes:
            print(f"ComfyUI-Image-Saver: Skipping manual hash already present in resources: {hash}")
            continue

        if name is None:
            unnamed_count += 1
            name = f"manual{unnamed_count}"
        elif name in manual_entries:
            print(f"ComfyUI-Image-Saver: Duplicate manual hash name '{name}' is being overwritten.")

        manual_entries[name] = (None, weight, hash)

        if len(manual_entries) > 29:
            print("ComfyUI-Image-Saver: Reached maximum limit of 30 manual hashes. Skipping the rest.")
            break

    return manual_entries
