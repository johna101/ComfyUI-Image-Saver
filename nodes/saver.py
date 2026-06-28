# ComfyUI-Image-Saver — Image Saver Nodes
# File writing, image encoding, and batch processing.

import os
import json
from typing import Any

import numpy as np
from PIL import Image
import torch

import folder_paths
from nodes import MAX_RESOLUTION

from .metadata import Metadata, MetadataCompiler
from ..services.file_utils import make_pathname, make_filename, save_json
from ..services.image_encoder import save_image


class ImageSaver:
    """Save images with CivitAI-compatible generation metadata."""

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "images":                ("IMAGE",   {                                                             "tooltip": "image(s) to save"}),
                "filename":              ("STRING",  {"default": '%time_%basemodelname_%seed', "multiline": False, "tooltip": "filename (available variables: %date, %time, %time_format<format>, %model, %width, %height, %seed, %counter, %sampler_name, %steps, %cfg, %scheduler_name, %basemodelname, %denoise, %clip_skip)"}),
                "path":                  ("STRING",  {"default": '', "multiline": False,                           "tooltip": "path to save the images (under Comfy's save directory)"}),
                "extension":             (['png', 'jpeg', 'jpg', 'webp'], {                                        "tooltip": "file extension/type to save image as"}),
            },
            "optional": {
                # Metadata input (new pattern: MetadataCompiler → ImageSaver)
                "metadata":              ("METADATA", {"default": None,                                            "tooltip": "metadata from MetadataCompiler (if provided, individual param inputs are ignored)"}),
                # File saving options
                "lossless_webp":         ("BOOLEAN", {"default": True,                                             "tooltip": "if True, saved WEBP files will be lossless"}),
                "quality_jpeg_or_webp":  ("INT",     {"default": 100, "min": 1, "max": 100,                        "tooltip": "quality setting of JPEG/WEBP"}),
                "optimize_png":          ("BOOLEAN", {"default": False,                                            "tooltip": "if True, saved PNG files will be optimized (can reduce file size but is slower)"}),
                "counter":               ("INT",     {"default": 0, "min": 0, "max": 0xffffffffffffffff,           "tooltip": "counter"}),
                "time_format":           ("STRING",  {"default": "%Y-%m-%d-%H%M%S", "multiline": False,            "tooltip": "timestamp format"}),
                "save_workflow_as_json": ("BOOLEAN", {"default": False,                                            "tooltip": "if True, also saves the workflow as a separate JSON file"}),
                "embed_workflow":        ("BOOLEAN", {"default": True,                                             "tooltip": "if True, embeds the workflow in the saved image files.\nStable for PNG, experimental for WEBP.\nJPEG experimental and only if metadata size is below 65535 bytes"}),
                "show_preview":          ("BOOLEAN", {"default": True,                                             "tooltip": "if True, displays saved images in the UI preview"}),
                # Backward-compat: individual generation params (used when metadata is not provided)
                "steps":                 ("INT",     {"default": 20, "min": 1, "max": 10000,                       "tooltip": "number of steps"}),
                "cfg":                   ("FLOAT",   {"default": 7.0, "min": 0.0, "max": 100.0,                    "tooltip": "CFG value"}),
                "modelname":             ("STRING",  {"default": '', "multiline": False,                           "tooltip": "model name (can be multiple, separated by commas)"}),
                "sampler_name":          ("STRING",  {"default": '', "multiline": False,                           "tooltip": "sampler name (as string)"}),
                "scheduler_name":        ("STRING",  {"default": 'normal', "multiline": False,                     "tooltip": "scheduler name (as string)"}),
                "positive":              ("STRING",  {"default": '', "multiline": True, "placeholder": "positive prompt",  "tooltip": "positive prompt"}),
                "negative":              ("STRING",  {"default": '', "multiline": True, "placeholder": "negative prompt", "tooltip": "negative prompt"}),
                "seed_value":            ("INT",     {"default": 0, "min": 0, "max": 0xffffffffffffffff,           "tooltip": "seed"}),
                "width":                 ("INT",     {"default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 8,  "tooltip": "image width"}),
                "height":                ("INT",     {"default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 8,  "tooltip": "image height"}),
                "denoise":               ("FLOAT",   {"default": 1.0, "min": 0.0, "max": 1.0,                      "tooltip": "denoise value"}),
                "clip_skip":             ("INT",     {"default": 0, "min": -24, "max": 24,                         "tooltip": "skip last CLIP layers (positive or negative value, 0 for no skip)"}),
                "additional_hashes":     ("STRING",  {"default": "", "multiline": False,                           "tooltip": "hashes separated by commas, optionally with names. 'Name:HASH' (e.g., 'MyLoRA:FF735FF83F98')"}),
                "download_civitai_data": ("BOOLEAN", {"default": True,                                             "tooltip": "Download and cache data from civitai.com to save correct metadata."}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("hashes", "gallery_metadata_json")
    OUTPUT_TOOLTIPS = ("Comma-separated list of the hashes to chain with other Image Saver additional_hashes", "Written parameters to the image metadata")
    FUNCTION = "save_files"

    OUTPUT_NODE = True

    CATEGORY = "ImageSaver"
    DESCRIPTION = "Save images with civitai-compatible generation metadata"

    def save_files(
        self,
        images: list[torch.Tensor],
        filename: str,
        path: str,
        extension: str,
        metadata: Metadata | None = None,
        lossless_webp: bool = True,
        quality_jpeg_or_webp: int = 100,
        optimize_png: bool = False,
        counter: int = 0,
        time_format: str = "%Y-%m-%d-%H%M%S",
        save_workflow_as_json: bool = False,
        embed_workflow: bool = True,
        show_preview: bool = True,
        # Backward-compat individual params (ignored when metadata is provided)
        steps: int = 20,
        cfg: float = 7.0,
        modelname: str = "",
        sampler_name: str = "",
        scheduler_name: str = "normal",
        positive: str = "",
        negative: str = "",
        seed_value: int = 0,
        width: int = 512,
        height: int = 512,
        denoise: float = 1.0,
        clip_skip: int = 0,
        additional_hashes: str = "",
        download_civitai_data: bool = True,
        prompt: dict[str, Any] | None = None,
        extra_pnginfo: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        # Use provided metadata or compile from individual params
        if metadata is None:
            metadata = MetadataCompiler.make_metadata(
                modelname, positive, negative, width, height, seed_value, steps, cfg,
                sampler_name, scheduler_name, denoise, clip_skip,
                None, additional_hashes, download_civitai_data
            )

        path = make_pathname(path, metadata.width, metadata.height, metadata.seed, metadata.model_name, counter, time_format, metadata.sampler_name, metadata.steps, metadata.cfg, metadata.scheduler_name, metadata.denoise, metadata.clip_skip, '')

        filenames = _save_images(images, filename, extension, path, quality_jpeg_or_webp, lossless_webp, optimize_png, prompt, extra_pnginfo, save_workflow_as_json, embed_workflow, counter, time_format, metadata)

        subfolder = os.path.normpath(path)

        result: dict[str, Any] = {
            "result": (metadata.final_hashes, json.dumps(metadata.gallery_metadata)),
        }

        if show_preview:
            result["ui"] = {"images": [{"filename": filename, "subfolder": subfolder if subfolder != '.' else '', "type": 'output'} for filename in filenames]}

        return result


class ImageSaverSimple:
    """Backward-compatible wrapper. Accepts pre-computed METADATA and saves images."""

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "images":                ("IMAGE",    {                                                             "tooltip": "image(s) to save"}),
                "filename":              ("STRING",   {"default": '%time_%basemodelname_%seed', "multiline": False, "tooltip": "filename (available variables: %date, %time, %time_format<format>, %model, %width, %height, %seed, %counter, %sampler_name, %steps, %cfg, %scheduler_name, %basemodelname, %denoise, %clip_skip)"}),
                "path":                  ("STRING",   {"default": '', "multiline": False,                           "tooltip": "path to save the images (under Comfy's save directory)"}),
                "extension":             (['png', 'jpeg', 'jpg', 'webp'], {                                         "tooltip": "file extension/type to save image as"}),
                "lossless_webp":         ("BOOLEAN",  {"default": True,                                             "tooltip": "if True, saved WEBP files will be lossless"}),
                "quality_jpeg_or_webp":  ("INT",      {"default": 100, "min": 1, "max": 100,                        "tooltip": "quality setting of JPEG/WEBP"}),
                "optimize_png":          ("BOOLEAN",  {"default": False,                                            "tooltip": "if True, saved PNG files will be optimized (can reduce file size but is slower)"}),
                "embed_workflow":        ("BOOLEAN",  {"default": True,                                             "tooltip": "if True, embeds the workflow in the saved image files."}),
                "save_workflow_as_json": ("BOOLEAN",  {"default": False,                                            "tooltip": "if True, also saves the workflow as a separate JSON file"}),
            },
            "optional": {
                "metadata":              ("METADATA", {"default": None,                                             "tooltip": "metadata to embed in the image"}),
                "counter":               ("INT",      {"default": 0, "min": 0, "max": 0xffffffffffffffff,           "tooltip": "counter"}),
                "time_format":           ("STRING",   {"default": "%Y-%m-%d-%H%M%S", "multiline": False,            "tooltip": "timestamp format"}),
                "show_preview":          ("BOOLEAN",  {"default": True,                                             "tooltip": "if True, displays saved images in the UI preview"}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("hashes", "gallery_metadata_json")
    OUTPUT_TOOLTIPS = ("Comma-separated list of the hashes to chain with other Image Saver additional_hashes", "Written parameters to the image metadata")
    FUNCTION = "save_images"

    OUTPUT_NODE = True

    CATEGORY = "ImageSaver"
    DESCRIPTION = "Save images with civitai-compatible generation metadata"

    def save_images(self,
        images: list[torch.Tensor],
        filename: str,
        path: str,
        extension: str,
        lossless_webp: bool,
        quality_jpeg_or_webp: int,
        optimize_png: bool,
        embed_workflow: bool = True,
        save_workflow_as_json: bool = False,
        show_preview: bool = True,
        metadata: Metadata | None = None,
        counter: int = 0,
        time_format: str = "%Y-%m-%d-%H%M%S",
        prompt: dict[str, Any] | None = None,
        extra_pnginfo: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if metadata is None:
            metadata = Metadata('', '', '', 512, 512, 0, 20, 7.0, '', 'normal', 1.0, 0, '', '', '', '', '')

        path = make_pathname(path, metadata.width, metadata.height, metadata.seed, metadata.model_name, counter, time_format, metadata.sampler_name, metadata.steps, metadata.cfg, metadata.scheduler_name, metadata.denoise, metadata.clip_skip, '')

        filenames = _save_images(images, filename, extension, path, quality_jpeg_or_webp, lossless_webp, optimize_png, prompt, extra_pnginfo, save_workflow_as_json, embed_workflow, counter, time_format, metadata)

        subfolder = os.path.normpath(path)

        result: dict[str, Any] = {
            "result": (metadata.final_hashes, json.dumps(metadata.gallery_metadata)),
        }

        if show_preview:
            result["ui"] = {"images": [{"filename": filename, "subfolder": subfolder if subfolder != '.' else '', "type": 'output'} for filename in filenames]}

        return result


# --- Static helpers ---

def _save_images(
    images: list[torch.Tensor],
    filename_pattern: str,
    extension: str,
    path: str,
    quality_jpeg_or_webp: int,
    lossless_webp: bool,
    optimize_png: bool,
    prompt: dict[str, Any] | None,
    extra_pnginfo: dict[str, Any] | None,
    save_workflow_as_json: bool,
    embed_workflow: bool,
    counter: int,
    time_format: str,
    metadata: Metadata
) -> list[str]:
    filename_prefix = make_filename(filename_pattern, metadata.width, metadata.height, metadata.seed, metadata.model_name, counter, time_format, metadata.sampler_name, metadata.steps, metadata.cfg, metadata.scheduler_name, metadata.denoise, metadata.clip_skip, '')

    output_path = os.path.join(folder_paths.output_directory, path)

    if output_path.strip() != '':
        if not os.path.exists(output_path.strip()):
            print(f'The path `{output_path.strip()}` specified doesn\'t exist! Creating directory.')
            os.makedirs(output_path, exist_ok=True)

    result_paths: list[str] = list()
    num_images = len(images)
    base_suffix = _get_base_suffix(output_path, filename_prefix, extension, num_images)
    for idx, image in enumerate(images):
        i = 255. * image.cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

        current_filename_prefix = _format_batch_filename(filename_prefix, base_suffix, idx)
        final_filename = f"{current_filename_prefix}.{extension}"
        filepath = os.path.join(output_path, final_filename)

        save_image(img, filepath, extension, quality_jpeg_or_webp, lossless_webp, optimize_png, metadata.gallery_metadata, prompt, extra_pnginfo, embed_workflow)

        if save_workflow_as_json:
            save_json(extra_pnginfo, os.path.join(output_path, current_filename_prefix))

        result_paths.append(final_filename)
    return result_paths


def _get_base_suffix(output_path: str, filename_prefix: str, extension: str, batch_size: int) -> int | None:
    """Calculate the starting suffix for batch naming. Returns None for a single new image."""
    existing_files = [f for f in os.listdir(output_path) if f.startswith(filename_prefix) and f.endswith(extension)]

    if batch_size == 1 and not existing_files:
        return None

    suffixes: list[int] = []
    suffix_prefix = f"{filename_prefix}_"
    for f in existing_files:
        name, _ = os.path.splitext(f)
        suffix = name.removeprefix(suffix_prefix)
        if name != suffix and suffix.isdigit():
            suffixes.append(int(suffix))

    if suffixes:
        return max(suffixes) + 1
    else:
        return 1


def _format_batch_filename(filename_prefix: str, base_suffix: int | None, batch_index: int) -> str:
    """Format a batch filename. Returns the plain prefix when base_suffix is None."""
    if base_suffix is None:
        return filename_prefix
    return f"{filename_prefix}_{base_suffix + batch_index:02d}"
