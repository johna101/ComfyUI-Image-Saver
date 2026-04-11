# ComfyUI-Image-Saver — LoRA Collector Node
# Reads structured LoRA data from Power Lora Loader (rgthree) via MODEL wire,
# resolves hashes and CivitAI metadata, outputs a clean LORA_INFO array.

import re
from typing import Any

from ..services.hashing import get_sha256
from ..services.file_utils import full_lora_path_for
from ..services.civitai import get_civitai_info


class LoraCollector:
    """Collects LoRA info from Power Lora Loader (rgthree) and outputs structured metadata."""

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "model": ("MODEL", {
                    "tooltip": "Connect the MODEL output from Power Lora Loader here"
                }),
            },
            "optional": {
                "include_disabled": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Include disabled LoRAs in the output (marked as enabled: false)"
                }),
                "download_civitai_data": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Download and cache CivitAI metadata for each LoRA"
                }),
            },
            "hidden": {
                "prompt": "PROMPT",
            },
        }

    RETURN_TYPES = ("LORA_INFO", "MODEL")
    RETURN_NAMES = ("lora_info", "model")
    OUTPUT_TOOLTIPS = ("Structured LoRA information array for MetadataCompiler", "Model passthrough")
    FUNCTION = "collect"
    CATEGORY = "ImageSaver"
    DESCRIPTION = "Reads LoRA data from Power Lora Loader (rgthree) and builds structured metadata"

    def collect(
        self,
        model,
        include_disabled: bool = True,
        download_civitai_data: bool = True,
        prompt: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], Any]:
        if not prompt:
            return ([], model)

        # Find our own node in the prompt to trace where 'model' came from
        source_node_id = _find_source_node_id(prompt, 'model')
        if source_node_id is None:
            print("LoraCollector: Could not determine source node for model input")
            return ([], model)

        # Read the source node (Power Lora Loader)
        node = prompt.get(source_node_id)
        if node is None:
            print(f"LoraCollector: Source node {source_node_id} not found in prompt")
            return ([], model)

        inputs = node.get("inputs", {})
        lora_list = []

        # Power Lora Loader uses lora_1, lora_2, ... keys
        # Each is a dict: {"on": bool, "lora": "path/name.safetensors", "strength": float}
        for key, value in inputs.items():
            if not re.match(r'^lora_\d+$', key):
                continue

            if not isinstance(value, dict):
                continue

            enabled = value.get("on", True)
            lora_path_name = value.get("lora", "")
            strength = value.get("strength", 1.0)

            if not lora_path_name:
                continue

            if not enabled and not include_disabled:
                continue

            # Build the lora info dict
            lora_info: dict[str, Any] = {
                "name": _clean_lora_name(lora_path_name),
                "path": lora_path_name,
                "weight": strength,
                "enabled": enabled,
            }

            # Resolve full path and compute hash
            full_path = full_lora_path_for(lora_path_name)
            if full_path:
                lora_hash = get_sha256(full_path)[:10]
                lora_info["hash"] = lora_hash

                # CivitAI lookup
                if download_civitai_data:
                    civitai_info = get_civitai_info(full_path, lora_hash)
                    if civitai_info:
                        civitai_data: dict[str, Any] = {
                            "modelName": civitai_info.get("model", {}).get("name", ""),
                            "versionName": civitai_info.get("name", ""),
                        }
                        if "air" in civitai_info:
                            civitai_data["air"] = civitai_info["air"]
                        elif "id" in civitai_info:
                            civitai_data["modelVersionId"] = civitai_info["id"]
                        lora_info["civitai"] = civitai_data

            lora_list.append(lora_info)

        return (lora_list, model)


def _find_source_node_id(prompt: dict, input_name: str) -> str | None:
    """
    Find which node's output is wired to our input by scanning the prompt.
    In the execution prompt, linked inputs are stored as [source_node_id, output_index].
    We find our own node (LoraCollector) and read the link for the given input.
    """
    for node_id, node in prompt.items():
        if node.get("class_type") == "Lora Collector (Image Saver)":
            inputs = node.get("inputs", {})
            link = inputs.get(input_name)
            if isinstance(link, list) and len(link) >= 1:
                return str(link[0])
    return None


def _clean_lora_name(lora_path: str) -> str:
    """Extract clean display name from a LoRA path like 'qwen/jib_qwen_nudity_fix.safetensors'."""
    import os
    basename = os.path.basename(lora_path)
    name, ext = os.path.splitext(basename)
    if ext.lower() in {'.safetensors', '.ckpt', '.pt', '.pth', '.bin'}:
        return name
    return basename
