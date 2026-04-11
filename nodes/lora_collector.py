# ComfyUI-Image-Saver — LoRA Collector Node
# Reads structured LoRA data from Power Lora Loader (rgthree) by node ID,
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
                "lora_loader_node_id": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "The node ID of the Power Lora Loader (rgthree) to read from"
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
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ("LORA_INFO",)
    RETURN_NAMES = ("lora_info",)
    OUTPUT_TOOLTIPS = ("Structured LoRA information array for MetadataCompiler",)
    FUNCTION = "collect"
    CATEGORY = "ImageSaver"
    DESCRIPTION = "Reads LoRA data from Power Lora Loader (rgthree) and builds structured metadata"

    def collect(
        self,
        lora_loader_node_id: str,
        include_disabled: bool = True,
        download_civitai_data: bool = True,
        prompt: dict[str, Any] | None = None,
        extra_pnginfo: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]],]:
        if not prompt or not lora_loader_node_id:
            return ([],)

        # Read the Power Lora Loader node from the execution prompt
        node = prompt.get(lora_loader_node_id)
        if node is None:
            print(f"LoraCollector: Node {lora_loader_node_id} not found in prompt")
            return ([],)

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

        return (lora_list,)


def _clean_lora_name(lora_path: str) -> str:
    """Extract clean display name from a LoRA path like 'qwen/jib_qwen_nudity_fix.safetensors'."""
    import os
    basename = os.path.basename(lora_path)
    name, ext = os.path.splitext(basename)
    # Only strip known model extensions
    if ext.lower() in {'.safetensors', '.ckpt', '.pt', '.pth', '.bin'}:
        return name
    return basename
