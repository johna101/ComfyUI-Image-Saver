from typing import Any

from .nodes.metadata import MetadataCompiler
from .nodes.lora_collector import LoraCollector
from .nodes.saver import ImageSaver, ImageSaverSimple
from .nodes.loaders import CheckpointLoaderWithName, UNETLoaderWithName
from .nodes.selectors import SamplerSelector, SchedulerSelector, SchedulerSelectorInspire, SchedulerSelectorEfficiency, InputParameters
from .nodes.introspection import AnyToString, WorkflowInputValue
from .nodes.literals import SeedGenerator, StringLiteral, SizeLiteral, IntLiteral, FloatLiteral, CfgLiteral
from .nodes.deprecated import ConditioningConcatOptional, RandomShapeGenerator, CivitaiHashFetcher, RandomTagPicker

# Display names must match exactly — ComfyUI workflows reference these strings
NODE_CLASS_MAPPINGS: dict[str, Any] = {
    "Checkpoint Loader with Name (Image Saver)": CheckpointLoaderWithName,
    "UNet loader with Name (Image Saver)": UNETLoaderWithName,
    "Image Saver": ImageSaver,
    "Image Saver Simple": ImageSaverSimple,
    "Image Saver Metadata": MetadataCompiler,
    "Lora Collector (Image Saver)": LoraCollector,
    "Sampler Selector (Image Saver)": SamplerSelector,
    "Scheduler Selector (Image Saver)": SchedulerSelector,
    "Scheduler Selector (inspire) (Image Saver)": SchedulerSelectorInspire,
    "Scheduler Selector (Eff.) (Image Saver)": SchedulerSelectorEfficiency,
    "Input Parameters (Image Saver)": InputParameters,
    "Any to String (Image Saver)": AnyToString,
    "Workflow Input Value (Image Saver)": WorkflowInputValue,
    "Seed Generator (Image Saver)": SeedGenerator,
    "String Literal (Image Saver)": StringLiteral,
    "Width/Height Literal (Image Saver)": SizeLiteral,
    "Cfg Literal (Image Saver)": CfgLiteral,
    "Int Literal (Image Saver)": IntLiteral,
    "Float Literal (Image Saver)": FloatLiteral,
    "Conditioning Concat Optional (Image Saver)": ConditioningConcatOptional,
    "RandomShapeGenerator": RandomShapeGenerator,
    "Civitai Hash Fetcher (Image Saver)": CivitaiHashFetcher,
    "Random Tag Picker (Image Saver)": RandomTagPicker,
}

WEB_DIRECTORY = "js"

__all__ = ['NODE_CLASS_MAPPINGS', 'WEB_DIRECTORY']
