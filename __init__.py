# vk-nodes ComfyUI __init__.py
from .vk_tiled_render import TiledRenderNode, PrepareJobs
from .vk_tiled_setup import TiledSetupNode, TiledCropNode, TiledConfigNode
from .vk_audio import LoadAudio
from .vk_headline import SketchyText, SketchyThumbnail

NODE_CLASS_MAPPINGS = {
    "TiledRenderNode": TiledRenderNode,
    "PrepareJobs": PrepareJobs,
    "TiledConfigNode": TiledConfigNode,
    "TiledSetupNode": TiledSetupNode,
    "TiledCropNode": TiledCropNode,
    "VKLoadAudio": LoadAudio,
    "SketchyText": SketchyText,
    "SketchyThumbnail": SketchyThumbnail,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TiledRenderNode": "Tiled Render",
    "PrepareJobs": "Prepare Jobs",
    "TiledConfigNode": "Tiled Config Node",
    "TiledSetupNode": "Tiled Setup Node",
    "TiledCropNode": "Tiled Crop Node",
    "VKLoadAudio": "VK Load Audio",
    "SketchyText": "Sketchy Text",
    "SketchyThumbnail": "Sketchy Thumbnail",
}


__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']


UE_VERSION = "1.0.6"
