# vk-nodes ComfyUI __init__.py
from .vk_tiled_render import TiledRenderNode, PrepareJobs
from .vk_tiled_setup import TiledSetupNode, TiledCropNode


from .vk_tiled_render import NODE_CLASS_MAPPINGS as NODE_CLASS_MAPPINGS
from .vk_tiled_render import NODE_DISPLAY_NAME_MAPPINGS as NODE_DISPLAY_NAME_MAPPINGS

from .vk_tiled_setup import NODE_CLASS_MAPPINGS as NODE_CLASS_MAPPINGS_A
from .vk_tiled_setup import NODE_DISPLAY_NAME_MAPPINGS as NODE_DISPLAY_NAME_MAPPINGS_A

NODE_CLASS_MAPPINGS.update(NODE_CLASS_MAPPINGS_A)
NODE_DISPLAY_NAME_MAPPINGS.update(NODE_DISPLAY_NAME_MAPPINGS_A)


__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']



UE_VERSION = "1.0.2"
