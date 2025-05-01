import os
import json
import logging
from nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class TiledConfigNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "output": (["landscape", "portrait", "insta", "square"],),
                "layout": (["SATB", "SMATBB"],)
            }
        }

    RETURN_TYPES = ("INT", "INT", "STRING", "STRING",)
    RETURN_NAMES = ("width", "height", "tiling", "output_name")
    FUNCTION = "process"
    CATEGORY = "vk-nodes"
    OUTPUT_NODE = False


    def process(self, output, layout):

        if output == "landscape":
            width = 1920
            height = 1080
        elif output == "portrait":
            width = 1080
            height = 1920
        elif output == "insta":
            width = 1080
            height = 1350
        elif output == "square":
            width = 1080
            height = 1080
        else:
            width = 512
            height = 512

        filename = output + ".mp4"

        if layout == "SATB":
            tiling = "[[1,3],[4,6]]"
        elif layout == "SMATBB":
            tiling = {
                "landscape": "[[1,2,3],[4,5,6]]",
                "portrait": "[[1,3],[2,5],[4,6]]",
                "insta": "[[1,3],[2,5],[4,6]]",
                "square": "[[1,2,3],[4,5,6]]",
            }.get(output, "[[1]]") 


        return width, height, tiling, filename    



class TiledSetupNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "start_width": ("INT", {"default": 960, "min": 1}),
                "start_height": ("INT", {"default": 960, "min": 1}),
                "output_width": ("INT", {"default": 1920, "min": 1}),
                "output_height": ("INT", {"default": 1080, "min": 1}),
                "tiling_strategy": ("STRING", {"default": "[[1,2,3],[4,5,6]]"}),
                "max_render_cells": ("INT", {"default": 64, "min": 1}),
                "max_tile_size": ("INT", {"default": 960, "min": 1}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("render_config", "log", "tile_str")
    FUNCTION = "process"
    CATEGORY = "vk-nodes"
    OUTPUT_NODE = False


    def process(self, **kwargs):
        log = ""

        output = {}
        output.update(kwargs)
        output["tiling_strategy"] = json.loads(output["tiling_strategy"])

        # number of rows and cols
        output["rows"] = len(output["tiling_strategy"])
        output["cols"] = len(output["tiling_strategy"][0])
        for el in output["tiling_strategy"]:
            if len(el) == output["cols"]:
                log = "ERROR: Only exact tiling is possible now"

        
        width = output["output_width"]
        height = output["output_height"]

        scale = 4.0
        output["render_cells"] = output["max_render_cells"] + 1
        while output["render_cells"] > output["max_render_cells"] and scale > 0.1:
            output["tile_width"] = int(width / output["cols"] * scale)
            output["tile_height"] = int(height / output["rows"]* scale)
            output["render_cells_x"] = int(output["tile_width"] / 64.0)
            output["render_cells_y"] = int(output["tile_height"] / 64.0)
            output["render_scale"] = scale
            output["render_cells"] = output["render_cells_x"] * output["render_cells_y"]
            scale = scale - 0.01
        
        output["tile_width"] = output["render_cells_x"] * 64
        output["tile_height"] = output["render_cells_y"] * 64
        output["tile_str"] = f'{output["render_cells_x"]}x{output["render_cells_y"]}'

        log = f"""Optimal render format:
  * {output["render_cells_x"]}x{output["render_cells_y"]} tiles with {output["tile_width"]}x{output["tile_height"]} pixels
  * total tiles: {output["render_cells"]}
  * scale factor: approx {output["render_scale"]}
        """
        
        output["input_width"] = output["tile_width"]*4
        output["input_height"] = output["tile_height"]*4

        log += "\nConfig:\n"  + json.dumps(output, indent=2)

        return json.dumps(output), log, output["tile_str"]


class TiledCropNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "render_config": ("STRING", {"default": "{}"}),  # JSON config input
                "start_time": ("FLOAT", {"default": 0.0}),  # JSON config input
            }
        }

    RETURN_TYPES = ("INT", "INT", "INT", "INT", "INT", "STRING")
    RETURN_NAMES = ("crop_width", "crop_height", "crop_x", "crop_y", "min_resolution", "tile_string")
    FUNCTION = "process"
    CATEGORY = "vk-nodes"
    OUTPUT_NODE = False

    def process(self, render_config, start_time):
        # Parse JSON config
        try:
            cfg = json.loads(render_config)
        except json.JSONDecodeError:
            logging.error("Invalid JSON in config")
            return 0, 0, 0, 0, 0, "8x8"  # Default return to prevent crashes

        logging.info("RECALC")

        # Extract values safely
        start_width = cfg.get("start_width", 1920)
        start_height = cfg.get("start_height", 1080)
        tile_width = cfg.get("tile_width", 960)
        tile_height = cfg.get("tile_height", 960)
        max_tile_size = cfg.get("max_tile_size", 960)

        start_ratio = start_width / start_height
        tile_ratio = tile_width / tile_height

        tile_str = cfg.get("tile_str", "8x8")

        # Decide which direction needs to be cropped
        if start_ratio > tile_ratio:
            # Original image is wider than needed, crop width
            crop_height = min(start_height, max_tile_size)
            crop_width = int(crop_height * tile_ratio)
            crop_x = (start_width - crop_width) // 2
            crop_y = 0
        else:
            # Original image is taller than needed, crop height
            crop_width = min(start_width, max_tile_size)
            crop_height = int(crop_width / tile_ratio)
            crop_x = (start_width - crop_width) // 2
            crop_y = 0

        min_resolution = min(tile_width, tile_height)

        logging.info(
            f"Crop: width={crop_width}, height={crop_height}, x={crop_x}, y={crop_y}, min_res={min_resolution}"
        )

        if start_time > 0.001:
            tile_str = f"{start_time}_{tile_str}"

        return crop_width, crop_height, crop_x, crop_y, min_resolution, tile_str




# Register nodes in ComfyUI
NODE_CLASS_MAPPINGS.update(
    {
        "TiledConfigNode": TiledConfigNode,
        "TiledSetupNode": TiledSetupNode,
        "TiledCropNode": TiledCropNode,
    }
)

NODE_DISPLAY_NAME_MAPPINGS.update(
    {
        "TiledConfigNode": "Tiled Config Node",
        "TiledSetupNode": "Tiled Setup Node",
        "TiledCropNode": "Tiled Crop Node",
    }
)
