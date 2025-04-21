import os
import json
import subprocess
import logging
import glob
import re
from PIL import Image
from nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

MY_OUTPUT_FOLDER = "./ComfyUI/output"

class TiledRenderNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "render_config": ("STRING", {"default": "{}"}),
                # "input_width": ("INT", {"default": 1920, "min": 1}),
                # "input_height": ("INT", {"default": 1080, "min": 1}),
                # "output_width": ("INT", {"default": 1920, "min": 1}),
                # "output_height": ("INT", {"default": 1080, "min": 1}),
                "padding": ("INT", {"default": 10, "min": 0}),
                "duration": ("FLOAT", {"default": 1, "min": 0.2}),
                # "tiling_strategy": ("STRING", {"default": "[[1,2,3],[4,5,6]]"}),
                "output_path": ("STRING", {"default": ""}),
                "output_name": ("STRING", {"default": "tiled_video.mp4"})
            },
            "optional": {
                "Sophia": ("INT", {"default": -1}),
                "Melina": ("INT", {"default": -1}),
                "Ada":    ("INT", {"default": -1}),
                "Taro":   ("INT", {"default": -1}),
                "Bari":   ("INT", {"default": -1}),
                "Barney": ("INT", {"default": -1})
            }            
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "process"
    CATEGORY = "vk-nodes"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")    

    def get_large_videos_for(self, output_path, name, index):
        all_vids = glob.glob(os.path.join(MY_OUTPUT_FOLDER, output_path, f"video_scaled/{self.tile_str}/*{name}*mp4"))
        no_sound = [el for el in all_vids if "audio" not in el]
        if len(no_sound) == 0:
            return None
        no_sound.sort()
        return no_sound[index]


    def get_overlay_images(self, output_path):
        all_overlays = glob.glob(os.path.join(MY_OUTPUT_FOLDER, output_path, "overlays/overlay*.png"))
        
        overlay_data = []
        pattern = re.compile(r"start=([\d\.]+)\s*end=([\d\.]+)")

        for overlay in all_overlays:
            match = pattern.search(overlay.replace(".png", ""))
            if match:
                start_time = float(match.group(1))
                end_time = float(match.group(2))
                overlay_data.append({"file": overlay, "start": start_time, "end": end_time})

        return overlay_data

    def get_image_size(self, image_path):
        with Image.open(image_path) as img:
            return img.size        


    def get_mix_file(self, output_path):
        mix_file = glob.glob(os.path.join(MY_OUTPUT_FOLDER, output_path, f"audio/mix.mp3"))
        if len(mix_file) == 0:
            return None
        return mix_file[0]
    
    def get_logo_image(self, output_path):
        logo = glob.glob(os.path.join(MY_OUTPUT_FOLDER, output_path, "overlays/logo.png"))
        if len(logo) == 0:
            return None
        return logo[0]

    def process(self, render_config, padding, duration, output_path, output_name,
                Sophia=-1, Melina=-1, Ada=-1, Taro=-1, Bari=-1, Barney=-1):
        logging.info("Starting Tiled Render process")

        config = json.loads(render_config)
        input_width = config["input_width"]
        input_height = config["input_height"]
        output_width = config["output_width"]
        output_height = config["output_height"]
        tiling_strategy = config["tiling_strategy"]
        self.tile_str = config["tile_str"]

        all_videos = [
            self.get_large_videos_for(output_path, "Sophia", Sophia),
            self.get_large_videos_for(output_path, "Melina", Melina),
            self.get_large_videos_for(output_path, "Ada", Ada),
            self.get_large_videos_for(output_path, "Taro", Taro),
            self.get_large_videos_for(output_path, "Bari", Bari),
            self.get_large_videos_for(output_path, "Barney", Barney)
        ]
        video_files = self.extract_videos_from_strategy(tiling_strategy, all_videos)
        logging.info(f"Ordered video files: {video_files}")


        # overlay files
        overlay_config = self.get_overlay_images(output_path)
        logging.info(f"overlays: {overlay_config}")       

        # logo file
        logo_file = self.get_logo_image(output_path)
        logging.info(f"logo: {logo_file}")

        # audio mix file
        mix_file = self.get_mix_file(output_path)
        logging.info(f"mix file: {mix_file}")

        filter_complex, inputs = self.generate_ffmpeg_filter(video_files, tiling_strategy, input_width, input_height, output_width, output_height, padding, overlay_config, logo_file)
        if not filter_complex:
            logging.error("Could not generate filter_complex")
            return ("Error: Could not generate filter_complex",)

        # add the mix file to the inputs if available
        audio_map = ["-map", "0:a?"]
        if mix_file:
            inputs.extend(["-i", mix_file])
            audio_map = ["-map", f"{len(inputs)//2-1}:a"]

        command = [
            "ffmpeg", "-y"
        ] + inputs + [
            "-filter_complex", filter_complex,
            "-map", f"[{self.final_output_label}]"
        ] + audio_map + [ 
            "-t", f"{duration}",
            "-preset", "slow", "-crf", "18",
            os.path.join(MY_OUTPUT_FOLDER, output_path, output_name)
        ]
        
        logging.info(f"Executing FFmpeg command: {' '.join(command)}")
        try:
            subprocess.run(command, check=True)
            logging.info("FFmpeg processing completed successfully")
            return (output_path,)
        except subprocess.CalledProcessError as e:
            logging.error(f"FFmpeg error: {e}")
            return (f"FFmpeg error: {str(e)}",)
    
    def extract_videos_from_strategy(self, tiling_strategy, all_videos):
        ordered_videos = []
        for row in tiling_strategy:
            for idx in row:
                if 1 <= idx <= len(all_videos):
                    ordered_videos.append(all_videos[idx - 1])
                else:
                    ordered_videos.append(None)
        return ordered_videos

    def generate_ffmpeg_filter(self, video_files, tiling_strategy, in_w, in_h, out_w, out_h, pad, overlay_config, logo_file):
        inputs = []
        filter_parts = []
        
        for i, file in enumerate(video_files):
            if isinstance(file, str):
                inputs.extend(["-i", file])
        
        base_filter = f"color=c=black:s={out_w}x{out_h} [base0]"
        overlay_chain = ""
        
        y_offset = 0
        index = 0
        prev_label = "base0"
        
        for row_idx, row in enumerate(tiling_strategy):
            x_offset = 0
            row_height = out_h // len(tiling_strategy)
            
            for col_idx, col in enumerate(row):
                col_width = out_w // len(row)
                crop_w, crop_h = in_w, in_h
                crop_x, crop_y = 0, 0
                aspect_ratio_in = in_w / in_h
                aspect_ratio_out = col_width / row_height
                
                if aspect_ratio_in > aspect_ratio_out:
                    crop_w = int(in_h * aspect_ratio_out)
                    crop_x = (in_w - crop_w) // 2
                else:
                    crop_h = int(in_w / aspect_ratio_out)
                    crop_y = 0
                
                filter_parts.append(f"[{index}:v]crop={crop_w}:{crop_h}:{crop_x}:{crop_y},scale={col_width}:{row_height}[v{index}]")
                overlay_label = f"base{index + 1}"
                overlay_chain += f"[{prev_label}][v{index}] overlay={x_offset}:{y_offset} [{overlay_label}]; "
                prev_label = overlay_label
                
                x_offset += col_width + pad
                index += 1
            y_offset += row_height + pad

        # Overlays with fade-in/out
        for i, overlay in enumerate(overlay_config):
            inputs.extend(["-i", overlay["file"]])
            overlay_w, overlay_h = self.get_image_size(overlay["file"])
            aspect_ratio = overlay_w / overlay_h

            max_w = out_w - 200
            max_h = out_h - 200

            if overlay_w / max_w > overlay_h / max_h:
                scale_w = max_w
                scale_h = int(max_w / aspect_ratio)
            else:
                scale_h = max_h
                scale_w = int(max_h * aspect_ratio)

            filter_parts.append(f"[{index}:v]scale={scale_w}:{scale_h}[overlay{i}]")
            overlay_chain += f"[{prev_label}][overlay{i}] overlay=(W-w)/2:(H-h)/2:format=auto:enable='between(t,{overlay['start']},{overlay['end']})' [{prev_label}_overlay{i}]; "
            prev_label = f"{prev_label}_overlay{i}"
            index += 1

        # Logo with bottom-right placement
        if logo_file:
            inputs.extend(["-i", logo_file])
            logo_w, logo_h = self.get_image_size(logo_file)
            filter_parts.append(f"[{index}:v]scale={logo_w}:{logo_h}[logo]")
            overlay_chain += f"[{prev_label}][logo] overlay={out_w - logo_w}:{out_h-logo_h} [{prev_label}_logo]; "
            prev_label = f"{prev_label}_logo"
        
        self.final_output_label = prev_label  # Ensure last label is used as final output
        filter_complex = f"{base_filter}; " + ", ".join(filter_parts) + "; " + overlay_chain.rstrip('; ')
        logging.info(f"Generated FFmpeg filter_complex: {filter_complex}")
        return filter_complex, inputs

NODE_CLASS_MAPPINGS["TiledRenderNode"] = TiledRenderNode
NODE_DISPLAY_NAME_MAPPINGS["TiledRenderNode"] = "Tiled Render"
