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
                "padding": ("INT", {"default": 10, "min": 0}),
                "duration": ("FLOAT", {"default": 1, "min": 0.2}),
                "start_time": ("FLOAT", {"default": 0, "min": 0.0}),
                "output_path": ("STRING", {"default": ""}),
                "output_name": ("STRING", {"default": "tiled_video.mp4"}),
                "jingle": ("BOOLEAN", {"default": True}),
                "jingle_duration": ("FLOAT", {"default": 3.9, "min": 0.0}),
                "transition_duration": ("FLOAT", {"default": 0.5, "min": 0}),
                "cut_start": ("FLOAT", {"default": 0, "min": 0.0}),
                "cut_end": ("FLOAT", {"default": 0, "min": 0.0}),
                "end_duration": ("FLOAT", {"default": 0, "min": 0.0}),
                "overlays": ("BOOLEAN", {"default": True}),
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


    def process_internal_overlay(self, image_path):
        from PIL import Image, ImageOps, ImageDraw, ImageFilter
        img = Image.open(image_path).convert("RGBA")

        # 1. Remove white borders
        bbox = ImageOps.invert(img.convert("L")).getbbox()
        img = img.crop(bbox)

        # 2. Add 20px white padding
        padding = 40
        padded_size = (img.width + 2 * padding, img.height + 2 * padding)
        padded_img = Image.new("RGBA", padded_size, (255, 255, 255, 255))
        padded_img.paste(img, (padding, padding), img)

        # 3. Create proper rounded rectangle mask
        corner_radius = 40
        scale = 4
        hr_size = (padded_img.width * scale, padded_img.height * scale)
        hr_mask = Image.new("L", hr_size, 0)
        hr_draw = ImageDraw.Draw(hr_mask)

        hr_draw.rounded_rectangle(
            [0, 0, hr_size[0], hr_size[1]],
            radius=corner_radius * scale,
            fill=255
        )

        # Downscale with anti-aliasing
        mask = hr_mask.resize(padded_img.size, resample=Image.LANCZOS)
        padded_img.putalpha(mask)

        # 4. Extend canvas before drop shadow
        extend = 10
        extended_size = (padded_img.width + 2 * extend, padded_img.height + 2 * extend)
        extended_img = Image.new("RGBA", extended_size, (0, 0, 0, 0))
        extended_img.paste(padded_img, (extend, extend), padded_img)
        extended_mask = Image.new("L", extended_size, 0)
        extended_mask.paste(mask, (extend, extend))

        # 5. Create and apply drop shadow
        shadow_offset = 5
        shadow_spread = 5
        shadow = Image.new("RGBA", extended_img.size, (0, 0, 0, 100))
        shadow_blur_mask = extended_mask.filter(ImageFilter.GaussianBlur(radius=shadow_spread))
        shadow.putalpha(shadow_blur_mask)

        shadow_canvas = Image.new("RGBA", extended_img.size, (0, 0, 0, 0))
        shadow_canvas.paste(shadow, (shadow_offset, shadow_offset), shadow)
        base_img = Image.alpha_composite(shadow_canvas, extended_img)

        # 6. Apply 80% global transparency
        alpha = base_img.getchannel("A").point(lambda p: int(p * 0.8))
        base_img.putalpha(alpha)

        # 7. Save
        internal_path = os.path.join(os.path.dirname(image_path), "internal_" + os.path.basename(image_path))
        base_img.save(internal_path, "PNG")

        return internal_path


    def get_overlay_images(self, output_path):
        all_overlays = glob.glob(os.path.join(MY_OUTPUT_FOLDER, output_path, "overlays/overlay*.png"))
        
        overlay_data = []
        pattern = re.compile(r"start=([\d\.]+)\s*end=([\d\.]+)")

        for overlay in all_overlays:
            match = pattern.search(overlay.replace(".png", ""))
            if match:
                start_time = float(match.group(1))
                end_time = float(match.group(2))
                internal_path = self.process_internal_overlay(overlay)
                overlay_data.append({"file": internal_path, "start": start_time, "end": end_time})

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

    def get_jingle_file(self, output_path, output_name):
        combined = os.path.join(MY_OUTPUT_FOLDER, output_path, "../Jingle",  output_name)
        logo = glob.glob(combined)
        if len(logo) == 0:
            return None
        return logo[0]        


    def process(self, render_config, padding, duration, start_time, output_path, output_name,
                jingle=True, jingle_duration=3.9, transition_duration=0.5, cut_start=0, cut_end=0, end_duration=2.0, overlays=True,
                Sophia=-1, Melina=-1, Ada=-1, Taro=-1, Bari=-1, Barney=-1):
        logging.info("Starting Tiled Render process")

        config = json.loads(render_config)
        input_width = config["input_width"]
        input_height = config["input_height"]
        output_width = config["output_width"]
        output_height = config["output_height"]
        tiling_strategy = config["tiling_strategy"]
        self.tile_str = config["tile_str"]

        if start_time > 0.001:
            start_time_str = json.dumps({"start_time": start_time})
            start_time = json.loads(start_time_str)["start_time"]
            self.tile_str = f"{start_time}_{self.tile_str}"
            print(f"start time: {start_time}")
            print(f"tile str: {self.tile_str}")

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
        if overlays:
            overlay_config = self.get_overlay_images(output_path)
            logging.info(f"overlays: {overlay_config}")       
        else:
            overlay_config = []

        # logo file
        logo_file = self.get_logo_image(output_path)
        logging.info(f"logo: {logo_file}")

        # audio mix file
        mix_file = self.get_mix_file(output_path)
        logging.info(f"mix file: {mix_file}")

        # jingle file
        jingle_file = self.get_jingle_file(output_path, output_name)
        logging.info(f"jingle file: {jingle_file}")

        if jingle_file is None:
            jingle = False
            dummy_name = output_name
        else:
            dummy_name = "dummy.mp4"

        filter_complex, inputs = self.generate_ffmpeg_filter(video_files, tiling_strategy, input_width, input_height, output_width, output_height, padding, overlay_config, logo_file)
        if not filter_complex:
            logging.error("Could not generate filter_complex")
            return ("Error: Could not generate filter_complex",)

        # add the mix file to the inputs if available
        audio_map = ["-map", "0:a?"]
        if mix_file:
            inputs.extend(["-i", mix_file])
            if start_time > 0.001:
                filter_complex += f";[{len(inputs)//2-1}:a]atrim=start={start_time},asetpts=PTS-STARTPTS[a]"
                audio_map = ["-map", f"[a]"]
            else:
                audio_map = ["-map", f"{len(inputs)//2-1}:a"]


        command = [
            "ffmpeg", "-y"
        ] + inputs + [
            "-filter_complex", filter_complex,
            "-map", f"[{self.final_output_label}]"
        ] + audio_map + [ 
            "-t", f"{duration}",
            "-preset", "slow", "-crf", "18",
            os.path.join(MY_OUTPUT_FOLDER, output_path, dummy_name)
        ]
        
        logging.info(f"Executing FFmpeg command: {' '.join(command)}")
        try:
            subprocess.run(command, check=True)
            logging.info("FFmpeg processing completed successfully")

            if not jingle:
                return (output_path,)            
            
        except subprocess.CalledProcessError as e:
            logging.error(f"FFmpeg error: {e}")
            return (f"FFmpeg error: {str(e)}",)


        transform_command = self.build_ffmpeg_transition(
            jingle_path=os.path.join(MY_OUTPUT_FOLDER, output_path, "../Jingle", output_name),
            dummy_path=os.path.join(MY_OUTPUT_FOLDER, output_path, dummy_name),
            output_path=os.path.join(MY_OUTPUT_FOLDER, output_path),
            output_name=output_name,
            duration=duration-cut_end,
            start_time=cut_start,
            transition_duration=transition_duration,
            still_duration=end_duration,
            jingle_time=jingle_duration,
            headline_path=os.path.join(MY_OUTPUT_FOLDER, output_path, "overlays", "headline.png"),
            bandname_path=os.path.join(MY_OUTPUT_FOLDER, output_path, "overlays", "bandname.png")
        )


        print("transform command")
        print(" ".join(transform_command))

        subprocess.run(transform_command, check=True)

        # remove the dummy file
        #os.remove(os.path.join(MY_OUTPUT_FOLDER, output_path, dummy_name))


        return (output_path,)
    
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

    def build_ffmpeg_transition(
        self,
        jingle_path,
        dummy_path,
        output_path,
        output_name,
        jingle_time=3.9,
        transition_duration=0.5,
        duration=16,
        start_time=0.0,
        still_duration=2,
        headline_path="",
        bandname_path="",
    ):

        start_transition = jingle_time - transition_duration
        end_start_transition = duration + start_transition - start_time
        

        # get the headline and bandname size
        headline_w, headline_h = self.get_image_size(headline_path)
        bandname_w, bandname_h = self.get_image_size(bandname_path)

        # create a complex filter that uses the jingle video and places the headline in the top part with a maximal width of 900px 
        # and the bandname in the bottom part with a maximal width of 900px
        # the jingle starts immediateld and fades into the dummy video after the start_transition time
        # the headline image fade in after 0.1 seconds and fades out after the end_start_transition time
        # 0:v - the jingle video
        # 1:v - the video to add the jingle start and end
        # 2:v - the headline image
        # 3:v - the bandname image

        
        filter_complex = f"""
            [0:v]format=yuva420p,fade=t=out:st={start_transition}:d={transition_duration}:alpha=1[v0];
            [1:v]format=yuva420p,tpad=start={start_transition*25},setpts=PTS-STARTPTS[cutv1];
            [2:v]scale=w=min(900\,iw):h=-1[v2];
            [3:v]scale=w=min(700\,iw):h=-1[v3];

            [v0][v2]overlay=x=(W-w)/2:y=(H-h)/2-300-h/2:format=auto:enable='between(t,0,{end_start_transition})'[v0];
            [v0][v3]overlay=x=(W-w)/2:y=(H-h)/2+300+h/2:format=auto:enable='between(t,0,{end_start_transition})'[v0];
            

            [cutv1]trim={start_time}:{duration+start_transition+transition_duration},setpts=PTS-STARTPTS,fade=t=in:st={start_transition}:d={transition_duration}:alpha=1,fade=t=out:st={end_start_transition}:d={transition_duration}:alpha=1[v1];
            [v0][v1]overlay,format=yuva420p[vid];
            [0:v]trim=start_frame=0:end_frame=1,loop={int((duration + jingle_time + still_duration - start_time)*25)}:1:0,setpts=N/FRAME_RATE/TB,format=yuva420p,fade=t=in:st={end_start_transition-transition_duration}:d={transition_duration}:alpha=1[still];

            [vid][still]overlay=enable='gte(t,{end_start_transition-transition_duration})',format=yuv420p[v];

            [0:a]afade=t=out:st={start_transition}:d={transition_duration}[a0];
            [1:a]adelay={start_transition*1000}|{start_transition*1000},asetpts=PTS-STARTPTS,atrim={start_time}:{duration+start_transition+transition_duration},asetpts=PTS-STARTPTS,afade=t=in:st={start_transition}:d={transition_duration},afade=t=out:st={end_start_transition-transition_duration}:d={transition_duration}[a1];
            [a0][a1]amix=inputs=2:dropout_transition={transition_duration}[aud];
            [aud]apad=pad_dur={still_duration}[a]
        """.replace("\n", "").replace(" ", "")        

        # Inputs
        command = [
            "ffmpeg", "-y",
            "-i", jingle_path,
            "-i", dummy_path,
            "-i", headline_path,
            "-i", bandname_path,
            "-filter_complex", filter_complex,
            "-map", "[v]", "-map", "[a]",
            "-preset", "slow", "-crf", "18",
            os.path.join(output_path, output_name)
        ]

        print("merge command")
        print(" ".join(command))

        return command


class PrepareJobs:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "target_node": ("INT",),
                "self_node": ("INT",),
                "character_id": ("INT", {"default": 1, "min": 1}),
                "output_path": ("STRING", {"default": ""}),
                "output_format": ("STRING", {"default": "landscape"}),
                "write_files": ("BOOLEAN", {"default": False, "tooltip": "Only enable if you want job files to be written"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING",)
    RETURN_NAMES = ("character_name", "character_audio_path", "character_image_path",)

    FUNCTION = "export_workflow"
    CATEGORY = "vk-nodes"
    OUTPUT_NODE = True
    old_kwargs = {}


    @classmethod
    def IS_CHANGED(cls, **kwargs):
        print("is changed ", kwargs)
        if "write_files" in kwargs and kwargs["write_files"]:
            return float("NaN")
        return kwargs


    def get_workflow(self):
        """ get the workflow. I guess things like this can change

        """

        from server import PromptServer
        prompt_queue = PromptServer.instance.prompt_queue

        currently_running = prompt_queue.currently_running
        key = list(currently_running.keys())[-1]
        return currently_running[key][3]["extra_pnginfo"]["workflow"]        

    def convert_editor_workflow_to_api_job(self, editor_workflow: dict) -> dict:
        api_workflow = {}

        # Build link map (from link ID to (node_id, output_index))
        def build_link_index(editor_workflow):
            link_map = {}
            for node in editor_workflow.get("nodes", []):
                for output_index, output in enumerate(node.get("outputs", [])):
                    links = output.get("links")
                    if links:
                        for link_id in links:
                            link_map[link_id] = (str(node["id"]), output_index)
            return link_map

        link_map = build_link_index(editor_workflow)

        # Find SetNode mappings: widgets_values → source (upstream node and output index)
        set_node_sources = {}
        node_id_to_node = {str(n["id"]): n for n in editor_workflow["nodes"]}

        for node in editor_workflow.get("nodes", []):
            if node["type"] == "SetNode":
                widgets_values = node.get("widgets_values")[0]
                input_conns = node.get("inputs", [])
                if input_conns and input_conns[0].get("link") is not None:
                    link_id = input_conns[0]["link"]
                    if link_id in link_map:
                        set_node_sources[widgets_values] = link_map[link_id]
                    else:
                        print(f"Warning: SetNode '{widgets_values}' has unresolvable link")
                else:
                    print(f"Warning: SetNode '{widgets_values}' has no input connection")

        # Prepare a reverse lookup: link ID → where it's consumed
        input_links_to_patch = {}  # get_node_id → actual input (from SetNode)

        for node in editor_workflow.get("nodes", []):
            if node["type"] == "GetNode":
                widgets_values = node.get("widgets_values")[0]
                if widgets_values not in set_node_sources:
                    print(f"Warning: GetNode '{widgets_values}' has no matching SetNode")
                    continue
                new_source = set_node_sources[widgets_values]
                node_id = str(node["id"])

                # All links pointing to this node's outputs should be redirected
                outputs = node.get("outputs", [])
                for output_index, output in enumerate(outputs):
                    for link_id in output.get("links", []):
                        input_links_to_patch[link_id] = new_source

        # Now build API job
        for node in editor_workflow.get("nodes", []):
            node_id = str(node["id"])
            class_type = node["type"]

            # Skip SetNode and GetNode — we've rerouted them already
            if class_type in ["SetNode", "GetNode"]:
                continue

            inputs = {}

            # widgets_values → positional inputs
            widget_names = []
            try:
                node_class = NODE_CLASS_MAPPINGS[class_type]
                input_types = node_class.INPUT_TYPES()
                print(input_types)
                widget_names = list(input_types.get("required", {}).keys())
                widget_names_optional = list(input_types.get("optional", {}).keys())
                widget_names.extend(widget_names_optional)
            except Exception as e:
                print(f"Could not resolve class {class_type}: {e}")

            if "widgets_values" in node:
                if isinstance(node["widgets_values"], dict):
                    inputs.update(node["widgets_values"])   

            if "widgets_values" in node:
                if isinstance(node["widgets_values"], list):
                    widge_name_delete_needed = len(node["widgets_values"]) != len(widget_names)

            # connection inputs
            if "inputs" in node:
                for input_entry in node["inputs"]:
                    name = input_entry["name"]
                    link_id = input_entry.get("link")
                    if link_id is not None:
                        # Check if we need to patch this link
                        if link_id in input_links_to_patch:
                            inputs[name] = list(input_links_to_patch[link_id])
                            if name in widget_names and widge_name_delete_needed:
                                widget_names.remove(name)                            
                        elif link_id in link_map:
                            inputs[name] = list(link_map[link_id])
                            if name in widget_names and widge_name_delete_needed:
                                widget_names.remove(name)
                        else:
                            print(f"Warning: Unresolved link {link_id} in node {node_id}")


            if "widgets_values" in node:
                if isinstance(node["widgets_values"], list):
                    for i, value in enumerate(node["widgets_values"]):
                        if i < len(widget_names):
                            if widget_names[i] not in inputs:
                                inputs[widget_names[i]] = value



            api_workflow[node_id] = {
                "inputs": inputs,
                "class_type": class_type,
                "_meta": {
                    "title": node.get("title", class_type)
                }
            }

        return api_workflow


    def filter_workflow(self, api_workflow: dict, target_node: int) -> dict:
        """
        Returns a filtered version of api_workflow, containing only the nodes
        required to compute the output of target_node.
        """
        required_nodes = set()
        queue = [str(target_node)]  # use strings for consistency with node IDs in api_workflow

        while queue:
            current = queue.pop()
            if current in required_nodes:
                continue
            required_nodes.add(current)

            node = api_workflow.get(current)
            if not node:
                print(f"Warning: Node {current} not found in API workflow")
                continue

            for input_value in node["inputs"].values():
                if isinstance(input_value, list) and len(input_value) == 2:
                    upstream_node = str(input_value[0])
                    queue.append(upstream_node)

        # Build filtered dictionary
        return {node_id: node for node_id, node in api_workflow.items() if node_id in required_nodes}


    def get_audio_for(self, output_path, index):
        all_videos = glob.glob(os.path.join(MY_OUTPUT_FOLDER, output_path, f"audio/{index}_*"))
        if len(all_videos) > 1:
            raise Exception(f"Multiple audio files for {index} in {output_path} audio")
        
        if len(all_videos) == 0:
            return None

        return all_videos[0]


    def get_stills_for(self, output_path, index):
        all_stills = glob.glob(os.path.join(MY_OUTPUT_FOLDER, output_path, f"stills/{index}_*"))
        if len(all_stills) > 1:
            raise Exception(f"Multiple image files for {index} in {output_path} stills")
        
        if len(all_stills) == 0:
            return None

        return all_stills[0]


    def replace_input_with_static_values(self,
        node_id,
        character_name,
        character_audio_path,
        character_image_path,
        api_workflow
    ):
        import copy
        api_workflow = copy.deepcopy(api_workflow)

        node_str = str(node_id)

        replace = {
            0: character_name,
            1: character_audio_path.replace("\\\\", "/").replace("\\", "/"),
            2: character_image_path.replace("\\\\", "/").replace("\\", "/"),
        }
        print(replace)

        for nk, nv in api_workflow.items():
            if "inputs" in nv:
                for ik, iv in nv["inputs"].items():
                    if isinstance(iv, list) and iv[0] == node_str:
                        nv["inputs"][ik] = replace[iv[1]]


        return api_workflow


    def remove_reroutes(self, api_workflow):
        """ remove reroutes
        """
        reroutes = {k:v for k, v in api_workflow.items() if v["class_type"] == "Reroute"}
        
        for nk, nv in api_workflow.items():
            if "inputs" in nv:
                for ik, iv in nv["inputs"].items():
                    if isinstance(iv, list) and iv[0] in reroutes:
                        nv["inputs"][ik] = reroutes[iv[0]]["inputs"][""]

        return {k:v for k, v in api_workflow.items() if k not in reroutes}
            

    def fix_paths(self, api_workflow):
        def fix(obj):
            if isinstance(obj, dict):
                return {k: fix(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [fix(item) for item in obj]
            elif isinstance(obj, str):
                return obj.replace("\\", "/")
            else:
                return obj

        return fix(api_workflow)




    def bundle_workflow_to_zip(
        self,
        workflow_data,
        character_id,
        character_name,
        character_audio_path,
        character_image_path,
        output_dir,
        base_filename
    ):
        """
        Saves workflow JSON, media files, and metadata into a ZIP archive.

        Parameters:
        - workflow_data (dict): The API workflow JSON.
        - character_id (int): Character ID (e.g., 1-6).
        - character_name (str): Full character name (e.g., '3_Ada').
        - character_audio_path (str): Path to audio file.
        - character_image_path (str): Path to image file.
        - output_dir (str): Directory to save files.
        - base_filename (str): Base filename for the zip and json (e.g., '3_Ada_json').
        - generated_by (str): Optional identifier for tracking origin.
        """

        import os
        import json
        import zipfile
        from datetime import datetime        
        
        os.makedirs(output_dir, exist_ok=True)

        json_filename = "run.json"
        json_path = os.path.join(output_dir, json_filename)

        # Save workflow JSON
        with open(json_path, "w") as f:
            json.dump(workflow_data, f, indent=2)

        # Prepare info.json
        info = {
            "character_id": character_id,
            "character_name": character_name,
            "audio_path": character_audio_path,
            "image_path": character_image_path,
            "workflow_json": json_filename,
            "timestamp": datetime.now().isoformat()
        }

        info_filename = "info.json"
        info_path = os.path.join(output_dir, info_filename)

        with open(info_path, "w") as f:
            json.dump(info, f, indent=2)

        # Create ZIP archive
        zip_path = os.path.join(output_dir, f"{base_filename}.zip")
        with zipfile.ZipFile(zip_path, "w") as zipf:
            zipf.write(json_path, arcname=json_filename)
            zipf.write(info_path, arcname=info_filename)
            zipf.write(character_audio_path, arcname="audio_file")
            zipf.write(character_image_path, arcname="image_file")

        # Optional: clean up loose files
        os.remove(json_path)
        os.remove(info_path)

        return zip_path
    

    def export_workflow(self, target_node, self_node, character_id, output_path, output_format, write_files):

        print("Export Workflow")

        output_format = output_format.replace(".mp4", "")

        name_dict = {
                1: "Sophia",
                2: "Melina",
                3: "Ada",
                4: "Taro",
                5: "Bari",
                6: "Barney"
        }

        if write_files:
            full_workflow = self.get_workflow()
            api_workflow = self.convert_editor_workflow_to_api_job(full_workflow)

            for k, v in name_dict.items():

                character_name = f"{k}_{name_dict[k]}"
                character_audio_path = self.get_audio_for(output_path, k).replace("\\\\", "/").replace("\\", "/")
                character_image_path = self.get_stills_for(output_path, k).replace("\\\\", "/").replace("\\", "/")

                new_workflow = self.replace_input_with_static_values(
                    self_node,
                    character_name,
                    character_audio_path,
                    character_image_path,
                    api_workflow
                )

                filtered_api_workflow = self.filter_workflow(new_workflow, target_node)

                filtered_api_workflow = self.remove_reroutes(filtered_api_workflow)

                filtered_api_workflow = self.fix_paths(filtered_api_workflow)
                
                # output_file_name = os.path.join(MY_OUTPUT_FOLDER, output_path, "jobs", f"{output_path}_{k}_{v}_{output_format}.json")

                output_dir = os.path.join(MY_OUTPUT_FOLDER, output_path, "jobs")
                base_filename = f"{output_path}_{k}_{v}_{output_format}"

                self.bundle_workflow_to_zip(
                        filtered_api_workflow,
                        character_id,
                        character_name,
                        character_audio_path,
                        character_image_path,
                        output_dir,
                        base_filename
                    )
                

        character_name = f"{character_id}_{name_dict[character_id]}"
        character_audio_path = self.get_audio_for(output_path, character_id)
        character_image_path = self.get_stills_for(output_path, character_id)

        return character_name, character_audio_path, character_image_path




# Register nodes in ComfyUI
NODE_CLASS_MAPPINGS.update(
    {
        "TiledRenderNode": TiledRenderNode,
        "PrepareJobs": PrepareJobs,
    }
)

NODE_DISPLAY_NAME_MAPPINGS.update(
    {
        "TiledRenderNode": "Tiled Render",
        "PrepareJobs": "Prepare Jobs",
    }
)
