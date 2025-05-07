import numpy as np
from freetype import Face
from PIL import Image, ImageDraw, ImageFilter
from fontTools.pens.basePen import BasePen
from fontTools.ttLib import TTFont
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
import colorsys
import os
from .vk_tiled_render import MY_OUTPUT_FOLDER
import torch


from nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS



class ShapelyPen(BasePen):
    def __init__(self, glyphSet, dx=0, dy=0, sketchiness=1.0, smooth_iters=10):
        super().__init__(glyphSet)
        self.contours = []  # This will hold the outer contours
        self.holes = []     # This will hold the inner contours (holes)
        self.current = []
        self.dx = dx
        self.dy = dy
        self.sketchiness = sketchiness
        self.smooth_iters = smooth_iters
        self.error = False

    def _moveTo(self, p):
        self.current = [self._jitter_point(p)]


    def _lineTo(self, p):
        self.current.append(self._jitter_point(p))

    def _qCurveToOne(self, p1, p2):
        self.current.append(self._jitter_point(p1))
        self.current.append(self._jitter_point(p2))

    def _curveToOne(self, p1, p2, p3):
        self.current.append(self._jitter_point(p1))
        self.current.append(self._jitter_point(p2))
        self.current.append(self._jitter_point(p3))

    def _closePath(self):
        if len(self.current) > 2:
            #pts = chaikin_smooth(self.current, self.smooth_iters)
            pts = chaikin_smooth_preserve_corners(self.current, self.smooth_iters)
            if is_hole(pts):  # Check if the contour is a hole
                self._mark_as_hole(pts)  # This marks it as a hole (inner contour)
            else:
                self.contours.append(pts)  # This is the outer contour
        self.current = []

    def _endPath(self):
        self._closePath()

    def _jitter_point(self, pt):
        return (
            pt[0] + self.dx + np.random.normal(0, self.sketchiness),
            -pt[1] + self.dy + np.random.normal(0, self.sketchiness)
        )

    def get_geometry(self):
        polygons = []

        for contour in self.contours:
            try:
                poly = Polygon(contour)
                if not poly.is_valid or poly.is_empty or poly.area < 1e-4:  # Add area check
                    self.error = True
                    continue
                polygons.append(poly)
            except Exception:
                self.error = True
                continue

        return MultiPolygon(polygons) if polygons else None

    def get_holes(self):
        polygons = []

        for hole in self.holes:
            try:
                poly = Polygon(hole)
                if not poly.is_valid or poly.is_empty or poly.area < 1e-4:  # Add area check
                    self.error = True
                    continue
                polygons.append(poly)
            except Exception:
                self.error = True
                continue

        return MultiPolygon(polygons) if polygons else None

    def _add_hole(self, hole_contour):
        """Method to add an inner contour as a hole."""
        self.holes.append(hole_contour)

    def _mark_as_hole(self, contour):
        """Mark a contour as a hole."""
        self._add_hole(contour)


def is_hole(contour):
    poly = Polygon(contour)
    if not poly.is_valid:
        return False
    return poly.exterior.is_ccw  # If it's clockwise, it's a hole


def detect_hard_corners(points, angle_threshold_deg=80):
    """Returns a boolean array marking sharp corners as True."""
    hard_corners = [False] * len(points)
    threshold = np.cos(np.radians(angle_threshold_deg))
    
    for i in range(len(points)):
        p0 = np.array(points[i - 1])
        p1 = np.array(points[i])
        p2 = np.array(points[(i + 1) % len(points)])

        v1 = p0 - p1
        v2 = p2 - p1
        v1 /= np.linalg.norm(v1)
        v2 /= np.linalg.norm(v2)

        dot = np.dot(v1, v2)
        if dot < threshold:  # Smaller dot => larger angle
            hard_corners[i] = True
    return hard_corners

def chaikin_smooth(points, iterations=2, tolerance=1e-2):
    for _ in range(iterations):
        new_points = []
        for i in range(len(points)):
            p0 = points[i]
            p1 = points[(i + 1) % len(points)]
            q = (0.75 * np.array(p0) + 0.25 * np.array(p1)).tolist()
            r = (0.25 * np.array(p0) + 0.75 * np.array(p1)).tolist()
            new_points.extend([q, r])
        points = new_points

    # Filter out points that are too close together
    filtered_points = []
    for i in range(len(points)):
        if i == 0 or np.linalg.norm(np.array(points[i]) - np.array(points[i-1])) > tolerance:
            filtered_points.append(points[i])
    
    return filtered_points    

def chaikin_smooth_preserve_corners(points, iterations=2, angle_threshold_deg=45, tolerance=1e-2):
    for _ in range(iterations):
        hard_corners = detect_hard_corners(points, angle_threshold_deg)
        new_points = []

        for i in range(len(points)):
            p0 = np.array(points[i])
            p1 = np.array(points[(i + 1) % len(points)])
            is_hard = hard_corners[i] or hard_corners[(i + 1) % len(points)]

            if is_hard:
                new_points.append(p0.tolist())  # Preserve point without smoothing
                continue

            q = (0.75 * p0 + 0.25 * p1).tolist()
            r = (0.25 * p0 + 0.75 * p1).tolist()
            new_points.extend([q, r])

        points = new_points

    # Filter very close points
    filtered_points = []
    for i in range(len(points)):
        if i == 0 or np.linalg.norm(np.array(points[i]) - np.array(points[i-1])) > tolerance:
            filtered_points.append(points[i])

    return filtered_points    


def render_sketchy_rainbow_text_with_holes(
    text,
    font_path="./fonts/MYRIADPRO-BOLD.OTF",
    font_size=120,
    canvas_padding=120,
    rainbow_start=0.3,
    rainbow_cycles=2,
    sketchiness=5.0,
    bg_blur_radius=10,
    bg_offset=(0, 5),
    bg_expand=90,
    bg_opacity=0.99,
    rainbow_saturation=0.7,
    rainbow_brightness=0.7,
    line_spacing=8,
    outline_width=5,
    outline_color=(0, 0, 0, 255)
):

    face = Face(font_path)
    face.set_char_size(font_size * 64)
    font = TTFont(font_path)
    glyph_set = font.getGlyphSet()

    pen_records = []
    pen_holes = []
    y_cursor = 0
    for line_num, line in enumerate(text.splitlines()):
        line_width = 0
        for char in line:
            if char == " ":
                space_glyph = font.getGlyphSet().get('space', None)
                space_width = space_glyph.width if space_glyph else 0
                line_width += space_width
                continue  # Skip spaces, no geometry drawn for them
            line_width += font.getGlyphSet()[char].width  # Adjust based on your units

        x_cursor = -line_width / 2
        for char in line:
            if char == " ":
                space_glyph = font.getGlyphSet().get('space', None)
                space_width = space_glyph.width if space_glyph else 0
                x_cursor += space_width
                continue  # Skip spaces, no geometry drawn for them
            
            problem = True
            while problem:
                
                pen = ShapelyPen(glyph_set, dx=x_cursor, dy=y_cursor, sketchiness=sketchiness, smooth_iters=3)

                glyph_set[char].draw(pen)

                poly = pen.get_geometry()
                holes = pen.get_holes()

                if pen.error:
                    continue

                if poly:
                    pen_records.append((char, poly, x_cursor, y_cursor))  # Store geometry with position
                
                if holes:
                    pen_holes.append((char, holes, x_cursor, y_cursor))  # Store geometry with position

                problem = False

            x_cursor += font.getGlyphSet()[char].width
        y_cursor += font_size * line_spacing

    char_polys = [poly for char, poly, dx, dy in pen_records if poly and poly.is_valid]
    char_holes = [hole for char, multi_poly, dx, dy in pen_holes for hole in multi_poly.geoms if hole and hole.is_valid]


    # Apply hole detection and subtraction
    union = unary_union(char_polys)
    minx, miny, maxx, maxy = union.bounds
    width = int(maxx - minx + 2 * canvas_padding)
    height = int(maxy - miny + 2 * canvas_padding)

    def to_canvas_coords(poly):
        if isinstance(poly, Polygon):
            return [[(x - minx + canvas_padding, y - miny + canvas_padding) for x, y in poly.exterior.coords]]
        elif isinstance(poly, MultiPolygon):
            return [[(x - minx + canvas_padding, y - miny + canvas_padding) for x, y in p.exterior.coords] for p in poly.geoms]
        return []

    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))

    bg_mask = Image.new("L", (width, height), 0)
    draw_bg = ImageDraw.Draw(bg_mask)
    for poly in [p.buffer(bg_expand) for p in char_polys]:
        for coords in to_canvas_coords(poly):
            draw_bg.polygon(coords, fill=255)
    blurred = bg_mask.filter(ImageFilter.GaussianBlur(bg_blur_radius))
    shadow = Image.new("RGBA", (width, height), (255, 255, 255, 0))
    shadow.paste((255, 255, 255, int(255 * bg_opacity)), box=bg_offset, mask=blurred)
    img.alpha_composite(shadow)

    shadow_mask = Image.new("L", (width, height), 0)
    draw_shadow = ImageDraw.Draw(shadow_mask)
    for poly in char_polys:
        for coords in to_canvas_coords(poly):
            draw_shadow.polygon(coords, fill=255)
    shadow_layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    shadow_layer.paste((0, 0, 0, 100), box=(0, 14), mask=shadow_mask.filter(ImageFilter.GaussianBlur(7)))
    img.alpha_composite(shadow_layer)

    total = len(char_polys)
    for i, poly in enumerate(char_polys):
        hue = ( rainbow_start + i / total) * rainbow_cycles % 1.0
        r, g, b = [int(x * 255) for x in colorsys.hsv_to_rgb(hue, rainbow_saturation, rainbow_brightness)]
        mask = Image.new("L", (width, height), 0)
        draw_mask = ImageDraw.Draw(mask)
        for coords in to_canvas_coords(poly):
            draw_mask.polygon(coords, fill=255)
        layer = Image.new("RGBA", (width, height), (r, g, b, 255))
        img.paste(layer, (0, 0), mask)

    draw = ImageDraw.Draw(img)
    for poly in char_polys:
        for coords in to_canvas_coords(poly):
            draw.line(coords + [coords[0]], fill=outline_color, width=outline_width)

    # Handle holes correctly by subtracting them from the main contours
    for hole in char_holes:
        mask = Image.new("L", (width, height), 0)
        draw_mask = ImageDraw.Draw(mask)
        for coords in to_canvas_coords(hole):
            draw_mask.polygon(coords, fill=255)
        hole_layer = Image.new("RGBA", (width, height), (255, 255, 255, int(255 * bg_opacity)))
        img.paste((0, 0, 0, 0), (0, 0), mask)
        
        img.paste(hole_layer, (0, 0), mask)

    draw = ImageDraw.Draw(img)
    for poly in char_holes:
        for coords in to_canvas_coords(poly):
            draw.line(coords + [coords[0]], fill=outline_color, width=outline_width)        

    return img

def render_sketchy_text(
    text,
    rainbow_start = None,
    bg_opacity=0.99,
    circle_increment = 0.11,
    saturation = 0.7,
    brightness = 0.7,
):
    rainbow_cycles = len(text) * circle_increment
    if rainbow_start is None:
        rainbow_start = np.random.random(1)[0]
    return render_sketchy_rainbow_text_with_holes(
        text=text,
        rainbow_cycles=rainbow_cycles,
        rainbow_start=rainbow_start,
        bg_opacity=bg_opacity,
        rainbow_saturation=saturation,
        rainbow_brightness=brightness,
    )



if __name__ == "__main__":
    img = render_sketchy_text(
        text="TheSketchTones",
        rainbow_start=0.17,
        bg_opacity=0.6
    )

    img.save("test_output.png")





class SketchyText:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
            "output_path": ("STRING", {"default": ""}),
            "output_name": ("STRING", {"default": "headline.png"}),
            "text": ("STRING", {"default": "TheSketchTones"}), 
            "rainbow_start": ("FLOAT", {"default": 0.17, "min": 0.0, "max": 1.0, "step": 0.01}),
            "bg_opacity": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.01}),
            "circle_increment": ("FLOAT", {"default": 0.11, "min": 0.0, "max": 1.0, "step": 0.01}),
            "saturation": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01}),
            "brightness": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01}),

            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    FUNCTION = "render_image"
    CATEGORY = "vk-nodes"
    OUTPUT_NODE = True
    old_kwargs = {}


    

    def render_image(self, output_path, output_name, text, rainbow_start, bg_opacity, circle_increment, saturation, brightness):

        outputs = []

        full_output_path = os.path.join(MY_OUTPUT_FOLDER, output_path, "overlays", output_name)

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        if os.path.exists(full_output_path):
            os.remove(full_output_path)

        image = render_sketchy_text(
            text=text,
            rainbow_start=rainbow_start,
            bg_opacity=bg_opacity,
            circle_increment=circle_increment,
            saturation=saturation,
            brightness=brightness

        )

        image.save(full_output_path)


        image_out = np.array(image)
        image_out = np.clip(image_out, 0, 255).astype(np.uint8)

        outputs.append(torch.from_numpy(image_out.astype(np.float32) / 255.0))

        return (outputs,)
        

class SketchyThumbnail :
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
            "output_path": ("STRING", {"default": ""}),
            "output_name": ("STRING", {"default": "headline.png"}),
            "timestamp": ("FLOAT", {"default": 5.0, "min": 0.0}),
            "text": ("STRING", {"default": "TheSketchTones"}), 
            "rainbow_start": ("FLOAT", {"default": 0.17, "min": 0.0, "max": 1.0, "step": 0.01}),
            "bg_opacity": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.01}),
            "circle_increment": ("FLOAT", {"default": 0.11, "min": 0.0, "max": 1.0, "step": 0.01}),
            "saturation": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01}),
            "brightness": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01}),
            "placement": (["top-left", "top-right", "bottom-left", "bottom-right", "center"], {"default": "top-left"}),
            "scale": ("FLOAT", {"default": .5, "min": 0.0, "max": 1.0, "step": 0.01}),
            "rotation": ("FLOAT", {"default": 0.0, "min": -360.0, "max": 360.0, "step": 1.0}),


            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    FUNCTION = "render_image"
    CATEGORY = "vk-nodes"
    OUTPUT_NODE = True
    old_kwargs = {}


    def render_image(self, output_path, output_name, timestamp, text, rainbow_start, bg_opacity, circle_increment, saturation, brightness, placement, scale, rotation):

        outputs = []

        full_video_path = os.path.join(MY_OUTPUT_FOLDER, output_path, output_name)
        full_output_path = os.path.join(MY_OUTPUT_FOLDER, output_path, output_name.replace(".mp4", ".png"))
        dummy_output_path = os.path.join(MY_OUTPUT_FOLDER, output_path, "dummy.png")

        if not os.path.exists(output_path):
            os.makedirs(output_path)


        image = render_sketchy_text(
            text=text,
            rainbow_start=rainbow_start,
            bg_opacity=bg_opacity,
            circle_increment=circle_increment,
            saturation=saturation,
            brightness=brightness
        )


        # use ffmpeg to extract the frame from the video at the given timestamp
        ffmpeg_cmd = [
            "ffmpeg",
            "-i", full_video_path,
            "-ss", str(timestamp),
            "-vframes", "1",
            "-q:v", "2",
            "-y",
            dummy_output_path,
        ]

        os.system(" ".join(ffmpeg_cmd))
        # check if the dummy output path exists and is not empty

        if os.path.exists(dummy_output_path) and os.path.getsize(dummy_output_path) > 0:
            # open the dummy image and the sketchy image
            dummy_image = Image.open(dummy_output_path).convert("RGBA")
            sketchy_image = image.convert("RGBA")

            # rotate the sketchy image
            sketchy_image = sketchy_image.rotate(rotation, expand=True)

            # use the scale the get the max size of the sketchy image from the dummy image
            dummy_width, dummy_height = dummy_image.size
            max_width = int(dummy_width * scale)
            max_height = int(dummy_height * scale)

            ratio_sketchy = sketchy_image.size[0] / sketchy_image.size[1]

            if ratio_sketchy > 1.0:
                new_width = max_width
                new_height = int(max_width / ratio_sketchy)
            else:
                new_height = max_height
                new_width = int(max_height * ratio_sketchy)

            sketchy_image = sketchy_image.resize((new_width, new_height), Image.LANCZOS)

            #use the placement to place the sketchy image on the dummy image
            if placement == "top-left":
                x_offset = 0
                y_offset = 0
            elif placement == "top-right":
                x_offset = dummy_width - new_width
                y_offset = 0
            elif placement == "bottom-left":
                x_offset = 0
                y_offset = dummy_height - new_height
            elif placement == "bottom-right":
                x_offset = dummy_width - new_width
                y_offset = dummy_height - new_height
            elif placement == "center":
                x_offset = (dummy_width - new_width) // 2
                y_offset = (dummy_height - new_height) // 2
            else:
                x_offset = 0
                y_offset = 0

            # create a new image with the same size as the dummy image
            combined_image = Image.new("RGBA", dummy_image.size, (0, 0, 0, 0))
            # paste the dummy image and the sketchy image on top of it
            combined_image.paste(dummy_image, (0, 0))
            combined_image.paste(sketchy_image, (x_offset, y_offset), mask=sketchy_image)

            # save the combined image
            combined_image.save(full_output_path)

            # add it to the outputs
            image_out = np.array(combined_image)
            image_out = np.clip(image_out, 0, 255).astype(np.uint8)
            outputs.append(torch.from_numpy(image_out.astype(np.float32) / 255.0))
        else:
            # if the dummy image is not valid, just save the sketchy image
            image.save(full_output_path)

            # add it to the outputs
            image_out = np.array(image)
            image_out = np.clip(image_out, 0, 255).astype(np.uint8)
            outputs.append(torch.from_numpy(image_out.astype(np.float32) / 255.0))

        # remove the dummy image
        if os.path.exists(dummy_output_path):
            os.remove(dummy_output_path)

            
        return (outputs,)


# Register nodes in ComfyUI
NODE_CLASS_MAPPINGS.update(
    {
        "SketchyText": SketchyText,
        "SketchyThumbnail": SketchyThumbnail,
    }
)

NODE_DISPLAY_NAME_MAPPINGS.update(
    {
        "SketchyText": "Sketchy Text",
        "SketchyThumbnail": "Sketchy Thumbnail",
    }
)