import os
import subprocess
import shutil
import torch
import re
import math
import numpy as np
from nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

ENCODE_ARGS = ("utf-8", 'backslashreplace')

# Minimal stubs (replace with your actual Comphy UI helpers)
def strip_path(path): return path
def validate_path(path): return os.path.exists(path)
def is_url(path): return path.startswith("http://") or path.startswith("https://")
def try_download_video(url): return None  # implement if needed

import logging
logger = logging.getLogger(__name__)

# --- FFMPEG detection ---
if "VHS_FORCE_FFMPEG_PATH" in os.environ:
    ffmpeg_path = os.environ.get("VHS_FORCE_FFMPEG_PATH")
else:
    ffmpeg_paths = []
    try:
        from imageio_ffmpeg import get_ffmpeg_exe
        imageio_ffmpeg_path = get_ffmpeg_exe()
        ffmpeg_paths.append(imageio_ffmpeg_path)
    except:
        if "VHS_USE_IMAGEIO_FFMPEG" in os.environ:
            raise
        logger.warning("Failed to import imageio_ffmpeg")

    if "VHS_USE_IMAGEIO_FFMPEG" in os.environ:
        ffmpeg_path = imageio_ffmpeg_path
    else:
        system_ffmpeg = shutil.which("ffmpeg")
        if system_ffmpeg is not None:
            ffmpeg_paths.append(system_ffmpeg)
        for fname in ("ffmpeg", "ffmpeg.exe"):
            if os.path.isfile(fname):
                ffmpeg_paths.append(os.path.abspath(fname))
        if not ffmpeg_paths:
            logger.error("No valid ffmpeg found.")
            ffmpeg_path = None
        else:
            ffmpeg_path = ffmpeg_paths[0]

def normalize_audio(waveform, target_rms=0.1, eps=1e-6):
    # Peak normalization
    peak = waveform.abs().max()
    if peak > 0:
        waveform = waveform / peak

    # RMS boost if too quiet
    rms = waveform.pow(2).mean().sqrt()
    if rms < target_rms:
        gain = target_rms / (rms + eps)
        waveform = waveform * gain
    return waveform

def add_input_shaped_noise(audio, level=0.01, seed=42):
    """
    Adds low-level, spectrally-shaped noise based on the input audio.
    """
    num_channels, num_samples = audio.shape[-2], audio.shape[-1]

    audio_np = audio.cpu().numpy()

    rng = np.random.default_rng(seed)
    shaped_noise = []

    for ch in range(num_channels):
        # FFT of the original signal
        spectrum = np.fft.rfft(audio_np[ch])
        mag = np.abs(spectrum)
        
        # Random phase
        random_phase = np.exp(1j * rng.uniform(0, 2 * np.pi, len(mag)))
        
        # Construct new spectrum: keep magnitude, random phase
        noisy_spectrum = mag * random_phase
        
        # IFFT to get back to time domain
        shaped = np.fft.irfft(noisy_spectrum, n=num_samples)

        # Normalize and scale
        shaped = shaped.astype(np.float32)
        shaped /= np.max(np.abs(shaped)) + 1e-8
        shaped_noise.append(shaped * level)

    # Stack and convert back to torch
    shaped_noise = torch.from_numpy(np.stack(shaped_noise))
    return audio + shaped_noise

def get_audio(file, start_time=0, duration=0, hum_volume=0.0):
    args = [ffmpeg_path, "-i", file]
    if start_time > 0:
        args += ["-ss", str(start_time)]
    if duration > 0:
        args += ["-t", str(duration)]

    try:
        res = subprocess.run(args + ["-f", "f32le", "-"],
                             capture_output=True, check=True)
        audio = torch.frombuffer(res.stdout, dtype=torch.float32)
        stderr_text = res.stderr.decode(*ENCODE_ARGS)
        match = re.search(r', (\d+) Hz, (\w+),', stderr_text)
    except subprocess.CalledProcessError as e:
        raise Exception(f"VHS failed to extract audio from {file}:\n" +
                        e.stderr.decode(*ENCODE_ARGS))

    if match:
        ar = int(match.group(1))
        ac = {"mono": 1, "stereo": 2}[match.group(2)]
    else:
        ar = 44100
        ac = 2

    audio = audio.reshape((-1, ac)).transpose(0, 1).unsqueeze(0)  # [1, C, T]

    audio = normalize_audio(audio)
    if hum_volume > 0.0:
        audio = add_input_shaped_noise(audio.squeeze(0), level=hum_volume).unsqueeze(0)

    return {'waveform': audio, 'sample_rate': ar}

class LoadAudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_file": ("STRING", {"default": "input/"}),
                "start_time": ("FLOAT", {"default": 0, "min": 0}),
                "duration": ("FLOAT", {"default": 0, "min": 0}),
                "hum_volume": ("FLOAT", {"default": 0.01, "min": 0, "max": 1, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    CATEGORY = "vk-nodes"
    FUNCTION = "load_audio"

    def load_audio(self, audio_file, start_time, duration, hum_volume):
        audio_file = strip_path(audio_file)
        if audio_file is None or not validate_path(audio_file):
            raise Exception("audio_file is not a valid path: " + str(audio_file))
        if is_url(audio_file):
            audio_file = try_download_video(audio_file) or audio_file

        return (get_audio(audio_file, start_time=start_time, duration=duration, hum_volume=hum_volume),)


# Register nodes in ComfyUI
NODE_CLASS_MAPPINGS.update(
    {
        "LoadAudio": LoadAudio
    }
)

NODE_DISPLAY_NAME_MAPPINGS.update(
    {
        "LoadAudio": "VK Load Audio"
    }
)
