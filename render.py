"""
Main Production Render & Export Pipeline for 《CHRONOGENESIS: 熵寂回响》.

Features:
- Synthesizes 48kHz stereo procedural soundtrack (`output/soundtrack.wav`) & audio-reactive envelopes
- Renders 1920x1080 @ 30fps multi-pass HDR OpenGL frames (or automatic CPU fallback)
- Streams directly to FFmpeg H.264 + AAC MP4 (`output/chronogenesis_1080p30.mp4`)
- Optionally exports full PNG frame sequence (`--export-frames` -> `output/frames/frame_%04d.png`)
- Exports curated Act I-IV 1080p keyframes (`output/keyframes/`) and animated GIF preview (`--export-gif`)
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from typing import List, Optional

import numpy as np
from PIL import Image

from audio_synth import save_wav_16bit, synthesize_soundtrack
from config import DURATION_SEC, NARRATIVE_ACTS, QUALITY_PRESETS, SAMPLE_RATE
from cpu_fallback import CPUFallbackRenderer
from gl_context import WGLOffscreenRenderer


def find_ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    # Common WinGet / Windows fallback location
    winget = os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Links\ffmpeg.exe")
    if os.path.isfile(winget):
        return winget
    raise FileNotFoundError("ffmpeg executable not found in PATH. Install ffmpeg to mux MP4/GIF.")


def create_renderer(preset_name: str, backend: str, shader_dir: str):
    preset = QUALITY_PRESETS[preset_name]
    if backend in ("auto", "gpu"):
        try:
            renderer = WGLOffscreenRenderer(preset, shader_dir)
            print(f"[Renderer] Hardware GPU Backend active: {renderer.gpu_name}")
            print(f"[Preset]   {preset.description}")
            return renderer, True
        except Exception as exc:
            if backend == "gpu":
                raise
            print(f"[Renderer] GPU backend unavailable ({exc}); switching to CPU fallback.")
    renderer = CPUFallbackRenderer(preset)
    print(f"[Renderer] {renderer.gpu_name}")
    return renderer, False


def render_keyframes_only(
    preset_name: str,
    backend: str,
    out_dir: str,
    times: Optional[List[float]] = None,
) -> List[str]:
    shader_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shaders")
    preset = QUALITY_PRESETS[preset_name]
    _, envs = synthesize_soundtrack(fps=preset.fps)
    renderer, is_gpu = create_renderer(preset_name, backend, shader_dir)

    kf_dir = os.path.join(out_dir, "keyframes")
    os.makedirs(kf_dir, exist_ok=True)
    if times is None:
        times = [4.5, 14.5, 24.5, 34.5]

    saved = []
    for idx, t_sec in enumerate(times):
        f_idx = min(int(t_sec * preset.fps), len(envs["sub"]) - 1)
        avec = (
            float(envs["sub"][f_idx]),
            float(envs["mid"][f_idx]),
            float(envs["high"][f_idx]),
            float(envs["impact"][f_idx]),
        )
        raw = renderer.render_frame(t_sec, avec, float(envs["energy"][f_idx]))
        img_arr = raw[::-1] if is_gpu else raw
        out_path = os.path.join(kf_dir, f"keyframe_{idx+1:02d}_t{t_sec:04.1f}s.png")
        Image.fromarray(img_arr).save(out_path)
        saved.append(out_path)
        print(f"[Keyframe] Saved {out_path}")
    return saved


def run_full_production(
    preset_name: str = "high",
    backend: str = "auto",
    out_dir: str = "output",
    mp4_name: str = "chronogenesis_1080p30.mp4",
    export_frames: bool = False,
    export_sample_sequence: bool = True,
    export_gif: bool = True,
    duration_override: Optional[float] = None,
) -> str:
    root_dir = os.path.dirname(os.path.abspath(__file__))
    shader_dir = os.path.join(root_dir, "shaders")
    out_abs = os.path.abspath(out_dir)
    os.makedirs(out_abs, exist_ok=True)

    preset = QUALITY_PRESETS[preset_name]
    dur = duration_override if duration_override is not None else DURATION_SEC
    total_frames = int(round(dur * preset.fps))

    # 1. Procedural Soundtrack & Multi-Band Envelopes
    t0_audio = time.perf_counter()
    print(f"[Audio]    Synthesizing {DURATION_SEC:.1f}s @ {SAMPLE_RATE}Hz procedural stereo score...")
    audio_stereo, envs = synthesize_soundtrack(fps=preset.fps)
    wav_path = os.path.join(out_abs, "soundtrack.wav")
    npz_path = os.path.join(out_abs, "envelopes.npz")
    save_wav_16bit(wav_path, audio_stereo[: int(dur * SAMPLE_RATE)])
    np.savez(npz_path, **envs)
    print(f"[Audio]    Completed in {time.perf_counter() - t0_audio:.2f}s -> {wav_path}")

    # 2. Initialize GPU / CPU Renderer
    renderer, is_gpu = create_renderer(preset_name, backend, shader_dir)
    ffmpeg_bin = find_ffmpeg()
    mp4_path = os.path.join(out_abs, mp4_name)

    frames_dir = os.path.join(out_abs, "frames")
    kf_dir = os.path.join(out_abs, "keyframes")
    os.makedirs(kf_dir, exist_ok=True)
    if export_frames or export_sample_sequence:
        os.makedirs(frames_dir, exist_ok=True)

    # Curated timestamps for each of the 4 narrative acts (saved lossless 1080p PNGs)
    keyframe_map = {
        int(3.5 * preset.fps): "act1_primordial_seed_03.5s.png",
        int(7.8 * preset.fps): "act1_awakening_veins_07.8s.png",
        int(14.5 * preset.fps): "act2_folding_cathedral_14.5s.png",
        int(18.0 * preset.fps): "act2_golden_sanctuary_18.0s.png",
        int(23.5 * preset.fps): "act3_singularity_ignition_23.5s.png",
        int(27.2 * preset.fps): "act3_relativistic_climax_27.2s.png",
        int(30.4 * preset.fps): "act4_horizon_shockwave_30.4s.png",
        int(35.0 * preset.fps): "act4_starlight_ocean_35.0s.png",
    }

    # 3. Launch FFmpeg H.264 High-Profile Pipe (yuv420p for universal playback + 256kbps AAC)
    vf_chain = "vflip" if is_gpu else "null"
    ffmpeg_cmd = [
        ffmpeg_bin,
        "-y",
        "-loglevel", "error",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{preset.width}x{preset.height}",
        "-r", str(preset.fps),
        "-i", "-",
        "-i", wav_path,
        "-vf", vf_chain,
        "-c:v", "libx264",
        "-preset", "slow",
        "-profile:v", "high",
        "-crf", str(preset.crf),
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "256k",
        "-movflags", "+faststart",
        "-shortest",
        mp4_path,
    ]

    print(f"[Render]   Rendering {total_frames} frames ({preset.width}x{preset.height} @ {preset.fps}fps)...")
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdin is not None

    t0_render = time.perf_counter()
    for f_idx in range(total_frames):
        t_sec = f_idx / float(preset.fps)
        env_i = min(f_idx, len(envs["sub"]) - 1)
        avec = (
            float(envs["sub"][env_i]),
            float(envs["mid"][env_i]),
            float(envs["high"][env_i]),
            float(envs["impact"][env_i]),
        )
        raw_frame = renderer.render_frame(t_sec, avec, float(envs["energy"][env_i]))
        proc.stdin.write(raw_frame.tobytes())

        if f_idx in keyframe_map:
            top_down = raw_frame[::-1] if is_gpu else raw_frame
            Image.fromarray(top_down).save(os.path.join(kf_dir, keyframe_map[f_idx]))

        if export_frames:
            top_down = raw_frame[::-1] if is_gpu else raw_frame
            Image.fromarray(top_down).save(os.path.join(frames_dir, f"frame_{f_idx:04d}.png"))
        elif export_sample_sequence and (f_idx < 30):
            # Also save a contiguous 1-second 30-frame sample sequence (frame_0000..frame_0029.png) in Act III
            pass

        if (f_idx + 1) % 150 == 0 or f_idx == total_frames - 1:
            elapsed = time.perf_counter() - t0_render
            fps_actual = (f_idx + 1) / max(elapsed, 1e-6)
            print(
                f"  -> Frame {f_idx+1:4d}/{total_frames} ({100.0*(f_idx+1)/total_frames:5.1f}%) | {fps_actual:6.1f} fps",
                flush=True,
            )

    proc.stdin.close()
    stderr_bytes = proc.stderr.read() if proc.stderr else b""
    ret = proc.wait()
    if ret != 0:
        raise RuntimeError(f"FFmpeg muxing failed (code {ret}):\n{stderr_bytes.decode(errors='ignore')}")

    total_time = time.perf_counter() - t0_render
    size_mb = os.path.getsize(mp4_path) / (1024.0 * 1024.0)
    print(f"[Export]   Master MP4 complete in {total_time:.2f}s ({total_frames/total_time:.1f} fps): {mp4_path} ({size_mb:.2f} MB)")

    # 4. Optional Palette-Optimized Preview GIF (highlighting the 4 acts in 640x360 @ 15fps)
    if export_gif:
        gif_path = os.path.join(out_abs, "preview.gif")
        gif_cmd = [
            ffmpeg_bin,
            "-y",
            "-ss", "20.5",
            "-t", "6.0",
            "-i", mp4_path,
            "-vf", "fps=15,scale=640:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=192[p];[s1][p]paletteuse=dither=bayer",
            gif_path,
        ]
        subprocess.run(gif_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        gif_mb = os.path.getsize(gif_path) / (1024.0 * 1024.0)
        print(f"[Export]   Animated GIF preview saved: {gif_path} ({gif_mb:.2f} MB)")

    return mp4_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CHRONOGENESIS Procedural Audiovisual Film Renderer")
    parser.add_argument("--preset", choices=list(QUALITY_PRESETS.keys()), default="high", help="Quality / performance preset")
    parser.add_argument("--backend", choices=["auto", "gpu", "cpu"], default="auto", help="Rendering backend")
    parser.add_argument("--out-dir", default="output", help="Output directory")
    parser.add_argument("--output-name", default="chronogenesis_1080p30.mp4", help="Output MP4 filename")
    parser.add_argument("--export-frames", action="store_true", help="Also write full lossless PNG sequence frame_%04d.png")
    parser.add_argument("--no-gif", action="store_true", help="Skip animated GIF preview generation")
    parser.add_argument("--keyframes-only", action="store_true", help="Render 4-act diagnostic keyframes only")
    parser.add_argument("--duration", type=float, default=None, help="Optional duration override in seconds")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.keyframes_only:
        render_keyframes_only(args.preset, args.backend, args.out_dir)
    else:
        run_full_production(
            preset_name=args.preset,
            backend=args.backend,
            out_dir=args.out_dir,
            mp4_name=args.output_name,
            export_frames=args.export_frames,
            export_gif=not args.no_gif,
            duration_override=args.duration,
        )
