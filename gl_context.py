"""
Hardware-accelerated Offscreen OpenGL 3.3+/4.6 Multi-Pass HDR Renderer via pure Python ctypes (Win32 WGL).
Requires zero third-party OpenGL wrapper packages on Windows; also supports optional `moderngl` if installed
on Linux/macOS, and automatically falls back to `cpu_fallback.py` if no GPU context is available.
"""

from __future__ import annotations

import ctypes
import ctypes.wintypes as wt
import os
import sys
from typing import Dict, Tuple

import numpy as np

from config import QualityPreset

# OpenGL Constants
GL_TRIANGLES = 0x0004
GL_TEXTURE_2D = 0x0DE1
GL_TEXTURE0 = 0x84C0
GL_RGBA = 0x1908
GL_RGB = 0x1907
GL_RGBA16F = 0x881A
GL_RGBA8 = 0x8058
GL_UNSIGNED_BYTE = 0x1401
GL_FLOAT = 0x1406
GL_TEXTURE_MIN_FILTER = 0x2801
GL_TEXTURE_MAG_FILTER = 0x2800
GL_TEXTURE_WRAP_S = 0x2802
GL_TEXTURE_WRAP_T = 0x2803
GL_LINEAR = 0x2601
GL_CLAMP_TO_EDGE = 0x812F
GL_FRAMEBUFFER = 0x8D40
GL_COLOR_ATTACHMENT0 = 0x8CE0
GL_FRAMEBUFFER_COMPLETE = 0x8CD5
GL_VERTEX_SHADER = 0x8B31
GL_FRAGMENT_SHADER = 0x8B30
GL_COMPILE_STATUS = 0x8B81
GL_LINK_STATUS = 0x8B82
GL_INFO_LOG_LENGTH = 0x8B84
GL_COLOR_BUFFER_BIT = 0x00004000
GL_BLEND = 0x0BE2
GL_ONE = 1
GL_FUNC_ADD = 0x8006
GL_PACK_ALIGNMENT = 0x0D05

FULLSCREEN_VERT_SRC = """#version 330 core
void main() {
    vec2 pos = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    gl_Position = vec4(pos * 2.0 - 1.0, 0.0, 1.0);
}
"""


class PFD(ctypes.Structure):
    _fields_ = [
        ("nSize", wt.WORD),
        ("nVersion", wt.WORD),
        ("dwFlags", wt.DWORD),
        ("iPixelType", ctypes.c_byte),
        ("cColorBits", ctypes.c_byte),
        ("pad", ctypes.c_byte * 31),
    ]


class WGLOffscreenRenderer:
    """Multi-pass HDR OpenGL 4.6 pipeline: Raymarch -> GPU Particles -> Karis Bloom & Streaks -> ACES Post."""

    def __init__(self, preset: QualityPreset, shader_dir: str) -> None:
        if sys.platform != "win32":
            raise RuntimeError("WGL backend is Windows-only; use CPU fallback or moderngl on non-Windows.")

        self.preset = preset
        self.w = preset.width
        self.h = preset.height
        # Internal supersampled resolution for Scene + Particles
        self.ss = preset.supersample
        self.sw = self.w * self.ss
        self.sh = self.h * self.ss
        self.shader_dir = shader_dir

        self._init_wgl_context()
        self._bind_gl_functions()
        self._init_pipeline()

    def _init_wgl_context(self) -> None:
        VP = ctypes.c_void_p
        self._u32 = ctypes.windll.user32
        self._g32 = ctypes.windll.gdi32
        self._gl = ctypes.windll.opengl32

        self._u32.CreateWindowExW.restype = VP
        self._u32.GetDC.argtypes, self._u32.GetDC.restype = [VP], VP
        self._gl.wglCreateContext.argtypes, self._gl.wglCreateContext.restype = [VP], VP
        self._gl.wglMakeCurrent.argtypes, self._gl.wglMakeCurrent.restype = [VP, VP], wt.BOOL
        self._gl.wglGetProcAddress.argtypes, self._gl.wglGetProcAddress.restype = [ctypes.c_char_p], VP
        self._g32.ChoosePixelFormat.argtypes = [VP, ctypes.POINTER(PFD)]
        self._g32.SetPixelFormat.argtypes = [VP, ctypes.c_int, ctypes.POINTER(PFD)]

        self._hwnd = self._u32.CreateWindowExW(0, "STATIC", "ChronogenesisGL", 0, 0, 0, 32, 32, 0, 0, 0, 0)
        if not self._hwnd:
            raise RuntimeError("Failed to create hidden Win32 window for OpenGL.")
        self._hdc = self._u32.GetDC(self._hwnd)
        pfd = PFD(ctypes.sizeof(PFD), 1, 4 | 32 | 1, 0, 32)
        pf = self._g32.ChoosePixelFormat(self._hdc, ctypes.byref(pfd))
        self._g32.SetPixelFormat(self._hdc, pf, ctypes.byref(pfd))
        self._hrc = self._gl.wglCreateContext(self._hdc)
        if not self._hrc or not self._gl.wglMakeCurrent(self._hdc, self._hrc):
            raise RuntimeError("Failed to activate WGL OpenGL context.")

        self._gl.glGetString.argtypes = [ctypes.c_uint]
        self._gl.glGetString.restype = ctypes.c_char_p
        renderer_b = self._gl.glGetString(0x1F01) or b"Unknown GPU"
        version_b = self._gl.glGetString(0x1F02) or b"Unknown GL"
        self.gpu_name = f"{renderer_b.decode(errors='ignore')} ({version_b.decode(errors='ignore')})"

    def _fn(self, name: str, restype, argtypes):
        ptr = self._gl.wglGetProcAddress(name.encode("ascii"))
        if ptr:
            proto = ctypes.WINFUNCTYPE(restype, *argtypes)
            return proto(ptr)
        func = getattr(self._gl, name)
        func.restype = restype
        func.argtypes = argtypes
        return func

    def _bind_gl_functions(self) -> None:
        u, i, f, vp, cp = ctypes.c_uint, ctypes.c_int, ctypes.c_float, ctypes.c_void_p, ctypes.c_char_p
        pu = ctypes.POINTER(u)
        pi = ctypes.POINTER(i)

        self.glViewport = self._fn("glViewport", None, [i, i, i, i])
        self.glClearColor = self._fn("glClearColor", None, [f, f, f, f])
        self.glClear = self._fn("glClear", None, [u])
        self.glEnable = self._fn("glEnable", None, [u])
        self.glDisable = self._fn("glDisable", None, [u])
        self.glBlendFunc = self._fn("glBlendFunc", None, [u, u])
        self.glBlendEquation = self._fn("glBlendEquation", None, [u])
        self.glPixelStorei = self._fn("glPixelStorei", None, [u, i])
        self.glReadPixels = self._fn("glReadPixels", None, [i, i, i, i, u, u, vp])
        self.glFinish = self._fn("glFinish", None, [])

        self.glGenTextures = self._fn("glGenTextures", None, [i, pu])
        self.glBindTexture = self._fn("glBindTexture", None, [u, u])
        self.glTexImage2D = self._fn("glTexImage2D", None, [u, i, i, i, i, i, u, u, vp])
        self.glTexParameteri = self._fn("glTexParameteri", None, [u, u, i])
        self.glActiveTexture = self._fn("glActiveTexture", None, [u])

        self.glGenFramebuffers = self._fn("glGenFramebuffers", None, [i, pu])
        self.glBindFramebuffer = self._fn("glBindFramebuffer", None, [u, u])
        self.glFramebufferTexture2D = self._fn("glFramebufferTexture2D", None, [u, u, u, u, i])
        self.glCheckFramebufferStatus = self._fn("glCheckFramebufferStatus", u, [u])

        self.glGenVertexArrays = self._fn("glGenVertexArrays", None, [i, pu])
        self.glBindVertexArray = self._fn("glBindVertexArray", None, [u])
        self.glDrawArrays = self._fn("glDrawArrays", None, [u, i, i])
        self.glDrawArraysInstanced = self._fn("glDrawArraysInstanced", None, [u, i, i, i])

        self.glCreateShader = self._fn("glCreateShader", u, [u])
        self.glShaderSource = self._fn("glShaderSource", None, [u, i, ctypes.POINTER(cp), pi])
        self.glCompileShader = self._fn("glCompileShader", None, [u])
        self.glGetShaderiv = self._fn("glGetShaderiv", None, [u, u, pi])
        self.glGetShaderInfoLog = self._fn("glGetShaderInfoLog", None, [u, i, pi, cp])
        self.glCreateProgram = self._fn("glCreateProgram", u, [])
        self.glAttachShader = self._fn("glAttachShader", None, [u, u])
        self.glLinkProgram = self._fn("glLinkProgram", None, [u])
        self.glGetProgramiv = self._fn("glGetProgramiv", None, [u, u, pi])
        self.glGetProgramInfoLog = self._fn("glGetProgramInfoLog", None, [u, i, pi, cp])
        self.glUseProgram = self._fn("glUseProgram", None, [u])

        self.glGetUniformLocation = self._fn("glGetUniformLocation", i, [u, cp])
        self.glUniform1i = self._fn("glUniform1i", None, [i, i])
        self.glUniform1f = self._fn("glUniform1f", None, [i, f])
        self.glUniform2f = self._fn("glUniform2f", None, [i, f, f])
        self.glUniform3f = self._fn("glUniform3f", None, [i, f, f, f])
        self.glUniform4f = self._fn("glUniform4f", None, [i, f, f, f, f])
        self.glUniform4i = self._fn("glUniform4i", None, [i, i, i, i, i])

    def _compile_shader(self, src: str, stage: int, label: str) -> int:
        sh = self.glCreateShader(stage)
        src_bytes = src.encode("utf-8")
        c_src = ctypes.c_char_p(src_bytes)
        length = ctypes.c_int(len(src_bytes))
        self.glShaderSource(sh, 1, ctypes.byref(c_src), ctypes.byref(length))
        self.glCompileShader(sh)
        ok = ctypes.c_int(0)
        self.glGetShaderiv(sh, GL_COMPILE_STATUS, ctypes.byref(ok))
        if not ok.value:
            log_len = ctypes.c_int(0)
            self.glGetShaderiv(sh, GL_INFO_LOG_LENGTH, ctypes.byref(log_len))
            buf = ctypes.create_string_buffer(max(log_len.value, 512))
            self.glGetShaderInfoLog(sh, len(buf), None, buf)
            raise RuntimeError(f"Shader compile error [{label}]:\n{buf.value.decode(errors='ignore')}")
        return sh

    def _build_program(self, vert_src: str, frag_src: str, label: str) -> int:
        vs = self._compile_shader(vert_src, GL_VERTEX_SHADER, f"{label}.vert")
        fs = self._compile_shader(frag_src, GL_FRAGMENT_SHADER, f"{label}.frag")
        prog = self.glCreateProgram()
        self.glAttachShader(prog, vs)
        self.glAttachShader(prog, fs)
        self.glLinkProgram(prog)
        ok = ctypes.c_int(0)
        self.glGetProgramiv(prog, GL_LINK_STATUS, ctypes.byref(ok))
        if not ok.value:
            buf = ctypes.create_string_buffer(2048)
            self.glGetProgramInfoLog(prog, 2048, None, buf)
            raise RuntimeError(f"Program link error [{label}]:\n{buf.value.decode(errors='ignore')}")
        return prog

    def _create_fbo(self, w: int, h: int, hdr: bool = True) -> Tuple[int, int]:
        tex = ctypes.c_uint(0)
        self.glGenTextures(1, ctypes.byref(tex))
        self.glBindTexture(GL_TEXTURE_2D, tex.value)
        internal_fmt = GL_RGBA16F if hdr else GL_RGBA8
        pix_type = GL_FLOAT if hdr else GL_UNSIGNED_BYTE
        self.glTexImage2D(GL_TEXTURE_2D, 0, internal_fmt, w, h, 0, GL_RGBA, pix_type, None)
        self.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        self.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        self.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        self.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

        fbo = ctypes.c_uint(0)
        self.glGenFramebuffers(1, ctypes.byref(fbo))
        self.glBindFramebuffer(GL_FRAMEBUFFER, fbo.value)
        self.glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex.value, 0)
        status = self.glCheckFramebufferStatus(GL_FRAMEBUFFER)
        if status != GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError(f"Incomplete framebuffer: status=0x{status:X}")
        return fbo.value, tex.value

    def _read_shader(self, filename: str) -> str:
        with open(os.path.join(self.shader_dir, filename), "r", encoding="utf-8") as fh:
            return fh.read()

    def _init_pipeline(self) -> None:
        vao = ctypes.c_uint(0)
        self.glGenVertexArrays(1, ctypes.byref(vao))
        self.glBindVertexArray(vao.value)
        self._vao = vao.value
        self.glPixelStorei(GL_PACK_ALIGNMENT, 1)

        # Compile shader programs
        self.prog_scene = self._build_program(FULLSCREEN_VERT_SRC, self._read_shader("scene_raymarch.frag"), "scene")
        self.prog_part = self._build_program(
            self._read_shader("particles.vert"), self._read_shader("particles.frag"), "particles"
        )
        self.prog_extract = self._build_program(
            FULLSCREEN_VERT_SRC, self._read_shader("bloom_extract.frag"), "bloom_extract"
        )
        self.glUseProgram(self.prog_extract)
        self.prog_blur = self._build_program(FULLSCREEN_VERT_SRC, self._read_shader("bloom_blur.frag"), "bloom_blur")
        self.prog_post = self._build_program(
            FULLSCREEN_VERT_SRC, self._read_shader("composite_post.frag"), "composite_post"
        )

        # Allocate FBOs:
        # 1. Primary raymarch HDR + linear depth (.a) at supersampled resolution
        self.fbo_scene, self.tex_scene = self._create_fbo(self.sw, self.sh, hdr=True)
        # 2. Combined HDR buffer (Scene + Additive GPU Particles)
        self.fbo_hdr, self.tex_hdr = self._create_fbo(self.sw, self.sh, hdr=True)
        # 3. Half-res & Quarter-res Bloom + Anamorphic Horizontal Streak FBOs
        self.bw, self.bh = max(1, self.w // 2), max(1, self.h // 2)
        self.fbo_bloom_a, self.tex_bloom_a = self._create_fbo(self.bw, self.bh, hdr=True)
        self.fbo_bloom_b, self.tex_bloom_b = self._create_fbo(self.bw, self.bh, hdr=True)
        self.fbo_streak_a, self.tex_streak_a = self._create_fbo(self.bw, self.bh, hdr=True)
        self.fbo_streak_b, self.tex_streak_b = self._create_fbo(self.bw, self.bh, hdr=True)
        # 4. Final 8-bit sRGB output FBO at target output resolution (1920x1080)
        self.fbo_out, self.tex_out = self._create_fbo(self.w, self.h, hdr=False)

        # Pre-allocate pixel readback buffer (RGB24)
        self._rgb_buf = (ctypes.c_ubyte * (self.w * self.h * 3))()

    def _u(self, prog: int, name: str) -> int:
        return self.glGetUniformLocation(prog, name.encode("ascii"))

    def render_frame(self, t: float, audio_vec: Tuple[float, float, float, float], energy: float) -> np.ndarray:
        """
        Renders a single frame at time `t` and returns bottom-to-top raw RGB uint8 array (H, W, 3).
        """
        sub, mid, high, impact = audio_vec
        p = self.preset

        # ---------------------------------------------------------------------
        # PASS 1: Primary Raymarched Scene + Volumetrics -> fbo_scene (RGBA16F)
        # ---------------------------------------------------------------------
        self.glDisable(GL_BLEND)
        self.glBindFramebuffer(GL_FRAMEBUFFER, self.fbo_scene)
        self.glViewport(0, 0, self.sw, self.sh)
        self.glUseProgram(self.prog_scene)
        self.glUniform2f(self._u(self.prog_scene, "u_resolution"), float(self.sw), float(self.sh))
        self.glUniform1f(self._u(self.prog_scene, "u_time"), float(t))
        self.glUniform4f(self._u(self.prog_scene, "u_audio"), sub, mid, high, impact)
        self.glUniform1f(self._u(self.prog_scene, "u_energy"), float(energy))
        self.glUniform2f(self._u(self.prog_scene, "u_jitter"), 0.0, 0.0)
        self.glUniform4i(
            self._u(self.prog_scene, "u_quality"),
            p.max_march_steps,
            p.volumetric_steps,
            p.shadow_steps,
            p.ao_samples,
        )
        self.glDrawArrays(GL_TRIANGLES, 0, 3)

        # Copy fbo_scene into fbo_hdr via Pass 2 base + additive instanced curl-noise particles
        # (so particles can read tex_scene.a depth without feedback loop!)
        self.glBindFramebuffer(GL_FRAMEBUFFER, self.fbo_hdr)
        self.glViewport(0, 0, self.sw, self.sh)
        # Blit scene RGB into fbo_hdr using bloom_blur with zero offset & tint=1
        self.glUseProgram(self.prog_blur)
        self.glActiveTexture(GL_TEXTURE0)
        self.glBindTexture(GL_TEXTURE_2D, self.tex_scene)
        self.glUniform1i(self._u(self.prog_blur, "u_tex"), 0)
        self.glUniform2f(self._u(self.prog_blur, "u_direction"), 0.0, 0.0)
        self.glUniform3f(self._u(self.prog_blur, "u_tint"), 1.0, 1.0, 1.0)
        self.glDrawArrays(GL_TRIANGLES, 0, 3)

        # ---------------------------------------------------------------------
        # PASS 2: Instanced GPU Curl-Noise Particles (Additive HDR, Soft Depth Occluded)
        # ---------------------------------------------------------------------
        if p.particle_count > 0:
            self.glEnable(GL_BLEND)
            self.glBlendEquation(GL_FUNC_ADD)
            self.glBlendFunc(GL_ONE, GL_ONE)
            self.glUseProgram(self.prog_part)
            self.glActiveTexture(GL_TEXTURE0)
            self.glBindTexture(GL_TEXTURE_2D, self.tex_scene)
            self.glUniform1i(self._u(self.prog_part, "u_sceneTex"), 0)
            self.glUniform2f(self._u(self.prog_part, "u_resolution"), float(self.sw), float(self.sh))
            self.glUniform1f(self._u(self.prog_part, "u_time"), float(t))
            self.glUniform4f(self._u(self.prog_part, "u_audio"), sub, mid, high, impact)
            self.glUniform1i(self._u(self.prog_part, "u_particle_count"), p.particle_count)
            self.glDrawArraysInstanced(GL_TRIANGLES, 0, 6, p.particle_count)
            self.glDisable(GL_BLEND)

        # ---------------------------------------------------------------------
        # PASS 3A: HDR Bright-Pass Extraction (Karis Firefly Filter) -> fbo_bloom_a
        # ---------------------------------------------------------------------
        self.glBindFramebuffer(GL_FRAMEBUFFER, self.fbo_bloom_a)
        self.glViewport(0, 0, self.bw, self.bh)
        self.glUseProgram(self.prog_extract)
        self.glActiveTexture(GL_TEXTURE0)
        self.glBindTexture(GL_TEXTURE_2D, self.tex_hdr)
        self.glUniform1i(self._u(self.prog_extract, "u_hdrTex"), 0)
        self.glUniform2f(self._u(self.prog_extract, "u_texelSize"), 1.0 / self.bw, 1.0 / self.bh)
        self.glUniform1f(self._u(self.prog_extract, "u_threshold"), 0.72)
        self.glDrawArrays(GL_TRIANGLES, 0, 3)

        # Copy extracted brights to streak source as well
        self.glBindFramebuffer(GL_FRAMEBUFFER, self.fbo_streak_a)
        self.glDrawArrays(GL_TRIANGLES, 0, 3)

        # ---------------------------------------------------------------------
        # PASS 3B: Multi-Pass Gaussian Bloom + Anamorphic Cinema Horizontal Streaks
        # ---------------------------------------------------------------------
        self.glUseProgram(self.prog_blur)
        self.glUniform1i(self._u(self.prog_blur, "u_tex"), 0)

        for i in range(p.bloom_passes):
            radius = 1.0 + i * 1.45
            # Horizontal
            self.glBindFramebuffer(GL_FRAMEBUFFER, self.fbo_bloom_b)
            self.glBindTexture(GL_TEXTURE_2D, self.tex_bloom_a)
            self.glUniform2f(self._u(self.prog_blur, "u_direction"), radius / self.bw, 0.0)
            self.glUniform3f(self._u(self.prog_blur, "u_tint"), 1.0, 1.0, 1.0)
            self.glDrawArrays(GL_TRIANGLES, 0, 3)
            # Vertical
            self.glBindFramebuffer(GL_FRAMEBUFFER, self.fbo_bloom_a)
            self.glBindTexture(GL_TEXTURE_2D, self.tex_bloom_b)
            self.glUniform2f(self._u(self.prog_blur, "u_direction"), 0.0, radius / self.bh)
            self.glDrawArrays(GL_TRIANGLES, 0, 3)

        # Anamorphic horizontal wide lens streak (cool cyan-sapphire cinema tint)
        for i in range(3):
            span = (3.5 ** (i + 1)) / self.bw
            dst_fbo = self.fbo_streak_b if i % 2 == 0 else self.fbo_streak_a
            src_tex = self.tex_streak_a if i % 2 == 0 else self.tex_streak_b
            self.glBindFramebuffer(GL_FRAMEBUFFER, dst_fbo)
            self.glBindTexture(GL_TEXTURE_2D, src_tex)
            self.glUniform2f(self._u(self.prog_blur, "u_direction"), span, 0.0)
            tint = (0.72, 0.88, 1.22) if i == 0 else (1.0, 1.0, 1.0)
            self.glUniform3f(self._u(self.prog_blur, "u_tint"), *tint)
            self.glDrawArrays(GL_TRIANGLES, 0, 3)
        final_streak_tex = self.tex_streak_b  # 3 iterations (0->B, 1->A, 2->B)

        # ---------------------------------------------------------------------
        # PASS 4: Final Composite (Chromatic Dispersion + Bloom + ACES + Grain)
        # ---------------------------------------------------------------------
        self.glBindFramebuffer(GL_FRAMEBUFFER, self.fbo_out)
        self.glViewport(0, 0, self.w, self.h)
        self.glUseProgram(self.prog_post)

        self.glActiveTexture(GL_TEXTURE0)
        self.glBindTexture(GL_TEXTURE_2D, self.tex_hdr)
        self.glUniform1i(self._u(self.prog_post, "u_hdrTex"), 0)

        self.glActiveTexture(GL_TEXTURE0 + 1)
        self.glBindTexture(GL_TEXTURE_2D, self.tex_bloom_a)
        self.glUniform1i(self._u(self.prog_post, "u_bloomTex"), 1)

        self.glActiveTexture(GL_TEXTURE0 + 2)
        self.glBindTexture(GL_TEXTURE_2D, final_streak_tex)
        self.glUniform1i(self._u(self.prog_post, "u_streakTex"), 2)

        self.glUniform2f(self._u(self.prog_post, "u_resolution"), float(self.w), float(self.h))
        self.glUniform1f(self._u(self.prog_post, "u_time"), float(t))
        self.glUniform4f(self._u(self.prog_post, "u_audio"), sub, mid, high, impact)
        self.glDrawArrays(GL_TRIANGLES, 0, 3)

        self.glReadPixels(0, 0, self.w, self.h, GL_RGB, GL_UNSIGNED_BYTE, self._rgb_buf)
        arr = np.frombuffer(self._rgb_buf, dtype=np.uint8).reshape((self.h, self.w, 3))
        return arr
