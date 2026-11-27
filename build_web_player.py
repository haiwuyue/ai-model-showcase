"""
Bundles the GLSL shaders from `shaders/` into a standalone, zero-dependency offline
single-page WebGL2 + WebAudio real-time cinema player (`web/index.html`), including
interactive timeline scrubbing, live FPS budget auto-degradation, and WebM recording.
"""

from __future__ import annotations

import json
import os


def to_webgl2(src: str, is_vert: bool = False) -> str:
    lines = src.splitlines()
    out = ["#version 300 es", "precision highp float;", "precision highp int;"]
    for line in lines:
        if line.strip().startswith("#version"):
            continue
        out.append(line)
    return "\n".join(out)


def build_single_page_html(root_dir: str) -> str:
    shader_dir = os.path.join(root_dir, "shaders")
    with open(os.path.join(shader_dir, "scene_raymarch.frag"), "r", encoding="utf-8") as f:
        s_scene = to_webgl2(f.read())
    with open(os.path.join(shader_dir, "particles.vert"), "r", encoding="utf-8") as f:
        s_pvert = to_webgl2(f.read(), is_vert=True)
    with open(os.path.join(shader_dir, "particles.frag"), "r", encoding="utf-8") as f:
        s_pfrag = to_webgl2(f.read())
    with open(os.path.join(shader_dir, "bloom_extract.frag"), "r", encoding="utf-8") as f:
        s_ext = to_webgl2(f.read())
    with open(os.path.join(shader_dir, "bloom_blur.frag"), "r", encoding="utf-8") as f:
        s_blur = to_webgl2(f.read())
    with open(os.path.join(shader_dir, "composite_post.frag"), "r", encoding="utf-8") as f:
        s_post = to_webgl2(f.read())

    fs_vert = "#version 300 es\nprecision highp float;\nvoid main(){vec2 p=vec2((gl_VertexID<<1)&2,gl_VertexID&2);gl_Position=vec4(p*2.0-1.0,0.0,1.0);}"

    shaders_json = json.dumps(
        {
            "fs_vert": fs_vert,
            "scene_frag": s_scene,
            "part_vert": s_pvert,
            "part_frag": s_pfrag,
            "extract_frag": s_ext,
            "blur_frag": s_blur,
            "post_frag": s_post,
        }
    )

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>CHRONOGENESIS: 熵寂回响 — Offline Procedural WebGL2 + WebAudio Cinema</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: #03050a; color: #e2e8f0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    overflow: hidden; width: 100vw; height: 100vh; display: flex; align-items: center; justify-content: center;
  }}
  canvas {{ width: 100vw; height: 100vh; object-fit: contain; display: block; background: #000; }}
  #hud {{
    position: fixed; bottom: 20px; left: 50%; transform: translateX(-50%);
    background: rgba(8, 13, 26, 0.78); backdrop-filter: blur(12px);
    border: 1px solid rgba(148, 163, 184, 0.22); border-radius: 12px;
    padding: 12px 20px; display: flex; align-items: center; gap: 14px;
    font-size: 13px; z-index: 10; box-shadow: 0 12px 40px rgba(0,0,0,0.6);
  }}
  button, select {{
    background: rgba(30, 41, 59, 0.9); color: #f8fafc;
    border: 1px solid rgba(148, 163, 184, 0.35); border-radius: 6px;
    padding: 6px 12px; cursor: pointer; font-size: 12px; transition: 0.15s;
  }}
  button:hover, select:hover {{ border-color: #38bdf8; background: rgba(56, 189, 248, 0.18); }}
  #scrub {{ width: 220px; accent-color: #38bdf8; cursor: pointer; }}
  #actLabel {{ color: #7dd3fc; font-weight: 600; min-width: 195px; }}
  #stats {{ color: #94a3b8; font-variant-numeric: tabular-nums; min-width: 120px; }}
</style>
</head>
<body>
<canvas id="glcanvas" width="1920" height="1080"></canvas>
<div id="hud">
  <button id="playBtn">▶ 播放视听作品</button>
  <input id="scrub" type="range" min="0" max="40" step="0.033" value="0" />
  <span id="actLabel">起 · 虚空微澜 (00.0s)</span>
  <select id="presetSel">
    <option value="high" selected>High (1080p)</option>
    <option value="medium">Medium (720p)</option>
    <option value="low">Low (540p)</option>
  </select>
  <label style="display:flex;align-items:center;gap:4px;cursor:pointer;">
    <input type="checkbox" id="autoScale" checked /> 自动降级保帧
  </label>
  <button id="recBtn">● 录制 WebM</button>
  <span id="stats">Ready</span>
</div>

<script>
const SHADERS = {shaders_json};

const PRESETS = {{
  high:   {{ w: 1920, h: 1080, steps: [90, 28, 18, 4], particles: 12000, bloom: 3 }},
  medium: {{ w: 1280, h: 720,  steps: [68, 20, 12, 3], particles: 6000,  bloom: 2 }},
  low:    {{ w: 960,  h: 540,  steps: [48, 12, 8,  2], particles: 2500,  bloom: 1 }}
}};

const canvas = document.getElementById('glcanvas');
const gl = canvas.getContext('webgl2', {{ antialias: false, powerPreference: 'high-performance' }});
if (!gl) alert('需要支持 WebGL2 的现代浏览器');
const extFloat = gl.getExtension('EXT_color_buffer_float');
gl.getExtension('OES_texture_float_linear');

function compile(type, src, name) {{
  const s = gl.createShader(type);
  gl.shaderSource(s, src);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {{
    console.error(name, gl.getShaderInfoLog(s));
    throw new Error(name + ': ' + gl.getShaderInfoLog(s));
  }}
  return s;
}}
function link(vsSrc, fsSrc, name) {{
  const p = gl.createProgram();
  gl.attachShader(p, compile(gl.VERTEX_SHADER, vsSrc, name+'V'));
  gl.attachShader(p, compile(gl.FRAGMENT_SHADER, fsSrc, name+'F'));
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
  return p;
}}

const progScene   = link(SHADERS.fs_vert, SHADERS.scene_frag, 'scene');
const progPart    = link(SHADERS.part_vert, SHADERS.part_frag, 'part');
const progExtract = link(SHADERS.fs_vert, SHADERS.extract_frag, 'extract');
const progBlur    = link(SHADERS.fs_vert, SHADERS.blur_frag, 'blur');
const progPost    = link(SHADERS.fs_vert, SHADERS.post_frag, 'post');
const vao = gl.createVertexArray();
gl.bindVertexArray(vao);

let curPresetName = 'high';
let fbos = {{}};

function makeFBO(w, h, hdr) {{
  const tex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, tex);
  if (hdr && extFloat) {{
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA16F, w, h, 0, gl.RGBA, gl.HALF_FLOAT, null);
  }} else {{
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
  }}
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  const fbo = gl.createFramebuffer();
  gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
  return {{ fbo, tex, w, h }};
}}

function applyPreset(name) {{
  curPresetName = name;
  document.getElementById('presetSel').value = name;
  const p = PRESETS[name];
  canvas.width = p.w; canvas.height = p.h;
  const bw = p.w >> 1, bh = p.h >> 1;
  fbos.scene   = makeFBO(p.w, p.h, true);
  fbos.hdr     = makeFBO(p.w, p.h, true);
  fbos.bloomA  = makeFBO(bw, bh, true);
  fbos.bloomB  = makeFBO(bw, bh, true);
  fbos.streakA = makeFBO(bw, bh, true);
  fbos.streakB = makeFBO(bw, bh, true);
}}
applyPreset('high');

// Procedural WebAudio Score Engine (96 BPM, D Dorian, zero external files)
let audioCtx = null, analyser = null, freqData = null, masterDest = null;
let playing = false, startWall = 0, pauseTime = 4.5;

function m2f(m) {{ return 440 * Math.pow(2, (m - 69) / 12); }}

function scheduleProceduralScore(ctx, offsetSec) {{
  if (window._stopNodes) window._stopNodes.forEach(n => {{ try {{ n.stop(); }} catch(e){{}} }});
  window._stopNodes = [];
  const master = ctx.createGain();
  master.gain.value = 0.55;
  analyser = ctx.createAnalyser();
  analyser.fftSize = 256;
  freqData = new Uint8Array(analyser.frequencyBinCount);
  master.connect(analyser);
  analyser.connect(ctx.destination);
  if (masterDest) master.connect(masterDest);

  // Convolver impulse response generated mathematically for cathedral shimmer reverb
  const irLen = ctx.sampleRate * 2.8;
  const irBuf = ctx.createBuffer(2, irLen, ctx.sampleRate);
  for (let ch = 0; ch < 2; ch++) {{
    const d = irBuf.getChannelData(ch);
    for (let i = 0; i < irLen; i++) d[i] = (Math.random() * 2 - 1) * Math.pow(1 - i / irLen, 2.6) * 0.35;
  }}
  const conv = ctx.createConvolver(); conv.buffer = irBuf;
  const revGain = ctx.createGain(); revGain.gain.value = 0.42;
  conv.connect(revGain); revGain.connect(master);

  const BEAT = 0.625, BAR = 2.5;
  const chords = [
    {{ root: 38, pad: [50,57,60,64,69], arp: [62,65,69,72,76,74,69,65] }},
    {{ root: 34, pad: [46,53,58,62,69], arp: [62,65,69,74,76,72,69,65] }},
    {{ root: 31, pad: [43,50,55,58,65], arp: [58,62,67,70,74,70,67,62] }},
    {{ root: 33, pad: [45,52,57,61,67], arp: [57,61,64,69,73,76,69,64] }}
  ];
  const tBase = ctx.currentTime - offsetSec;

  for (let bar = 0; bar < 16; bar++) {{
    const bt = bar * BAR;
    if (bt + BAR < offsetSec) continue;
    const act = (bar >> 2) + 1;
    const ch = chords[bar % 4];
    const start = Math.max(ctx.currentTime, tBase + bt);

    // Sub & Reese Bass
    const osc = ctx.createOscillator();
    const g = ctx.createGain();
    osc.type = act >= 2 && act <= 3 ? 'sawtooth' : 'sine';
    osc.frequency.value = m2f(ch.root);
    const lp = ctx.createBiquadFilter();
    lp.type = 'lowpass'; lp.frequency.value = act === 3 ? 420 : 190;
    g.gain.setValueAtTime(0.001, start);
    g.gain.linearRampToValueAtTime(act === 3 ? 0.35 : 0.24, start + 0.08);
    g.gain.exponentialRampToValueAtTime(0.001, tBase + bt + BAR);
    osc.connect(lp); lp.connect(g); g.connect(master);
    osc.start(start); osc.stop(tBase + bt + BAR + 0.05);
    window._stopNodes.push(osc);

    // Lush Pad Voices
    ch.pad.forEach(n => {{
      const po = ctx.createOscillator(), pg = ctx.createGain(), pf = ctx.createBiquadFilter();
      po.type = 'triangle'; po.frequency.value = m2f(n);
      pf.type = 'lowpass'; pf.frequency.value = act === 3 ? 2800 : 1400;
      pg.gain.setValueAtTime(0.001, start);
      pg.gain.linearRampToValueAtTime(0.055, start + 0.45);
      pg.gain.exponentialRampToValueAtTime(0.001, tBase + bt + BAR + 0.3);
      po.connect(pf); pf.connect(pg); pg.connect(master); pg.connect(conv);
      po.start(start); po.stop(tBase + bt + BAR + 0.35);
      window._stopNodes.push(po);
    }});

    // Crystalline Arpeggio
    for (let s = 0; s < 16; s++) {{
      if (act === 1 && ![0,6,12].includes(s)) continue;
      if (act === 2 && s % 2 === 1 && s !== 7) continue;
      if (act === 4 && ![0,4,10].includes(s)) continue;
      const nt = tBase + bt + s * (BEAT * 0.25);
      if (nt < ctx.currentTime) continue;
      const ao = ctx.createOscillator(), ag = ctx.createGain();
      ao.type = 'sine';
      ao.frequency.value = m2f(ch.arp[s % 8] + (act === 3 && s % 4 === 2 ? 12 : 0));
      ag.gain.setValueAtTime(0.001, nt);
      ag.gain.linearRampToValueAtTime(act === 3 ? 0.11 : 0.07, nt + 0.01);
      ag.gain.exponentialRampToValueAtTime(0.001, nt + 0.35);
      ao.connect(ag); ag.connect(master); ag.connect(conv);
      ao.start(nt); ao.stop(nt + 0.38);
      window._stopNodes.push(ao);
    }}

    // Taiko Kick in Act II & III
    if (act === 2 || act === 3) {{
      const beats = act === 3 ? [0,1,2,3] : [0, 2.5];
      beats.forEach(b => {{
        const kt = tBase + bt + b * BEAT;
        if (kt < ctx.currentTime) return;
        const ko = ctx.createOscillator(), kg = ctx.createGain();
        ko.frequency.setValueAtTime(140, kt);
        ko.frequency.exponentialRampToValueAtTime(36, kt + 0.16);
        kg.gain.setValueAtTime(act === 3 ? 0.48 : 0.32, kt);
        kg.gain.exponentialRampToValueAtTime(0.001, kt + 0.38);
        ko.connect(kg); kg.connect(master);
        ko.start(kt); ko.stop(kt + 0.4);
        window._stopNodes.push(ko);
      }});
    }}
  }}
}}

function sampleAudio(t) {{
  if (playing && analyser && freqData) {{
    analyser.getByteFrequencyData(freqData);
    let sub = (freqData[1] + freqData[2] + freqData[3]) / (3 * 255);
    let mid = (freqData[8] + freqData[12] + freqData[16]) / (3 * 255);
    let high = (freqData[28] + freqData[40] + freqData[56]) / (3 * 255);
    let imp = Math.max(0, sub - 0.45) * 1.6;
    return [sub, mid, high, imp, 0.5 * (sub + mid)];
  }}
  // Analytical rhythmic fallback when scrubbing paused
  const beatPhase = (t / 0.625) % 1.0;
  const pulse = Math.exp(-beatPhase * 4.5);
  const act3 = (t > 20 && t < 30) ? 1.0 : 0.4;
  return [0.55 * pulse * act3 + 0.15, 0.45, 0.35 + 0.3 * Math.sin(t * 6.0), 0.4 * pulse * act3, 0.5];
}}

function renderGL(t) {{
  const p = PRESETS[curPresetName];
  const [sub, mid, high, imp, energy] = sampleAudio(t);

  // Pass 1: Raymarch
  gl.disable(gl.BLEND);
  gl.bindFramebuffer(gl.FRAMEBUFFER, fbos.scene.fbo);
  gl.viewport(0, 0, p.w, p.h);
  gl.useProgram(progScene);
  gl.uniform2f(gl.getUniformLocation(progScene, 'u_resolution'), p.w, p.h);
  gl.uniform1f(gl.getUniformLocation(progScene, 'u_time'), t);
  gl.uniform4f(gl.getUniformLocation(progScene, 'u_audio'), sub, mid, high, imp);
  gl.uniform1f(gl.getUniformLocation(progScene, 'u_energy'), energy);
  gl.uniform2f(gl.getUniformLocation(progScene, 'u_jitter'), 0, 0);
  gl.uniform4i(gl.getUniformLocation(progScene, 'u_quality'), ...p.steps);
  gl.drawArrays(gl.TRIANGLES, 0, 3);

  // Copy Scene -> HDR
  gl.bindFramebuffer(gl.FRAMEBUFFER, fbos.hdr.fbo);
  gl.useProgram(progBlur);
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, fbos.scene.tex);
  gl.uniform1i(gl.getUniformLocation(progBlur, 'u_tex'), 0);
  gl.uniform2f(gl.getUniformLocation(progBlur, 'u_direction'), 0, 0);
  gl.uniform3f(gl.getUniformLocation(progBlur, 'u_tint'), 1, 1, 1);
  gl.drawArrays(gl.TRIANGLES, 0, 3);

  // Pass 2: Curl-Noise Particles
  gl.enable(gl.BLEND);
  gl.blendFunc(gl.ONE, gl.ONE);
  gl.useProgram(progPart);
  gl.bindTexture(gl.TEXTURE_2D, fbos.scene.tex);
  gl.uniform1i(gl.getUniformLocation(progPart, 'u_sceneTex'), 0);
  gl.uniform2f(gl.getUniformLocation(progPart, 'u_resolution'), p.w, p.h);
  gl.uniform1f(gl.getUniformLocation(progPart, 'u_time'), t);
  gl.uniform4f(gl.getUniformLocation(progPart, 'u_audio'), sub, mid, high, imp);
  gl.uniform1i(gl.getUniformLocation(progPart, 'u_particle_count'), p.particles);
  gl.drawArraysInstanced(gl.TRIANGLES, 0, 6, p.particles);
  gl.disable(gl.BLEND);

  // Pass 3: Bloom & Anamorphic Streaks
  const bw = fbos.bloomA.w, bh = fbos.bloomA.h;
  gl.bindFramebuffer(gl.FRAMEBUFFER, fbos.bloomA.fbo);
  gl.viewport(0, 0, bw, bh);
  gl.useProgram(progExtract);
  gl.bindTexture(gl.TEXTURE_2D, fbos.hdr.tex);
  gl.uniform1i(gl.getUniformLocation(progExtract, 'u_hdrTex'), 0);
  gl.uniform2f(gl.getUniformLocation(progExtract, 'u_texelSize'), 1 / bw, 1 / bh);
  gl.uniform1f(gl.getUniformLocation(progExtract, 'u_threshold'), 0.72);
  gl.drawArrays(gl.TRIANGLES, 0, 3);
  gl.bindFramebuffer(gl.FRAMEBUFFER, fbos.streakA.fbo);
  gl.drawArrays(gl.TRIANGLES, 0, 3);

  gl.useProgram(progBlur);
  for (let i = 0; i < p.bloom; i++) {{
    const r = 1.0 + i * 1.45;
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbos.bloomB.fbo);
    gl.bindTexture(gl.TEXTURE_2D, fbos.bloomA.tex);
    gl.uniform2f(gl.getUniformLocation(progBlur, 'u_direction'), r / bw, 0);
    gl.uniform3f(gl.getUniformLocation(progBlur, 'u_tint'), 1, 1, 1);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbos.bloomA.fbo);
    gl.bindTexture(gl.TEXTURE_2D, fbos.bloomB.tex);
    gl.uniform2f(gl.getUniformLocation(progBlur, 'u_direction'), 0, r / bh);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }}
  for (let i = 0; i < 3; i++) {{
    const span = Math.pow(3.5, i + 1) / bw;
    gl.bindFramebuffer(gl.FRAMEBUFFER, i % 2 === 0 ? fbos.streakB.fbo : fbos.streakA.fbo);
    gl.bindTexture(gl.TEXTURE_2D, i % 2 === 0 ? fbos.streakA.tex : fbos.streakB.tex);
    gl.uniform2f(gl.getUniformLocation(progBlur, 'u_direction'), span, 0);
    gl.uniform3f(gl.getUniformLocation(progBlur, 'u_tint'), ...(i === 0 ? [0.72, 0.88, 1.22] : [1, 1, 1]));
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }}

  // Pass 4: Composite to Screen
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gl.viewport(0, 0, p.w, p.h);
  gl.useProgram(progPost);
  gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, fbos.hdr.tex);
  gl.uniform1i(gl.getUniformLocation(progPost, 'u_hdrTex'), 0);
  gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, fbos.bloomA.tex);
  gl.uniform1i(gl.getUniformLocation(progPost, 'u_bloomTex'), 1);
  gl.activeTexture(gl.TEXTURE2); gl.bindTexture(gl.TEXTURE_2D, fbos.streakB.tex);
  gl.uniform1i(gl.getUniformLocation(progPost, 'u_streakTex'), 2);
  gl.uniform2f(gl.getUniformLocation(progPost, 'u_resolution'), p.w, p.h);
  gl.uniform1f(gl.getUniformLocation(progPost, 'u_time'), t);
  gl.uniform4f(gl.getUniformLocation(progPost, 'u_audio'), sub, mid, high, imp);
  gl.drawArrays(gl.TRIANGLES, 0, 3);
}}

// Interactive Controls & Adaptive FPS Budget Monitor
const playBtn = document.getElementById('playBtn');
const scrub = document.getElementById('scrub');
const actLabel = document.getElementById('actLabel');
const statsEl = document.getElementById('stats');
let fpsFrames = 0, fpsLast = performance.now(), lowFpsStreak = 0;

function actName(t) {{
  if (t < 10) return '起 · 虚空微澜';
  if (t < 20) return '承 · 秩序织造';
  if (t < 30) return '转 · 奇点共振';
  return '合 · 负熵归寂';
}}

playBtn.onclick = () => {{
  if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  if (audioCtx.state === 'suspended') audioCtx.resume();
  if (!playing) {{
    if (pauseTime >= 39.9) pauseTime = 0;
    startWall = performance.now() / 1000 - pauseTime;
    scheduleProceduralScore(audioCtx, pauseTime);
    playing = true;
    playBtn.textContent = '⏸ 暂停';
  }} else {{
    pauseTime = performance.now() / 1000 - startWall;
    if (window._stopNodes) window._stopNodes.forEach(n => {{ try {{ n.stop(); }} catch(e){{}} }});
    playing = false;
    playBtn.textContent = '▶ 继续';
  }}
}};

scrub.oninput = (e) => {{
  pauseTime = parseFloat(e.target.value);
  if (playing) {{
    startWall = performance.now() / 1000 - pauseTime;
    scheduleProceduralScore(audioCtx, pauseTime);
  }}
}};

document.getElementById('presetSel').onchange = (e) => applyPreset(e.target.value);

// WebM In-Browser Recorder
let recorder = null, chunks = [];
document.getElementById('recBtn').onclick = (e) => {{
  if (recorder && recorder.state === 'recording') {{
    recorder.stop();
    e.target.textContent = '● 录制 WebM';
    return;
  }}
  if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  masterDest = audioCtx.createMediaStreamDestination();
  const stream = canvas.captureStream(30);
  masterDest.stream.getAudioTracks().forEach(tr => stream.addTrack(tr));
  chunks = [];
  recorder = new MediaRecorder(stream, {{ mimeType: 'video/webm;codecs=vp9', videoBitsPerSecond: 12000000 }});
  recorder.ondataavailable = ev => {{ if (ev.data.size > 0) chunks.push(ev.data); }};
  recorder.onstop = () => {{
    const url = URL.createObjectURL(new Blob(chunks, {{ type: 'video/webm' }}));
    const a = document.createElement('a'); a.href = url; a.download = 'chronogenesis_webgl.webm'; a.click();
  }};
  pauseTime = 0;
  if (!playing) playBtn.click(); else {{ startWall = performance.now() / 1000; scheduleProceduralScore(audioCtx, 0); }}
  recorder.start();
  e.target.textContent = '■ 停止并导出';
}};

function tick(now) {{
  let t = playing ? (now / 1000 - startWall) : pauseTime;
  if (playing && t >= 40.0) {{
    t = 40.0; pauseTime = 0; playing = false; playBtn.textContent = '▶ 重播';
    if (recorder && recorder.state === 'recording') recorder.stop();
  }}
  scrub.value = t;
  actLabel.textContent = `${{actName(t)}} (${{t.toFixed(1)}}s)`;
  renderGL(t);

  fpsFrames++;
  if (now - fpsLast >= 1000) {{
    const fps = (fpsFrames * 1000) / (now - fpsLast);
    const p = PRESETS[curPresetName];
    statsEl.textContent = `${{p.w}}×${{p.h}} | ${{fps.toFixed(0)}} FPS`;
    if (document.getElementById('autoScale').checked && playing && fps < 24) {{
      lowFpsStreak++;
      if (lowFpsStreak >= 2) {{
        if (curPresetName === 'high') applyPreset('medium');
        else if (curPresetName === 'medium') applyPreset('low');
        lowFpsStreak = 0;
      }}
    }} else lowFpsStreak = 0;
    fpsFrames = 0; fpsLast = now;
  }}
  requestAnimationFrame(tick);
}}
requestAnimationFrame(tick);
</script>
</body>
</html>
"""
    web_dir = os.path.join(root_dir, "web")
    os.makedirs(web_dir, exist_ok=True)
    out_html = os.path.join(web_dir, "index.html")
    with open(out_html, "w", encoding="utf-8") as fh:
        fh.write(html)
    return out_html


if __name__ == "__main__":
    path = build_single_page_html(os.path.dirname(os.path.abspath(__file__)))
    print(f"[WebPlayer] Built standalone offline WebGL2 player: {path}")
