# 《CHRONOGENESIS: 熵寂回响》(Echoes of Negative Entropy)

纯代码程序化视听短片项目（**零外部图片、零视频、零音频采样、零生成式 AI API**）。  
分辨率 **1920×1080 @ 30fps**，时长 **40.0 秒（1,200 帧，96 BPM 严格 16 小节四幕对位）**，立体声 **48kHz PCM/AAC** 程序化配乐。

---

## 目录结构

```text
zhihu/
├── config.py                     # 96BPM 四幕时间轴、色彩语言与四档性能预算/降级预设
├── audio_synth.py                # 纯数学多轨立体声合成器（FM 晶铃/Reese 贝斯/共振峰唱诗班/FDN 混响）与包络提取
├── gl_context.py                 # 零第三方包装库依赖的 Win32 WGL OpenGL 4.6 多通道 HDR 离屏渲染管线
├── cpu_fallback.py               # 无 GPU 环境自动降级的 NumPy/SciPy 矢量化光线步进后端
├── render.py                     # 生产级渲染主入口（合成 WAV、渲染 1080p30 MP4、导出 PNG 帧序列与 GIF）
├── build_web_player.py           # 将 shaders/ 编译打包为离线单文件 WebGL2 + WebAudio 实时播放器
├── requirements.txt              # Python 依赖说明
├── shaders/
│   ├── scene_raymarch.frag       # Pass 1: 四幕混合 Kleinian-Mandelbox SDF + 爱因斯坦引力透镜 + 多普勒吸积盘 + 丁达尔体积光 + 薄膜干涉 PBR
│   ├── particles.vert            # Pass 2: 无散旋度场（Curl-Noise）平流生物发光流丝与浅景深散景粒子（顶点实例化）
│   ├── particles.frag            # Pass 2: 软深度遮挡（读取 Pass 1 线性深度）与高光能量核
│   ├── bloom_extract.frag        # Pass 3A: HDR 亮度阈值提取 + Karis 平均抑制亚像素噪点闪烁
│   ├── bloom_blur.frag           # Pass 3B: 可分离高斯泛光 + 变形宽银幕水平眩光（Anamorphic Streak）
│   └── composite_post.frag       # Pass 4: 径向色散（Chromatic Dispersion）+ ACES Filmic 色调映射 + 暗角与胶片颗粒
├── web/
│   └── index.html                # 独立零依赖单页浏览器实时版（双击离线运行，支持自动降级保帧与 WebM 录制）
└── output/
    ├── chronogenesis_1080p30.mp4 # 最终导出的 1920×1080 @ 30fps H.264 + AAC 成片
    ├── soundtrack.wav            # 48kHz 16-bit 立体声程序化原声母带
    ├── envelopes.npz             # 逐帧提取的 5 维音频驱动曲线（sub/mid/high/impact/energy）
    ├── preview.gif               # 调色板优化的动态预览图
    ├── keyframes/                # 四幕 1080p 无损诊断关键帧
    └── frames/                   # 可选完整无损帧序列 frame_0000.png .. frame_1199.png
```

---

## 环境要求与快速运行

### 1. Python + OpenGL + FFmpeg 离线高画质母带渲染

- **系统环境**：Python 3.10+，`ffmpeg`（位于系统 `PATH` 中）
- **Python 依赖**：
  ```bash
  pip install -r requirements.txt
  ```
- **一键生成完整 1920×1080 @ 30fps 视频与配乐**：
  ```bash
  # 标准 1080p30 高画质成片（约 15~25 秒完成 1200 帧渲染与音视频封装）
  python render.py --preset high

  # 极限超采样母带级画质（内部 3840×2160 2×2 SSAA 超采样 + 18,000 GPU 粒子）
  python render.py --preset ultra

  # 同时导出完整 1,200 张无损 PNG 帧序列（output/frames/frame_0000.png ~ frame_1199.png）
  python render.py --preset high --export-frames

  # 低配核显/无独显机器降级渲染
  python render.py --preset medium
  python render.py --preset low --backend cpu
  ```

### 2. 从帧序列手动合成为 MP4 / GIF 的 FFmpeg 命令

当使用 `--export-frames` 导出完整帧序列 `output/frames/frame_%04d.png` 与 `output/soundtrack.wav` 后，可执行以下命令手动合成：

```bash
# 帧序列 + WAV 合成为 1920x1080 30fps H.264 MP4
ffmpeg -y -framerate 30 -i output/frames/frame_%04d.png -i output/soundtrack.wav \
  -c:v libx264 -preset slow -profile:v high -crf 16 -pix_fmt yuv420p \
  -c:a aac -b:a 256k -movflags +faststart -shortest output/chronogenesis_from_seq.mp4

# 合成高质量调色板自适应 GIF 动图
ffmpeg -y -ss 20.5 -t 6.0 -i output/chronogenesis_1080p30.mp4 \
  -vf "fps=15,scale=640:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=192[p];[s1][p]paletteuse=dither=bayer" \
  output/preview.gif
```

### 3. 浏览器离线单页实时运行（WebGL2 + WebAudio）

无需任何本地服务器或网络请求，直接用现代浏览器（Chrome / Edge / Firefox）打开：
```text
web/index.html
```
点击底部控制栏 **「▶ 播放视听作品」** 即可实时合成多轨配乐并驱动四幕着色器，支持进度拖拽、三档分辨率实时切换、帧率低于 24 FPS 自动降级保帧，以及浏览器端一键录制导出 `.webm` 视频。若修改了 `shaders/` 源码，运行 `python build_web_player.py` 即可重新打包单文件 `web/index.html`。
