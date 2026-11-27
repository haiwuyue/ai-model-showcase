"""
Unified configuration, narrative structure, and performance/degradation presets
for 《CHRONOGENESIS: 熵寂回响》(Echoes of Negative Entropy).
"""

from dataclasses import dataclass
from typing import Dict


# Core timeline constants (96 BPM -> 1 beat = 0.625s, 1 bar = 2.50s, 16 bars = 40.0s)
BPM: float = 96.0
BEAT_SEC: float = 60.0 / BPM          # 0.625s
BAR_SEC: float = BEAT_SEC * 4.0       # 2.500s
TOTAL_BARS: int = 16
DURATION_SEC: float = BAR_SEC * TOTAL_BARS  # 40.0s exact
SAMPLE_RATE: int = 48000


@dataclass(frozen=True)
class QualityPreset:
    """Performance budget & graceful degradation presets."""
    name: str
    width: int
    height: int
    fps: int
    max_march_steps: int
    volumetric_steps: int
    shadow_steps: int
    ao_samples: int
    particle_count: int
    bloom_passes: int
    supersample: int  # 1 = native, 2 = 2x2 temporal/spatial jitter tap
    crf: int
    description: str


QUALITY_PRESETS: Dict[str, QualityPreset] = {
    "ultra": QualityPreset(
        name="ultra",
        width=1920,
        height=1080,
        fps=30,
        max_march_steps=112,
        volumetric_steps=36,
        shadow_steps=24,
        ao_samples=5,
        particle_count=18000,
        bloom_passes=4,
        supersample=2,
        crf=16,
        description="1080p30 Cinema Master (2x Subpixel AA, 112-step SDF, 18k GPU particles, 4-pass HDR Bloom)",
    ),
    "high": QualityPreset(
        name="high",
        width=1920,
        height=1080,
        fps=30,
        max_march_steps=90,
        volumetric_steps=28,
        shadow_steps=18,
        ao_samples=4,
        particle_count=12000,
        bloom_passes=3,
        supersample=1,
        crf=17,
        description="1080p30 Production High (1x AA, 90-step SDF, 12k GPU particles, 3-pass HDR Bloom)",
    ),
    "medium": QualityPreset(
        name="medium",
        width=1280,
        height=720,
        fps=30,
        max_march_steps=68,
        volumetric_steps=20,
        shadow_steps=12,
        ao_samples=3,
        particle_count=6000,
        bloom_passes=2,
        supersample=1,
        crf=19,
        description="720p30 Integrated GPU / Laptop Balanced preset",
    ),
    "low": QualityPreset(
        name="low",
        width=960,
        height=540,
        fps=24,
        max_march_steps=48,
        volumetric_steps=12,
        shadow_steps=8,
        ao_samples=2,
        particle_count=2500,
        bloom_passes=1,
        supersample=1,
        crf=21,
        description="540p24 Legacy GPU / CPU Fallback fast preview preset",
    ),
}


NARRATIVE_ACTS = [
    {
        "act": 1,
        "code": "ACT_I_SEED",
        "title": "起 · 虚空微澜 (Primordial Seed)",
        "t_start": 0.0,
        "t_end": 10.0,
        "palette": ("#04070F", "#0E294B", "#38BDF8", "#BAE6FD"),
        "summary": "幽暗深渊中悬浮液态黑曜分形原核，随低频心跳闪烁生物青蓝脉络，微距浅景深推镜。",
    },
    {
        "act": 2,
        "code": "ACT_II_CATHEDRAL",
        "title": "承 · 秩序织造 (Cathedral of Folding Space)",
        "t_start": 10.0,
        "t_end": 20.0,
        "palette": ("#090D16", "#1E293B", "#D97706", "#FDE68A"),
        "summary": "Kleinian-Mandelbox 空间折叠展开为恢弘鎏金分形圣殿，体积丁达尔光束斜切，旋度粒子盘旋升腾。",
    },
    {
        "act": 3,
        "code": "ACT_III_SINGULARITY",
        "title": "转 · 奇点共振 (Relativistic Resonance)",
        "t_start": 20.0,
        "t_end": 30.0,
        "palette": ("#0B0418", "#C026D3", "#FF8A1E", "#22D3EE"),
        "summary": "引力透镜奇点点燃爱因斯坦环，陀螺戴森分形环高速咬合，镜头俯冲穿越万千流体光轨，视听高潮。",
    },
    {
        "act": 4,
        "code": "ACT_IV_HORIZON",
        "title": "合 · 负熵归寂 (Negative Entropy Horizon)",
        "t_start": 30.0,
        "t_end": 40.0,
        "palette": ("#050811", "#64748B", "#E2E8F0", "#F8FAFC"),
        "summary": "纯白冲击波将狂暴等离子凝华为半透明星晶矩阵，倒映于镜面星海之上，拉远归于宇宙微尘。",
    },
]
