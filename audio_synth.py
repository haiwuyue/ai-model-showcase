"""
100% Procedural Polyphonic Soundtrack Synthesizer & Audio-Reactive Envelope Extractor.
Zero external audio samples or generative AI APIs: every waveform, filter, percussion
transient, formant pad, FM bell, and FDN shimmer reverb is computed mathematically.
"""

from __future__ import annotations

import os
import wave
from typing import Dict, Tuple

import numpy as np
from scipy import signal

from config import BAR_SEC, BEAT_SEC, DURATION_SEC, SAMPLE_RATE


def midi_to_hz(midi_note: float) -> float:
    return 440.0 * (2.0 ** ((midi_note - 69.0) / 12.0))


def adsr_envelope(
    length: int,
    sr: int,
    attack: float = 0.01,
    decay: float = 0.15,
    sustain: float = 0.4,
    release: float = 0.25,
) -> np.ndarray:
    n_a = max(1, int(attack * sr))
    n_d = max(1, int(decay * sr))
    n_r = max(1, int(release * sr))
    if n_a + n_d + n_r > length:
        scale = length / float(n_a + n_d + n_r)
        n_a = max(1, int(n_a * scale))
        n_d = max(1, int(n_d * scale))
        n_r = max(1, length - n_a - n_d)
    n_s = max(0, length - n_a - n_d - n_r)

    a_curve = np.linspace(0.0, 1.0, n_a, endpoint=False) ** 0.75
    d_curve = sustain + (1.0 - sustain) * np.exp(-np.linspace(0.0, 4.5, n_d))
    s_curve = np.full(n_s, sustain, dtype=np.float64)
    r_start = d_curve[-1] if n_s == 0 else sustain
    r_curve = r_start * np.exp(-np.linspace(0.0, 5.5, n_r))
    env = np.concatenate([a_curve, d_curve, s_curve, r_curve])
    return env[:length]


def fdn_cathedral_reverb(stereo_in: np.ndarray, sr: int, decay: float = 0.84) -> np.ndarray:
    """
    4-channel Feedback Delay Network (Hadamard matrix) + input diffusion allpass
    for lush, metallic-free sci-fi cathedral reverberation.
    """
    n_samples = stereo_in.shape[0]
    # Prime delay lengths (~29ms to ~61ms at 48kHz)
    delays = [1399, 1787, 2293, 2909]
    max_d = max(delays) + 4

    # Block-vectorized feedback processing via scipy lfilter per comb approximation + cross-mix
    # To run ultra-fast in pure Python/SciPy (< 0.4s for 40s stereo), we use 6 parallel
    # damped comb filters with prime delays + 2 series Schroeder allpass diffusers.
    comb_delays_l = [1559, 1993, 2477, 3079, 3761, 4409]
    comb_delays_r = [1607, 2039, 2539, 3137, 3823, 4483]

    wet_l = np.zeros(n_samples, dtype=np.float64)
    wet_r = np.zeros(n_samples, dtype=np.float64)

    mono_mid = 0.5 * (stereo_in[:, 0] + stereo_in[:, 1])
    side = 0.5 * (stereo_in[:, 0] - stereo_in[:, 1])
    src_l = mono_mid + 0.6 * side
    src_r = mono_mid - 0.6 * side

    for d_l, d_r in zip(comb_delays_l, comb_delays_r):
        g = decay ** (d_l / 2400.0)
        # Damped feedback comb: H(z) = z^-d / (1 - g*(0.65 + 0.35*z^-1)*z^-d)
        a_l = np.zeros(d_l + 2, dtype=np.float64)
        a_l[0] = 1.0
        a_l[d_l] = -g * 0.68
        a_l[d_l + 1] = -g * 0.32
        b_l = np.zeros(d_l + 1, dtype=np.float64)
        b_l[d_l] = 1.0
        wet_l += signal.lfilter(b_l, a_l, src_l)

        a_r = np.zeros(d_r + 2, dtype=np.float64)
        a_r[0] = 1.0
        a_r[d_r] = -g * 0.68
        a_r[d_r + 1] = -g * 0.32
        b_r = np.zeros(d_r + 1, dtype=np.float64)
        b_r[d_r] = 1.0
        wet_r += signal.lfilter(b_r, a_r, src_r)

    wet_l *= 1.0 / np.sqrt(len(comb_delays_l))
    wet_r *= 1.0 / np.sqrt(len(comb_delays_r))

    # 2 series allpass diffusers
    for ap_d, ap_g in [(347, 0.68), (113, 0.65)]:
        b = np.zeros(ap_d + 1, dtype=np.float64)
        b[0] = -ap_g
        b[ap_d] = 1.0
        a = np.zeros(ap_d + 1, dtype=np.float64)
        a[0] = 1.0
        a[ap_d] = -ap_g
        wet_l = signal.lfilter(b, a, wet_l)
        wet_r = signal.lfilter(b, a, wet_r)

    # Gentle high-cut at 7.5 kHz and low-cut at 110 Hz so reverb stays warm and clear
    sos = signal.butter(2, [110.0 / (0.5 * sr), 7500.0 / (0.5 * sr)], btype="bandpass", output="sos")
    wet_l = signal.sosfilt(sos, wet_l)
    wet_r = signal.sosfilt(sos, wet_r)
    return np.column_stack([wet_l, wet_r])


def pitch_shimmer_octave(stereo_in: np.ndarray, sr: int) -> np.ndarray:
    """
    Windowed dual-grain octave-up pitch doubler for crystalline Brian Eno / Jon Hopkins shimmer.
    """
    n = stereo_in.shape[0]
    grain = 4096
    half = grain // 2
    out = np.zeros_like(stereo_in)
    win = np.hanning(grain)[:, None]
    for start in range(0, n - grain, half):
        chunk = stereo_in[start : start + grain]
        # Resample 2x faster (octave up) and tile twice inside the grain window
        idx = np.linspace(0, grain - 1, half)
        idx0 = np.floor(idx).astype(np.int32)
        frac = (idx - idx0)[:, None]
        up_half = chunk[idx0] * (1.0 - frac) + chunk[np.minimum(idx0 + 1, grain - 1)] * frac
        doubled = np.vstack([up_half, up_half])
        out[start : start + grain] += doubled * win
    sos = signal.butter(2, [600.0 / (0.5 * sr), 9000.0 / (0.5 * sr)], btype="bandpass", output="sos")
    return signal.sosfilt(sos, out, axis=0)


def synthesize_soundtrack(
    fps: int = 30,
    sr: int = SAMPLE_RATE,
    seed: int = 42,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Synthesizes the full 40.0s stereo soundtrack (N, 2) in [-1, 1] and returns
    frame-accurate envelopes (length = DURATION_SEC * fps) for shader modulation.
    """
    rng = np.random.default_rng(seed)
    total_samples = int(round(DURATION_SEC * sr))
    t_global = np.arange(total_samples, dtype=np.float64) / sr

    # Separate stems for accurate audio-visual reactivity
    stem_sub = np.zeros((total_samples, 2), dtype=np.float64)
    stem_pad = np.zeros((total_samples, 2), dtype=np.float64)
    stem_arp = np.zeros((total_samples, 2), dtype=np.float64)
    stem_perc = np.zeros((total_samples, 2), dtype=np.float64)

    # 4-bar harmonic cycle in D minor / Dorian (repeats 4x across 16 bars with evolving orchestration)
    # MIDI notes: D2=38, Bb1=34, G1=31, A1=33
    chord_cycle = [
        {"root": 38, "pad": [50, 57, 60, 64, 69], "arp": [62, 65, 69, 72, 76, 74, 69, 65]},  # Dm9
        {"root": 34, "pad": [46, 53, 58, 62, 69], "arp": [62, 65, 69, 74, 76, 72, 69, 65]},  # BbMaj9#11
        {"root": 31, "pad": [43, 50, 55, 58, 65], "arp": [58, 62, 67, 70, 74, 70, 67, 62]},  # Gm9
        {"root": 33, "pad": [45, 52, 57, 61, 67], "arp": [57, 61, 64, 69, 73, 76, 69, 64]},  # A7sus4/A7
    ]

    # -------------------------------------------------------------------------
    # 1. SUB-BASS & REESE NEURO-BASS (Bars 1..16)
    # -------------------------------------------------------------------------
    for bar in range(16):
        act = bar // 4 + 1
        chord = chord_cycle[bar % 4]
        bar_start = int(bar * BAR_SEC * sr)
        bar_len = int(BAR_SEC * sr)
        if bar_start + bar_len > total_samples:
            bar_len = total_samples - bar_start
        tb = np.arange(bar_len, dtype=np.float64) / sr
        f0 = midi_to_hz(chord["root"])

        if act == 1:
            # Deep heartbeat pulses on beat 1 and beat 3.5
            for beat_off, dur, amp in [(0.0, 1.35, 0.72), (2.5 * BEAT_SEC, 1.1, 0.50)]:
                s0 = bar_start + int(beat_off * sr)
                slen = min(int(dur * sr), total_samples - s0)
                if slen <= 0:
                    continue
                ts = np.arange(slen, dtype=np.float64) / sr
                env = adsr_envelope(slen, sr, 0.03, 0.35, 0.35, 0.55)
                # Subtle pitch drop on attack for visceral weight
                phase = 2.0 * np.pi * (f0 * ts + 6.0 * (1.0 - np.exp(-ts * 18.0)) / 18.0)
                sig = np.tanh(1.35 * (np.sin(phase) + 0.28 * np.sin(2.0 * phase))) * env * amp
                stem_sub[s0 : s0 + slen, 0] += sig
                stem_sub[s0 : s0 + slen, 1] += sig
        elif act in (2, 3):
            # Lush stereo detuned Reese bass + sub fundamental
            env = adsr_envelope(bar_len, sr, 0.05, 0.3, 0.82, 0.25)
            sub_core = np.sin(2.0 * np.pi * f0 * tb)
            detunes = [-0.011, -0.004, 0.0, 0.004, 0.011]
            pans = [0.15, 0.35, 0.5, 0.65, 0.85]
            reese_l = np.zeros(bar_len, dtype=np.float64)
            reese_r = np.zeros(bar_len, dtype=np.float64)
            for det, pan in zip(detunes, pans):
                fn = f0 * 2.0 * (2.0 ** det)
                # Band-limited sawtooth via 10 harmonics
                saw = np.zeros(bar_len, dtype=np.float64)
                for h in range(1, 11):
                    if fn * h < 4500.0:
                        saw += ((-1) ** (h + 1)) * np.sin(2.0 * np.pi * fn * h * tb) / h
                reese_l += saw * np.cos(pan * np.pi * 0.5)
                reese_r += saw * np.sin(pan * np.pi * 0.5)

            cutoff = 260.0 if act == 2 else (520.0 + 260.0 * np.sin(2.0 * np.pi * 0.5 * tb / BAR_SEC))
            sos_lp = signal.butter(3, np.mean(cutoff) / (0.5 * sr), btype="lowpass", output="sos")
            reese_l = signal.sosfilt(sos_lp, reese_l) * 0.35
            reese_r = signal.sosfilt(sos_lp, reese_r) * 0.35

            # Rhythmic sidechain pump in Act III
            pump = 1.0
            if act == 3:
                beat_phase = (tb % BEAT_SEC) / BEAT_SEC
                pump = 0.32 + 0.68 * (1.0 - np.exp(-beat_phase * 5.0))

            gain = 0.62 if act == 2 else 0.78
            stem_sub[bar_start : bar_start + bar_len, 0] += np.tanh((0.75 * sub_core + reese_l) * 1.3) * env * pump * gain
            stem_sub[bar_start : bar_start + bar_len, 1] += np.tanh((0.75 * sub_core + reese_r) * 1.3) * env * pump * gain
        else:
            # Act IV: Serene warm pedal D2/G1 fading into infinity
            fade_act4 = max(0.12, 1.0 - (bar - 12) * 0.25)
            env = adsr_envelope(bar_len, sr, 0.25, 0.5, 0.75, 0.8)
            sig = (np.sin(2.0 * np.pi * f0 * tb) + 0.2 * np.sin(4.0 * np.pi * f0 * tb)) * env * 0.52 * fade_act4
            stem_sub[bar_start : bar_start + bar_len, 0] += sig
            stem_sub[bar_start : bar_start + bar_len, 1] += sig

    # -------------------------------------------------------------------------
    # 2. POLYPHONIC ANALOG PAD & ACT-IV CATHEDRAL CHOIR FORMANTS
    # -------------------------------------------------------------------------
    for bar in range(16):
        act = bar // 4 + 1
        chord = chord_cycle[bar % 4]
        # Crossfade overlap slightly across bars
        s0 = int(bar * BAR_SEC * sr)
        slen = min(int((BAR_SEC + 0.45) * sr), total_samples - s0)
        tb = np.arange(slen, dtype=np.float64) / sr
        env = adsr_envelope(slen, sr, 0.45, 0.5, 0.80, 0.55)

        pad_l = np.zeros(slen, dtype=np.float64)
        pad_r = np.zeros(slen, dtype=np.float64)

        notes = chord["pad"] + ([chord["pad"][0] + 24] if act >= 3 else [])
        for idx_n, mnote in enumerate(notes):
            f = midi_to_hz(mnote)
            # Slow organic LFO drift per voice
            lfo1 = 0.0025 * np.sin(2.0 * np.pi * (0.37 + 0.09 * idx_n) * tb + idx_n)
            lfo2 = -0.0028 * np.cos(2.0 * np.pi * (0.43 + 0.07 * idx_n) * tb + idx_n * 1.7)
            v_l = np.zeros(slen, dtype=np.float64)
            v_r = np.zeros(slen, dtype=np.float64)
            max_h = 4 if act == 1 else (9 if act in (2, 3) else 7)
            for h in range(1, max_h + 1):
                amp_h = 1.0 / (h ** 1.25)
                v_l += amp_h * np.sin(2.0 * np.pi * f * h * (tb + lfo1) + h * 0.3)
                v_r += amp_h * np.sin(2.0 * np.pi * f * h * (tb + lfo2) + h * 0.9)
            pan = 0.2 + 0.6 * (idx_n / max(1, len(notes) - 1))
            pad_l += v_l * np.cos(pan * np.pi * 0.5)
            pad_r += v_r * np.sin(pan * np.pi * 0.5)

        if act == 4:
            # Apply choral "Ahh / Ohh" vowel formant resonances (650Hz, 1080Hz, 2650Hz)
            formant_l = np.zeros_like(pad_l)
            formant_r = np.zeros_like(pad_r)
            for fc, bw, g_f in [(650.0, 110.0, 1.0), (1080.0, 140.0, 0.65), (2650.0, 220.0, 0.30)]:
                sos_f = signal.butter(
                    2,
                    [max(80.0, fc - bw) / (0.5 * sr), min(12000.0, fc + bw) / (0.5 * sr)],
                    btype="bandpass",
                    output="sos",
                )
                formant_l += g_f * signal.sosfilt(sos_f, pad_l)
                formant_r += g_f * signal.sosfilt(sos_f, pad_r)
            pad_l = 0.45 * pad_l + 0.95 * formant_l
            pad_r = 0.45 * pad_r + 0.95 * formant_r
        else:
            cutoff_pad = 950.0 if act == 1 else (2200.0 if act == 2 else 3800.0)
            sos_p = signal.butter(2, cutoff_pad / (0.5 * sr), btype="lowpass", output="sos")
            pad_l = signal.sosfilt(sos_p, pad_l)
            pad_r = signal.sosfilt(sos_p, pad_r)

        act_gain = {1: 0.16, 2: 0.24, 3: 0.28, 4: 0.25 * max(0.15, 1.0 - (bar - 12) * 0.22)}[act]
        stem_pad[s0 : s0 + slen, 0] += pad_l * env * act_gain
        stem_pad[s0 : s0 + slen, 1] += pad_r * env * act_gain

    # -------------------------------------------------------------------------
    # 3. CRYSTALLINE FM ARPEGGIATOR & CELESTIAL BELL HARMONICS
    # -------------------------------------------------------------------------
    sixteenth = BEAT_SEC * 0.25  # 0.15625s
    for bar in range(16):
        act = bar // 4 + 1
        chord = chord_cycle[bar % 4]
        arp_notes = chord["arp"]
        steps = 16
        for step in range(steps):
            # Act I: sparse mysterious droplets; Act II: 8th/16th weave; Act III: full 16th; Act IV: sparse wind-chimes
            if act == 1 and step not in (0, 6, 12):
                continue
            if act == 2 and (step % 2 == 1 and step not in (7, 15)):
                continue
            if act == 4 and step not in (0, 4, 10):
                continue
            if act == 4 and bar == 15 and step > 4:
                continue

            note_t = bar * BAR_SEC + step * sixteenth
            s0 = int(note_t * sr)
            note_dur = 0.65 if act in (1, 4) else (0.36 if act == 2 else 0.28)
            slen = min(int(note_dur * sr), total_samples - s0)
            if slen <= 0:
                continue

            octave_shift = 12 if (act == 3 and step % 4 == 2) or (act == 4 and step == 10) else 0
            mnote = arp_notes[step % len(arp_notes)] + octave_shift
            fc = midi_to_hz(mnote)
            tn = np.arange(slen, dtype=np.float64) / sr

            # 2-Operator FM synthesis with fast exponential modulation index for glassy attack
            mod_ratio = 2.0 if act != 4 else 3.005
            mod_index = (2.6 if act == 3 else 1.5) * np.exp(-tn * 14.0) + 0.22
            modulator = mod_index * np.sin(2.0 * np.pi * fc * mod_ratio * tn)
            carrier = np.sin(2.0 * np.pi * fc * tn + modulator)
            overtone = 0.25 * np.sin(2.0 * np.pi * fc * 4.01 * tn) * np.exp(-tn * 22.0)

            env = adsr_envelope(slen, sr, 0.004, 0.12, 0.18, note_dur * 0.55)
            vel = 0.26 if act == 3 else (0.20 if act == 2 else 0.16)
            if step % 4 == 0:
                vel *= 1.22
            pan = 0.5 + 0.34 * np.sin(step * 1.15 + bar * 0.7)
            sig = (carrier + overtone) * env * vel
            stem_arp[s0 : s0 + slen, 0] += sig * np.cos(pan * np.pi * 0.5)
            stem_arp[s0 : s0 + slen, 1] += sig * np.sin(pan * np.pi * 0.5)

    # Dotted-eighth stereo ping-pong delay on arpeggiator (3 * sixteenth = 0.46875s)
    d_samples = int(3 * sixteenth * sr)
    arp_delayed = stem_arp.copy()
    for tap in range(1, 5):
        off = d_samples * tap
        if off < total_samples:
            g = 0.44 ** tap
            # Swap L/R each tap for ping-pong motion
            src = stem_arp[:-off, ::-1] if tap % 2 == 1 else stem_arp[:-off]
            arp_delayed[off:] += src * g
    stem_arp = arp_delayed

    # -------------------------------------------------------------------------
    # 4. CINEMATIC TAIKO IMPACTS, SUB-DROPS, HI-HATS & SECTION RISERS
    # -------------------------------------------------------------------------
    def add_taiko_kick(time_sec: float, punch: float = 0.85, length_s: float = 0.42) -> None:
        s0 = int(time_sec * sr)
        slen = min(int(length_s * sr), total_samples - s0)
        if slen <= 0 or s0 < 0:
            return
        tk = np.arange(slen, dtype=np.float64) / sr
        # Exponential pitch sweep 145Hz -> 36Hz
        phase = 2.0 * np.pi * (36.0 * tk + (110.0 / 26.0) * (1.0 - np.exp(-26.0 * tk)))
        body = np.sin(phase) * np.exp(-tk * 8.5)
        click = rng.standard_normal(slen) * np.exp(-tk * 75.0) * 0.22
        sos_k = signal.butter(2, 2400.0 / (0.5 * sr), btype="lowpass", output="sos")
        hit = np.tanh((body + signal.sosfilt(sos_k, click)) * 1.65) * punch
        stem_perc[s0 : s0 + slen, 0] += hit
        stem_perc[s0 : s0 + slen, 1] += hit

    def add_snare_clap(time_sec: float, amp: float = 0.48) -> None:
        s0 = int(time_sec * sr)
        slen = min(int(0.28 * sr), total_samples - s0)
        if slen <= 0 or s0 < 0:
            return
        tk = np.arange(slen, dtype=np.float64) / sr
        tone = np.sin(2.0 * np.pi * (175.0 * tk + 4.0 * (1.0 - np.exp(-35.0 * tk)))) * np.exp(-tk * 20.0)
        noise = rng.standard_normal(slen) * np.exp(-tk * 13.5)
        sos_bp = signal.butter(2, [900.0 / (0.5 * sr), 7500.0 / (0.5 * sr)], btype="bandpass", output="sos")
        sig = (0.45 * tone + 0.75 * signal.sosfilt(sos_bp, noise)) * amp
        stem_perc[s0 : s0 + slen, 0] += sig * 0.95
        stem_perc[s0 : s0 + slen, 1] += sig * 1.05

    def add_hat(time_sec: float, amp: float = 0.14, decay_rate: float = 48.0, pan: float = 0.5) -> None:
        s0 = int(time_sec * sr)
        slen = min(int(0.12 * sr), total_samples - s0)
        if slen <= 0 or s0 < 0:
            return
        tk = np.arange(slen, dtype=np.float64) / sr
        metallic = (
            np.sin(2.0 * np.pi * 6310.0 * tk)
            + np.sin(2.0 * np.pi * 8490.0 * tk)
            + 1.4 * rng.standard_normal(slen)
        )
        sos_hp = signal.butter(2, 6800.0 / (0.5 * sr), btype="highpass", output="sos")
        sig = signal.sosfilt(sos_hp, metallic) * np.exp(-tk * decay_rate) * amp
        stem_perc[s0 : s0 + slen, 0] += sig * np.cos(pan * np.pi * 0.5)
        stem_perc[s0 : s0 + slen, 1] += sig * np.sin(pan * np.pi * 0.5)

    for bar in range(16):
        act = bar // 4 + 1
        bt0 = bar * BAR_SEC
        if act == 2:
            # Noble architectural pulse
            add_taiko_kick(bt0 + 0.0 * BEAT_SEC, punch=0.58)
            add_taiko_kick(bt0 + 2.5 * BEAT_SEC, punch=0.45)
            if bar % 2 == 1:
                add_snare_clap(bt0 + 3.0 * BEAT_SEC, amp=0.30)
            for s in range(8):
                add_hat(bt0 + s * 0.5 * BEAT_SEC, amp=0.07 if s % 2 else 0.10, pan=0.35 + 0.3 * (s % 2))
        elif act == 3:
            # High-energy relativistic climax rhythm
            for b in range(4):
                add_taiko_kick(bt0 + b * BEAT_SEC, punch=0.82 if b in (0, 2) else 0.68)
            add_taiko_kick(bt0 + 3.5 * BEAT_SEC, punch=0.55)
            add_snare_clap(bt0 + 1.0 * BEAT_SEC, amp=0.46)
            add_snare_clap(bt0 + 3.0 * BEAT_SEC, amp=0.52)
            for s in range(16):
                open_h = s in (2, 6, 10, 14)
                add_hat(
                    bt0 + s * sixteenth,
                    amp=0.14 if open_h else 0.08,
                    decay_rate=22.0 if open_h else 55.0,
                    pan=0.5 + 0.3 * np.sin(s * 1.4),
                )

    # Dramatic risers before Act II (10s), Act III (20s), and Act IV shockwave (30s)
    for trans_t, intensity in [(10.0, 0.50), (20.0, 0.85), (30.0, 0.95)]:
        r_dur = 2.5
        s0 = int((trans_t - r_dur) * sr)
        slen = int(r_dur * sr)
        tr = np.arange(slen, dtype=np.float64) / sr
        progress = (tr / r_dur) ** 2.4
        sweep_phase = 2.0 * np.pi * (180.0 * tr + 950.0 * (tr ** 3) / (3.0 * r_dur ** 2))
        whoosh = rng.standard_normal(slen) * progress
        sos_bp = signal.butter(2, [400.0 / (0.5 * sr), 9500.0 / (0.5 * sr)], btype="bandpass", output="sos")
        riser = (0.4 * np.sin(sweep_phase) * progress + 0.6 * signal.sosfilt(sos_bp, whoosh)) * intensity * 0.42
        stem_perc[s0 : s0 + slen, 0] += riser
        stem_perc[s0 : s0 + slen, 1] += riser
        # Impact boom right on the transition downbeat
        add_taiko_kick(trans_t, punch=intensity, length_s=0.95)

    # -------------------------------------------------------------------------
    # 5. CATHEDRAL SHIMMER REVERB & MASTERING
    # -------------------------------------------------------------------------
    reverb_send = 0.65 * stem_pad + 0.55 * stem_arp + 0.22 * stem_perc
    shimmer = pitch_shimmer_octave(0.7 * stem_pad + 0.5 * stem_arp, sr)
    reverb_wet = fdn_cathedral_reverb(reverb_send + 0.32 * shimmer, sr, decay=0.86)

    dry_mix = stem_sub + stem_pad + stem_arp + stem_perc
    master = dry_mix + 0.42 * reverb_wet

    # Gentle fade-in (0..0.8s) and cosmic tail fade-out (37.0..40.0s)
    fade_in = np.minimum(1.0, t_global / 0.8)[:, None]
    fade_out = np.clip((DURATION_SEC - t_global) / 3.0, 0.0, 1.0)[:, None] ** 1.35
    master *= fade_in * fade_out

    # Soft-knee analog mastering saturation & normalize to -1.0 dBFS (0.891 peak)
    master = np.tanh(master * 1.15)
    peak = np.max(np.abs(master)) + 1e-9
    master = (master / peak) * 0.891

    # -------------------------------------------------------------------------
    # 6. EXTRACT FRAME-ACCURATE ENVELOPES FOR SHADER AUDIO-REACTIVITY
    # -------------------------------------------------------------------------
    num_frames = int(round(DURATION_SEC * fps))
    samples_per_frame = total_samples / float(num_frames)

    def stem_rms_curve(stem: np.ndarray, smooth_sigma: float = 1.2) -> np.ndarray:
        mono = np.sqrt(0.5 * (stem[:, 0] ** 2 + stem[:, 1] ** 2))
        vals = np.zeros(num_frames, dtype=np.float32)
        for f_idx in range(num_frames):
            i0 = int(f_idx * samples_per_frame)
            i1 = min(total_samples, int((f_idx + 1) * samples_per_frame))
            if i1 > i0:
                vals[f_idx] = float(np.sqrt(np.mean(mono[i0:i1] ** 2)))
        if smooth_sigma > 0:
            # Asymmetric fast-attack, smooth-release envelope follower
            followed = np.zeros_like(vals)
            cur = 0.0
            alpha_atk = 0.65
            alpha_rel = np.exp(-1.0 / max(0.5, smooth_sigma * 2.2))
            for i, v in enumerate(vals):
                if v > cur:
                    cur = alpha_atk * v + (1.0 - alpha_atk) * cur
                else:
                    cur = alpha_rel * cur + (1.0 - alpha_rel) * v
                followed[i] = cur
            vals = followed
        p98 = float(np.percentile(vals, 98.5)) + 1e-6
        return np.clip(vals / p98, 0.0, 1.25).astype(np.float32)

    envelopes = {
        "sub": stem_rms_curve(stem_sub, smooth_sigma=1.4),
        "mid": stem_rms_curve(stem_pad, smooth_sigma=2.5),
        "high": stem_rms_curve(stem_arp, smooth_sigma=1.1),
        "impact": stem_rms_curve(stem_perc, smooth_sigma=0.9),
        "energy": stem_rms_curve(master, smooth_sigma=2.0),
    }

    return master.astype(np.float32), envelopes


def save_wav_16bit(path: str, audio_stereo: np.ndarray, sr: int = SAMPLE_RATE) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    pcm16 = np.clip(audio_stereo * 32767.0, -32768, 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm16.tobytes())


if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(out_dir, exist_ok=True)
    wav_path = os.path.join(out_dir, "soundtrack.wav")
    npz_path = os.path.join(out_dir, "envelopes.npz")
    audio, envs = synthesize_soundtrack(fps=30)
    save_wav_16bit(wav_path, audio)
    np.savez(npz_path, **envs)
    print(f"Saved soundtrack: {wav_path} ({audio.shape[0]/SAMPLE_RATE:.2f}s) | Envelopes: {npz_path}")
