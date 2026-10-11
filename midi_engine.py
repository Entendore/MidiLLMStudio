"""
MIDI Engine — LLM generation, demo patterns, synth, file I/O.
"""

import os
import json
import random
import math
import struct
import threading
import uuid
import tempfile
import time
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple, Callable, Dict
from urllib.request import urlretrieve
from urllib.error import URLError

import numpy as np

# Optional imports
try:
    from llama_cpp import Llama
    LLAMA_AVAILABLE = True
except ImportError:
    LLAMA_AVAILABLE = False

try:
    from huggingface_hub import hf_hub_download, HfApi
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

try:
    import mido
    from mido import Message, MidiFile, MidiTrack, MetaMessage
    MIDO_AVAILABLE = True
except ImportError:
    MIDO_AVAILABLE = False

try:
    import sounddevice as sd
    SD_AVAILABLE = True
except ImportError:
    SD_AVAILABLE = False


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass
class MidiNote:
    tick: int
    pitch: int       # 0-127
    velocity: int    # 1-127
    duration: int    # ticks
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])

    def to_dict(self):
        return {"tick": self.tick, "pitch": self.pitch,
                "velocity": self.velocity, "duration": self.duration, "id": self.id}

    @staticmethod
    def from_dict(d):
        return MidiNote(tick=d["tick"], pitch=d["pitch"],
                        velocity=d["velocity"], duration=d["duration"],
                        id=d.get("id", uuid.uuid4().hex[:8]))


# ---------------------------------------------------------------------------
# Music theory helpers
# ---------------------------------------------------------------------------

SCALES = {
    "major":      [0, 2, 4, 5, 7, 9, 11],
    "minor":      [0, 2, 3, 5, 7, 8, 10],
    "dorian":     [0, 2, 3, 5, 7, 9, 10],
    "mixolydian": [0, 2, 4, 5, 7, 9, 10],
    "pentatonic": [0, 2, 4, 7, 9],
    "blues":      [0, 3, 5, 6, 7, 10],
    "harmonic minor": [0, 2, 3, 5, 7, 8, 11],
    "melodic minor":  [0, 2, 3, 5, 7, 9, 11],
}

KEY_OFFSETS = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
    "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8,
    "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11,
}

CHORD_PROGRESSIONS = {
    "pop":    [[0, 4, 7], [7, 11, 2], [9, 0, 4], [5, 9, 0]],
    "jazz":   [[2, 5, 9, 0], [7, 11, 2, 5], [0, 4, 7, 11], [9, 0, 4, 7]],
    "blues":  [[0, 4, 7], [0, 4, 7], [0, 4, 7], [0, 4, 7],
               [5, 9, 0], [5, 9, 0], [0, 4, 7], [0, 4, 7],
               [7, 11, 2], [5, 9, 0], [0, 4, 7], [7, 11, 2]],
    "lofi":   [[9, 0, 4], [2, 5, 9], [7, 11, 2], [4, 7, 11]],
    "edm":    [[0, 4, 7], [0, 4, 7], [5, 9, 0], [7, 11, 2]],
    "classical": [[0, 4, 7], [5, 9, 0], [7, 11, 2], [0, 4, 7]],
    "ambient": [[0, 4, 7, 11], [5, 9, 0, 4], [9, 0, 4, 7], [7, 11, 2, 5]],
    "hiphop": [[0, 4, 7], [9, 0, 4], [5, 9, 0], [0, 4, 7]],
    "reggaeton": [[0, 4, 7], [5, 9, 0], [8, 0, 4], [0, 4, 7]],
    "rnb": [[0, 4, 7, 11], [5, 9, 0, 4], [7, 11, 2, 5], [0, 4, 7, 11]],
    "rock": [[0, 7], [5, 0], [7, 2], [0, 7]],
    "latin": [[0, 4, 7], [5, 9, 0], [7, 11, 2], [9, 0, 4]],
    "funk": [[0, 4, 7, 11], [5, 9, 0, 4], [0, 4, 7, 11], [9, 0, 4, 7]],
    "soul": [[0, 4, 7, 11], [2, 5, 9, 0], [7, 11, 2, 5], [9, 0, 4, 7]],
}

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
BLACK_KEYS = {1, 3, 6, 8, 10}

TICKS_PER_BEAT = 480

# Pitch-class color palette as (r, g, b) tuples — 12 colors for C through B
PITCH_COLORS_RGB = [
    (230, 60, 60),    # C  - red
    (240, 100, 50),   # C# - orange-red
    (245, 160, 40),   # D  - orange
    (230, 200, 30),   # D# - yellow
    (120, 210, 50),   # E  - yellow-green
    (50, 200, 100),   # F  - green
    (40, 190, 180),   # F# - teal
    (50, 160, 220),   # G  - blue
    (70, 110, 230),   # G# - indigo
    (120, 80, 220),   # A  - purple
    (180, 60, 200),   # A# - magenta
    (220, 60, 150),   # B  - pink
]


def note_name(pitch: int) -> str:
    return NOTE_NAMES[pitch % 12] + str(pitch // 12 - 1)


def is_black_key(pitch: int) -> bool:
    return (pitch % 12) in BLACK_KEYS


def pitch_class_color_rgb(pitch: int, velocity: int = 100) -> Tuple[int, int, int]:
    """Return (r, g, b) based on pitch class, brightness modulated by velocity."""
    r, g, b = PITCH_COLORS_RGB[pitch % 12]
    factor = 0.5 + (velocity / 127.0) * 0.5
    return (min(255, int(r * factor)), min(255, int(g * factor)), min(255, int(b * factor)))


def build_scale(key: str, scale_name: str, octaves: int = 2) -> List[int]:
    root = KEY_OFFSETS.get(key, 0)
    intervals = SCALES.get(scale_name, SCALES["major"])
    notes = []
    for octave in range(octaves):
        base = 60 + root - 12 + octave * 12
        for interval in intervals:
            p = base + interval
            if 21 <= p <= 108:
                notes.append(p)
    return sorted(set(notes))


# ---------------------------------------------------------------------------
# Demo pattern generator
# ---------------------------------------------------------------------------

def generate_demo_notes(genre: str = "pop", key: str = "C",
                        scale: str = "major", tempo: int = 120,
                        num_bars: int = 4) -> List[MidiNote]:
    notes: List[MidiNote] = []
    tpb = TICKS_PER_BEAT
    beats_per_bar = 4
    ticks_per_bar = tpb * beats_per_bar
    total_ticks = ticks_per_bar * num_bars
    prog_key = genre.lower() if genre.lower() in CHORD_PROGRESSIONS else "pop"
    progression = CHORD_PROGRESSIONS[prog_key]
    scale_notes = build_scale(key, scale, 3)
    root = KEY_OFFSETS.get(key, 0)

    # --- Bass ---
    bar, tick = 0, 0
    while tick < total_ticks:
        chord = progression[bar % len(progression)]
        bass_root = (chord[0] + root + 36) % 12 + 36
        bass_root = min(max(bass_root, 28), 55)
        if prog_key in ("edm", "hiphop"):
            pattern = [0, 0, tpb // 2, tpb // 2, tpb, tpb, 3 * tpb // 2, 3 * tpb // 2]
        elif prog_key == "jazz":
            pattern = [0, tpb // 3, 2 * tpb // 3, tpb]
        else:
            pattern = [0, tpb // 2, tpb, 3 * tpb // 2]
        for pt in pattern:
            if tick + pt >= total_ticks:
                break
            vel = random.randint(70, 100)
            dur = tpb // 2 if prog_key != "jazz" else tpb // 3
            notes.append(MidiNote(tick=tick + pt, pitch=bass_root, velocity=vel, duration=dur))
        tick += ticks_per_bar
        bar += 1

    # --- Chords / pads ---
    bar, tick = 0, 0
    while tick < total_ticks:
        chord = progression[bar % len(progression)]
        chord_pitches = [(cp + root + 60) % 12 + 60 for cp in chord]
        voiced = []
        for i, cp in enumerate(chord_pitches):
            voiced.append(cp + (i // 3) * 12 - 12)
        voiced = [p for p in voiced if 36 <= p <= 96]
        if prog_key in ("edm", "ambient"):
            dur = ticks_per_bar - 10
            for idx, cp in enumerate(voiced[:4]):
                notes.append(MidiNote(tick=tick, pitch=cp,
                                      velocity=max(30, 60 - idx * 8), duration=dur))
        else:
            chord_times = [0, tpb] if prog_key not in ("jazz", "lofi") else [tpb // 3, tpb]
            for ct in chord_times:
                for cp in voiced[:4]:
                    if tick + ct >= total_ticks:
                        break
                    vel = random.randint(40, 70)
                    dur = tpb // 2 if ct == 0 else tpb // 3
                    notes.append(MidiNote(tick=tick + ct, pitch=cp, velocity=vel, duration=dur))
        tick += ticks_per_bar
        bar += 1

    # --- Melody ---
    bar, tick = 0, 0
    prev_pitch = 60 + root
    while tick < total_ticks:
        chord = progression[bar % len(progression)]
        chord_tones = set((cp + root + 72) % 12 + 72 for cp in chord)
        melody_ticks = sorted(random.sample(range(0, ticks_per_bar, tpb // 2),
                                            random.randint(3, 6)))
        for mt in melody_ticks:
            if tick + mt >= total_ticks:
                break
            if random.random() < 0.6:
                target = random.choice(list(chord_tones))
            else:
                candidates = [p for p in scale_notes if 60 <= p <= 84]
                target = random.choice(candidates) if candidates else 72
            diff = target - prev_pitch
            if abs(diff) > 7:
                target = prev_pitch + (7 if diff > 0 else -7)
            target = max(55, min(96, target))
            prev_pitch = target
            vel = random.randint(70, 110)
            dur = random.choice([tpb // 2, tpb // 4, tpb, 3 * tpb // 4])
            if tick + mt + dur > total_ticks:
                dur = total_ticks - tick - mt
            if dur > 0:
                notes.append(MidiNote(tick=tick + mt, pitch=target, velocity=vel, duration=dur))
        tick += ticks_per_bar
        bar += 1

    return notes


# ---------------------------------------------------------------------------
# Recommended GGUF models for download
# ---------------------------------------------------------------------------

RECOMMENDED_MODELS = [
    {
        "name": "TinyLlama 1.1B Chat (Q4)",
        "repo": "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
        "file": "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        "size_mb": "~660 MB",
        "desc": "Fast inference, good for testing",
    },
    {
        "name": "Qwen 2.5 1.5B Instruct (Q4)",
        "repo": "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        "file": "qwen2.5-1.5b-instruct-q4_k_m.gguf",
        "size_mb": "~1.0 GB",
        "desc": "Strong small model, great quality",
    },
    {
        "name": "Llama 3.2 1B Instruct (Q4)",
        "repo": "bartowski/Llama-3.2-1B-Instruct-GGUF",
        "file": "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        "size_mb": "~780 MB",
        "desc": "Meta's latest small model",
    },
    {
        "name": "Llama 3.2 3B Instruct (Q4)",
        "repo": "bartowski/Llama-3.2-3B-Instruct-GGUF",
        "file": "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        "size_mb": "~1.9 GB",
        "desc": "Best quality, needs more RAM",
    },
    {
        "name": "Phi-3 Mini 3.8B (Q4)",
        "repo": "bartowski/Phi-3-mini-4k-instruct-gguf",
        "file": "Phi-3-mini-4k-instruct-q4_k_m.gguf",
        "size_mb": "~2.2 GB",
        "desc": "Microsoft's compact powerhouse",
    },
]
# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

def _llama_wrap(sys, user):
    return "[INST]<<SYS>>" + sys + "<</SYS>>" + user + "[/INST]"

def _chatml_wrap(sys, user):
    return "<|im_start|>system\n" + sys + "<|im_end|>\n<|im_start|>user\n" + user + "<|im_end|>\n<|im_start|>assistant\n"

def _raw_wrap(sys, user):
    return sys + "\n\n" + user + "\n\nResponse:\n"

PROMPT_TEMPLATES = {
    "LlamaChat": {
        "wrap": _llama_wrap,
        "stop": ["[/INST]", "\n\n\n", "Human:", "User:", "ASSISTANT:"],
        "detect": lambda n: any(k in n.lower() for k in ["llama", "tinyllama", "codellama", "mistral", "phi", "qwen"]),
    },
    "ChatML": {
        "wrap": _chatml_wrap,
        "stop": ["<|im_end|>", "<|im_start|>"],
        "detect": lambda n: any(k in n.lower() for k in ["qwen2", "gemma", "yi-", "deepseek"]),
    },
    "Raw": {
        "wrap": _raw_wrap,
        "stop": ["\n\n\n", "Human:", "User:", "ASSISTANT:", "<|"],
        "detect": lambda n: True,
    },
}


# ---------------------------------------------------------------------------
# Genre-aware music theory details for enhanced prompts
# ---------------------------------------------------------------------------

GENRE_DETAILS = {
    "pop": {
        "chord_voicings": "Use root position and first inversion triads. Common: I-V-vi-IV or I-IV-V-I.",
        "rhythm": "Steady 4/4 rhythm. Kick on 1,3. Snare on 2,4. Melody uses quarter and eighth notes.",
        "density": "8-16 simultaneous notes across bass+chords+melody. Melody: 4-8 notes per bar.",
        "bass_range": "36-48", "chord_range": "48-72", "melody_range": "60-84",
    },
    "jazz": {
        "chord_voicings": "Use 7th chords (maj7, min7, dom7). Include 9ths and 13ths. ii-V-I progressions.",
        "rhythm": "Swing feel. Dotted quarter-eighth pairs. Syncopated melodies. Walking bass lines.",
        "density": "4-8 notes per chord. Spread voicings. Melody: 6-12 notes per bar with swing.",
        "bass_range": "28-48", "chord_range": "48-76", "melody_range": "60-90",
    },
    "blues": {
        "chord_voicings": "Dominant 7th chords throughout. 12-bar blues: I7-IV7-I7-V7-IV7-I7-V7.",
        "rhythm": "12/8 or shuffle feel. Triplet-based rhythms. Blue notes (b3, b5, b7).",
        "density": "6-12 notes. Call-and-response phrasing. Melody: 4-8 notes per bar with bends.",
        "bass_range": "36-48", "chord_range": "48-72", "melody_range": "58-80",
    },
    "lofi": {
        "chord_voicings": "Maj7, min7, add9 chords. Soft jazz voicings. Neapolitan and borrowed chords.",
        "rhythm": "Slow tempo. Offbeat chord stabs. Dotted rhythms. Space between phrases.",
        "density": "3-6 simultaneous notes. Sparse arrangement. Melody: 2-4 notes per bar.",
        "bass_range": "36-45", "chord_range": "52-72", "melody_range": "60-80",
    },
    "edm": {
        "chord_voicings": "Power chords (root+5th) or suspended chords. Single-note ostinato patterns.",
        "rhythm": "Four-on-the-floor kick. 16th-note bass patterns. Build-ups with rising pitches.",
        "density": "2-4 notes typically, expanding to 8-16 in drops. Melody: repeating 4-8 note motifs.",
        "bass_range": "24-48", "chord_range": "48-72", "melody_range": "60-84",
    },
    "classical": {
        "chord_voicings": "Triads in all inversions. Alberti bass patterns. Secondary dominants.",
        "rhythm": "Strict rhythmic patterns. Arpeggiated chords. Contrapuntal voice leading.",
        "density": "4-8 notes (piano texture). Melody+accompaniment. 8-16 notes per bar in arpeggios.",
        "bass_range": "36-55", "chord_range": "48-84", "melody_range": "60-96",
    },
    "ambient": {
        "chord_voicings": "Open voicings with wide spacing. 4th and 5th intervals. Sustained pads.",
        "rhythm": "Very slow. Long sustained notes (1-2 bars). Minimal rhythmic movement.",
        "density": "3-6 sustained notes. Heavy reverb implied. 1-2 notes change per bar.",
        "bass_range": "36-48", "chord_range": "48-84", "melody_range": "60-84",
    },
    "hiphop": {
        "chord_voicings": "Simple triads or power chords. Minor keys common. Tritone subs.",
        "rhythm": "Boom-bap or trap. Syncopated kicks. 16th-note hi-hat patterns. Offbeat snares.",
        "density": "3-6 notes. Bass-heavy. Melody: 4-8 notes per bar with trap-style fills.",
        "bass_range": "28-45", "chord_range": "48-72", "melody_range": "60-84",
    },

    "reggaeton": {
        "chord_voicings": "Minor triads, power chords. Simple i-iv-V-i patterns. Syncopated offbeat rhythms borrowed from dancehall reggae.",
        "rhythm": "Dembow rhythm pattern. Strong 4/4 with syncopated offbeat stabs. 90-100 BPM typical. Repetitive hook melodies.",
        "density": "4-8 notes per bar. Bass-heavy with minimal chord stabs. Melody: 4-8 notes per bar, catchy and repetitive.",
        "bass_range": "28-45", "chord_range": "48-67", "melody_range": "60-79",
    },
    "rnb": {
        "chord_voicings": "Extended 7th, 9th, 11th chords. Neo-soul voicings with 2nds and 4ths. Chromatic mediant movement.",
        "rhythm": "Slow to mid-tempo (60-90 BPM). Laid-back groove with triplet feel. Syncopated melodic phrases with space.",
        "density": "6-12 notes. Rich harmonies, gentle dynamics. Melody: 4-8 notes per bar with vocal-style phrasing.",
        "bass_range": "28-48", "chord_range": "48-76", "melody_range": "60-84",
    },
    "rock": {
        "chord_voicings": "Power chords (root+5th) for heavy parts, full triads for lighter sections. I-IV-V-I and i-bVII-bVI-V progressions.",
        "rhythm": "4/4 driving rhythm. Straight eighth notes. Accent on beats 2 and 4. Palm-muted patterns.",
        "density": "4-10 notes. Distorted texture with octaves. Melody: 4-8 notes per bar with pentatonic/blues emphasis.",
        "bass_range": "28-48", "chord_range": "48-72", "melody_range": "60-84",
    },
    "latin": {
        "chord_voicings": "Montuno patterns with repeated rhythmic figures. Extended chords (maj7, min9). Clave-based voicings.",
        "rhythm": "Clave rhythm (2-3 or 3-2 son clave). Syncopated tumbao bass. 90-130 BPM. Polyrhythmic feel.",
        "density": "6-12 notes per bar. Interlocking patterns between bass and chords. Melody: 4-8 notes per bar.",
        "bass_range": "28-48", "chord_range": "48-76", "melody_range": "60-84",
    },
    "funk": {
        "chord_voicings": "Sharp, percussive chord stabs (often dominant 9th or minor 7th). Wide voicings with 4ths and 2nds.",
        "rhythm": "16th-note-based syncopation. Strong downbeat with ghost notes. 100-120 BPM. Tightly locked bass and drums.",
        "density": "3-8 notes. Staccato chord hits with gaps. Melody: 4-8 notes per bar with rhythmic motifs.",
        "bass_range": "28-45", "chord_range": "48-72", "melody_range": "60-84",
    },
    "soul": {
        "chord_voicings": "Gospel-influenced 7th and 9th chords. Suspended chords resolving to major/minor. Walking bass lines.",
        "rhythm": "Mid-tempo (70-100 BPM). Triplet-based feel. Call-and-response phrasing. Accented backbeats.",
        "density": "6-12 notes. Warm, full-bodied harmonies. Melody: 4-8 notes per bar with expressive phrasing.",
        "bass_range": "28-48", "chord_range": "48-76", "melody_range": "60-84",
    },
}


def _build_system_prompt(genre: str = "", key: str = "", scale: str = "",
                         layer: str = "", extra: str = "") -> str:
    """Build an enhanced, genre-aware system prompt."""
    base = (
        "You are an expert music composer and MIDI file generator. "
        "Generate music as structured MIDI note data.\n\n"
        "Output EXACTLY one note per line in this format:\n"
        "TICK:PITCH:VELOCITY:DURATION\n\n"
        "Rules:\n"
        "- TICK: start time in MIDI ticks (480 per quarter note/beat)\n"
        "- PITCH: MIDI note number 0-127 (60 = middle C)\n"
        "- VELOCITY: 1-127\n"
        "- DURATION: length in ticks\n"
        "- Output ONLY note lines, no other text, comments, or explanations\n"
        "- Use realistic velocities (bass: 70-100, chords: 40-70, melody: 70-120)\n"
        "- Ensure musical coherence with proper harmony and rhythm\n"
    )
    gd = GENRE_DETAILS.get(genre)
    if gd:
        base += (
            "\nGenre-specific guidance for " + genre.upper() + ":\n"
            "- Chord voicings: " + gd['chord_voicings'] + "\n"
            "- Rhythm style: " + gd['rhythm'] + "\n"
            "- Note density: " + gd['density'] + "\n"
        )
        if layer == "bass":
            base += "- THIS IS A BASS LAYER. Only output bass notes in range " + gd['bass_range'] + ".\n"
        elif layer == "chords":
            base += "- THIS IS A CHORDS LAYER. Only output chord/harmony notes in range " + gd['chord_range'] + ".\n"
        elif layer == "melody":
            base += "- THIS IS A MELODY LAYER. Only output melody notes in range " + gd['melody_range'] + ".\n"
    if key and scale:
        scale_notes = build_scale(key, scale)
        if scale_notes:
            note_names_str = ", ".join(NOTE_NAMES[n % 12] + str(n // 12 - 1) for n in scale_notes)
            base += "\nKey/Scale: " + key + " " + scale + ". Valid scale pitches: " + note_names_str + "\n"
            base += "- Prioritize these pitches for melody. Chord tones should come from this scale.\n"
            # Add diatonic chord suggestions
            root = KEY_OFFSETS.get(key, 0)
            scale_intervals = SCALES.get(scale, SCALES["major"])
            chord_degrees = []
            for i, interval in enumerate(scale_intervals):
                pc = (root + interval) % 12
                third_i = (i + 2) % len(scale_intervals)
                fifth_i = (i + 4) % len(scale_intervals)
                third_pc = (root + scale_intervals[third_i]) % 12
                fifth_pc = (root + scale_intervals[fifth_i]) % 12
                is_min = (scale_intervals[third_i] - scale_intervals[i]) in (3, 15)
                chord_degrees.append(f"{NOTE_NAMES[pc]}{'m' if is_min else ''}")
            if chord_degrees:
                base += f"- Diatonic chords in {key} {scale}: {', '.join(chord_degrees[:7])}\n"
    if extra:
        base += "\nAdditional instructions: " + extra + "\n"

    # Add chord progression suggestions
    if genre:
        progs = get_genre_progressions(genre)
        if progs:
            base += "\nSuggested chord progressions: " + "; ".join(progs[:3]) + "\n"

    # Add few-shot example for better format adherence
    genre_key = genre.lower() if genre else "pop"
    if genre_key in FEW_SHOT_EXAMPLES:
        base += "\n" + FEW_SHOT_EXAMPLES[genre_key]

    return base




# ---------------------------------------------------------------------------
# Generation statistics
# ---------------------------------------------------------------------------

@dataclass
class GenerationStats:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    generation_time: float = 0.0
    tokens_per_sec: float = 0.0
    mode: str = "demo"
    note_count: int = 0
    model_name: str = ""
    timestamp: float = 0.0

    @property
    def summary(self) -> str:
        if self.mode == "demo":
            return f"Demo | {self.note_count} notes"
        return (f"LLM ({self.model_name}) | {self.completion_tokens} tokens | "
                f"{self.tokens_per_sec:.1f} tok/s | {self.generation_time:.1f}s | "
                f"{self.note_count} notes")


class GenerationHistory:
    """Tracks the last N generations for undo/recall."""

    def __init__(self, max_entries: int = 10):
        self._entries: List[Tuple[List[MidiNote], GenerationStats]] = []
        self._max = max_entries

    def push(self, notes: List[MidiNote], stats: GenerationStats):
        import copy
        self._entries.append((copy.deepcopy(notes), stats))
        if len(self._entries) > self._max:
            self._entries.pop(0)

    def latest(self):
        return self._entries[-1] if self._entries else None

    def __len__(self):
        return len(self._entries)

    def __getitem__(self, idx):
        return self._entries[idx]


# ---------------------------------------------------------------------------
# LLM MIDI generator
# ---------------------------------------------------------------------------


def analyze_notes(notes: List[MidiNote]) -> dict:
    """Analyze a set of MIDI notes and return statistical summary for context-aware prompts."""
    if not notes:
        return {"note_count": 0}
    pitches = [n.pitch for n in notes]
    velocities = [n.velocity for n in notes]
    durations = [n.duration for n in notes]
    ticks = [n.tick for n in notes]
    max_tick = max(n.tick + n.duration for n in notes)

    pitch_range = f"{note_name(min(pitches))} - {note_name(max(pitches))}"
    avg_vel = sum(velocities) / len(velocities)
    avg_dur = sum(durations) / len(durations)

    # Pitch class histogram
    pc_hist = [0] * 12
    for p in pitches:
        pc_hist[p % 12] += 1
    dominant_pcs = sorted(range(12), key=lambda i: -pc_hist[i])[:5]
    dominant_names = [NOTE_NAMES[i] for i in dominant_pcs if pc_hist[i] > 0]

    # Rhythm analysis
    total_bars = max(1, max_tick // (TICKS_PER_BEAT * 4))
    notes_per_bar = len(notes) / total_bars
    unique_durs = sorted(set(round(d / (TICKS_PER_BEAT // 4)) * (TICKS_PER_BEAT // 4) for d in durations))

    # Density: how many simultaneous notes at peak
    time_slots = {}
    for n in notes:
        for t in range(n.tick, min(n.tick + n.duration, n.tick + TICKS_PER_BEAT), TICKS_PER_BEAT // 4):
            time_slots[t] = time_slots.get(t, 0) + 1
    max_simultaneous = max(time_slots.values()) if time_slots else 0

    # Estimate genre feel
    is_dense = notes_per_bar > 12
    is_sparse = notes_per_bar < 4
    has_wide_range = (max(pitches) - min(pitches)) > 24
    is_high_vel = avg_vel > 90

    # Detect likely chord roots from pitch class histogram
    chord_roots = []
    for pc in dominant_pcs:
        if pc_hist[pc] > len(notes) * 0.05:
            root_name = NOTE_NAMES[pc]
            third_pc = (pc + 4) % 12
            fifth_pc = (pc + 7) % 12
            is_major = pc_hist[third_pc] > len(notes) * 0.02 and pc_hist[fifth_pc] > len(notes) * 0.02
            third_min = (pc + 3) % 12
            is_minor = pc_hist[third_min] > len(notes) * 0.02 and pc_hist[fifth_pc] > len(notes) * 0.02
            if is_major:
                chord_roots.append(root_name + "maj")
            elif is_minor:
                chord_roots.append(root_name + "min")
            else:
                chord_roots.append(root_name)

    # Detect time signature feel
    strong_beats = sum(1 for n in notes if n.tick % (TICKS_PER_BEAT * 4) < TICKS_PER_BEAT)
    weak_beats = len(notes) - strong_beats
    time_sig_feel = "4/4" if strong_beats > weak_beats * 0.5 else "irregular"

    return {
        "note_count": len(notes),
        "pitch_range": pitch_range,
        "min_pitch": min(pitches), "max_pitch": max(pitches),
        "avg_velocity": round(avg_vel, 1),
        "min_velocity": min(velocities), "max_velocity": max(velocities),
        "avg_duration_ticks": round(avg_dur),
        "total_bars": total_bars,
        "notes_per_bar": round(notes_per_bar, 1),
        "dominant_pitch_classes": dominant_names,
        "max_simultaneous": max_simultaneous,
        "unique_rhythm_values": len(unique_durs),
        "density_profile": "dense" if is_dense else ("sparse" if is_sparse else "moderate"),
        "has_wide_range": has_wide_range,
        "is_high_energy": is_high_vel,
        "total_ticks": max_tick,
        "detected_chords": chord_roots[:6],
        "time_sig_feel": time_sig_feel,
    }


# Adjust mode definitions
ADJUST_MODES = {
    "full": {
        "label": "Full Arrange",
        "desc": "Full AI rearrangement of all aspects",
        "system_extra": "",
    },
    "melody": {
        "label": "Melody",
        "desc": "Restructure melodic contour, add passing tones",
        "system_extra": (
            "FOCUS ON MELODY ADJUSTMENT. Keep the same rhythm and harmony structure. "
            "Only modify pitches to create a better melodic contour. "
            "Use stepwise motion primarily, with occasional leaps. "
            "Aim for smooth voice leading with no awkward intervals."
        ),
    },
    "harmony": {
        "label": "Harmony",
        "desc": "Reharmonize with richer chord voicings",
        "system_extra": (
            "FOCUS ON HARMONY ADJUSTMENT. Restructure the harmonic content. "
            "Add chord extensions (7ths, 9ths) where appropriate. "
            "Improve voice leading between chords. Keep the overall rhythm similar."
        ),
    },
    "rhythm": {
        "label": "Rhythm",
        "desc": "Adjust timing, add syncopation, swing",
        "system_extra": (
            "FOCUS ON RHYTHM ADJUSTMENT. Modify note start times and durations. "
            "Keep the same pitches. Add rhythmic interest through syncopation, "
            "varied note lengths, and rhythmic displacement. Maintain groove."
        ),
    },
    "velocity": {
        "label": "Velocity",
        "desc": "Humanize dynamics and accent patterns",
        "system_extra": (
            "FOCUS ON VELOCITY/DYNAMICS ADJUSTMENT. Keep pitches and timing the same. "
            "Only modify velocity values. Create natural dynamics: "
            "accent downbeats, crescendo/decrescendo phrases, "
            "humanize with subtle velocity variations (+/-5-15)."
        ),
    },
    "density": {
        "label": "Density",
        "desc": "Add or remove notes to change texture",
        "system_extra": (
            "FOCUS ON NOTE DENSITY ADJUSTMENT. Add passing notes, fills, or "
            "remove redundant notes to change the texture. "
            "You may add new notes or remove existing ones. "
            "Maintain musical coherence."
        ),
    },
}

# Few-shot examples for generation
FEW_SHOT_EXAMPLES = {
    "pop": (
        "Example (pop in C major, 2 bars, bass+chords+melody):\n"
        "0:48:90:960\n0:60:50:960\n0:64:45:960\n0:67:45:960\n"
        "960:72:95:480\n1440:74:90:240\n1680:76:100:240\n"
        "1920:48:85:960\n1920:59:50:960\n1920:62:45:960\n1920:67:45:960\n"
        "2160:79:105:480\n2640:77:85:480\n"
    ),
    "jazz": (
        "Example (jazz chord voicings, 1 bar):\n"
        "0:60:55:960\n0:64:50:960\n0:67:50:960\n0:71:45:960\n"
        "960:65:55:960\n960:69:50:960\n960:72:50:960\n960:76:45:960\n"
    ),
    "blues": (
        "Example (blues in A, 1 bar):\n"
        "0:45:85:480\n0:57:50:960\n0:60:45:960\n0:64:40:960\n"
        "480:49:80:240\n720:52:90:480\n"
        "960:45:85:480\n960:57:50:480\n1440:53:95:240\n"
    ),
    "lofi": (
        "Example (lofi chords, 2 bars):\n"
        "0:60:40:1920\n0:67:35:1920\n0:74:30:1920\n0:79:25:1920\n"
        "1920:65:40:1920\n1920:69:35:1920\n1920:72:30:1920\n1920:76:25:1920\n"
    ),
    "edm": (
        "Example (edm bass+lead, 1 bar):\n"
        "0:36:100:120\n120:36:90:120\n240:36:100:120\n"
        "360:36:80:120\n480:36:100:120\n600:36:90:120\n"
        "720:72:95:480\n1200:76:100:240\n1440:79:110:480\n"
    ),
    "classical": (
        "Example (classical arpeggiated, 1 bar):\n"
        "0:60:70:240\n240:64:65:240\n480:67:60:240\n720:72:70:240\n"
        "960:76:65:240\n1200:72:60:240\n1440:67:55:240\n"
        "1680:64:60:240\n\n"
    ),
    "reggaeton": (
        "Example (reggaeton dembow rhythm, 1 bar):\n"
        "0:36:95:120\n0:48:40:240\n120:36:85:120\n"
        "240:36:90:120\n360:36:85:120\n480:36:95:120\n"
        "0:67:80:240\n960:67:80:240\n"
    ),
    "rnb": (
        "Example (rnb neo-soul, 2 bars):\n"
        "0:60:40:1920\n0:64:35:1920\n0:67:30:1920\n0:71:25:1920\n"
        "0:76:55:480\n480:74:50:480\n960:72:60:480\n"
        "1920:60:40:1920\n1920:65:35:1920\n1920:69:30:1920\n1920:72:25:1920\n"
        "1920:79:55:480\n2400:77:50:480\n2880:76:60:480\n"
    ),
    "rock": (
        "Example (rock power chords, 1 bar):\n"
        "0:48:100:960\n0:55:95:960\n960:45:100:480\n"
        "1440:43:95:480\n1920:48:100:960\n1920:55:95:960\n"
    ),
    "funk": (
        "Example (funk stabs, 1 bar):\n"
        "0:60:80:120\n0:64:70:120\n0:67:65:120\n0:71:60:120\n"
        "240:36:90:240\n480:60:80:120\n480:64:70:120\n480:67:65:120\n"
        "720:36:85:240\n960:60:80:120\n960:64:70:120\n960:67:65:120\n"
    ),
    "soul": (
        "Example (soul gospel, 1 bar):\n"
        "0:60:45:960\n0:64:40:960\n0:67:35:960\n0:71:30:960\n"
        "960:62:45:960\n960:65:40:960\n960:69:35:960\n960:72:30:960\n"
        "0:72:70:480\n480:74:65:480\n960:76:70:480\n1440:79:65:480\n"
    ),
}


# ---------------------------------------------------------------------------
# Note utilities
# ---------------------------------------------------------------------------

def merge_notes(base_notes: List[MidiNote], new_notes: List[MidiNote],
                 replace_mode: str = "extend") -> List[MidiNote]:
    """Merge new_notes into base_notes.
    
    Modes:
        extend  — append new_notes after the last tick of base_notes (shifted)
        overlay — add new_notes at their original tick positions
        replace — replace notes in the same tick range, keep the rest
    """
    if replace_mode == "extend" and base_notes and new_notes:
        last_tick = max(n.tick + n.duration for n in base_notes)
        shifted = [MidiNote(tick=n.tick + last_tick, pitch=n.pitch,
                            velocity=n.velocity, duration=n.duration)
                   for n in new_notes]
        return base_notes + shifted
    elif replace_mode == "replace" and base_notes and new_notes:
        new_min = min(n.tick for n in new_notes)
        new_max = max(n.tick + n.duration for n in new_notes)
        kept = [n for n in base_notes
                if n.tick + n.duration <= new_min or n.tick >= new_max]
        return kept + new_notes
    else:  # overlay
        return base_notes + new_notes


def separate_layers(notes: List[MidiNote]) -> dict:
    """Separate notes into bass, chords, and melody layers by pitch range."""
    bass = [n for n in notes if n.pitch < 48]
    chords = [n for n in notes if 48 <= n.pitch < 60]
    melody = [n for n in notes if n.pitch >= 60]
    return {"bass": bass, "chords": chords, "melody": melody}


def notes_to_text(notes: List[MidiNote], max_notes: int = 80) -> str:
    """Convert notes to the TICK:PITCH:VELOCITY:DURATION text format for prompts."""
    selected = notes[-max_notes:] if len(notes) > max_notes else notes
    return "\n".join(f"{n.tick}:{n.pitch}:{n.velocity}:{n.duration}" for n in selected)


# ---------------------------------------------------------------------------
# Chord progression suggestions by genre
# ---------------------------------------------------------------------------

CHORD_PROGRESSION_SUGGESTIONS = {
    "pop": ["I-V-vi-IV (C-G-Am-F)", "I-IV-V-I (C-F-G-C)", "vi-IV-I-V (Am-F-C-G)"],
    "jazz": ["ii-V-I (Dm7-G7-Cmaj7)", "I-vi-ii-V (Cmaj7-Am7-Dm7-G7)", "iii-vi-ii-V (Em7-Am7-Dm7-G7)"],
    "blues": ["I7-I7-I7-I7 (12-bar blues)", "I7-IV7-I7-V7", "i7-bVI7-bVII7-IV7 (minor blues)"],
    "lofi": ["Imaj7-iv7-VII3-bVIImaj7", "iim7-V7-Imaj7-vim7", "IVmaj7-vii7b5-iiim7-VI7"],
    "edm": ["i-VI-III-VII (Am-F-C-G)", "i-iv-VI-VII (Am-Dm-F-G)", "I-V-vi-IV (power chords)"],
    "classical": ["I-IV-V-I", "I-V-vi-IV-I-V-ii-V", "i-iv-V-i (minor)"],
    "ambient": ["I-IV (sustained)", "I-vi-IV-V (slow)", "i-bVII-bVI-V (ethereal)"],
    "hiphop": ["i-iv-V-i", "i-bVI-bVII-i", "I-V-vi-IV (sampled)"],
    "reggaeton": ["i-iv-V-i (minor)", "i-bVII-bVI-V", "i-iv-bVI-V"],
    "rnb": ["Imaj7-viim7b5-iiim7-VI7", "ii7-V7-Imaj7-vi7", "IVmaj7-iv7-iiim7-VI7"],
    "rock": ["I-IV-V-I (power)", "i-bVII-bVI-V (classic)", "I-V-vi-IV (pop rock)"],
    "latin": ["i-IV-V-i (montuno)", "II7-V7-I-IV (salsa)", "i-bIII-bVII-IV (bossa)"],
    "funk": ["i7-IV7-vi7-V7", "I9-IV9-vii7b5-I9", "i7-bVII7-IV7-V7"],
    "soul": ["I7-IV7-I7-V7 (gospel)", "ii7-V7-Imaj7-vi7", "I-IV-I-V ( Amen)"],
}


def get_genre_progressions(genre: str) -> List[str]:
    """Return suggested chord progressions for a genre."""
    return CHORD_PROGRESSION_SUGGESTIONS.get(genre.lower(), CHORD_PROGRESSION_SUGGESTIONS["pop"])


class MidiLLMGenerator:
    """LLM-powered MIDI generator with multi-template, streaming, and history support."""

    def __init__(self, model_path: Optional[str] = None):
        self.model: Optional[Llama] = None
        self.model_path = model_path or os.environ.get("LLM_MODEL_PATH", "")
        self._lock = threading.Lock()
        self._loading = False
        self._load_error = ""
        self._abort = False

        # Prompt template
        self._template_name = "LlamaChat"
        self._custom_system_prompt = ""
        self._model_name_hint = ""

        # Streaming
        self._stream_callback: Optional[Callable[[str], None]] = None

        # Stats & history
        self.last_stats: Optional[GenerationStats] = None
        self.history = GenerationHistory()

        # Chat history for multi-turn
        self._chat_messages: List[Dict[str, str]] = []

        # Sampling parameters
        self.top_k: int = 40
        self.top_p: float = 0.95
        self.repeat_penalty: float = 1.1
        self.frequency_penalty: float = 0.0
        self.presence_penalty: float = 0.0
        self.min_p: float = 0.05
        self.max_tokens: int = 4096
        self.n_ctx: int = 4096
        self.n_threads: int = os.cpu_count() or 4

        # Current notes context for awareness
        self._current_notes: List[MidiNote] = []

    # -- Properties --

    @property
    def is_available(self) -> bool:
        if not LLAMA_AVAILABLE:
            return False
        if not self.model_path or not os.path.isfile(self.model_path):
            return False
        return True

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    @property
    def status_text(self) -> str:
        if not LLAMA_AVAILABLE:
            return "llama-cpp-python not installed"
        if self._loading:
            return "Loading model..."
        if self.model is not None:
            return f"Loaded: {os.path.basename(self.model_path)} ({self._template_name})"
        if self.model_path and os.path.isfile(self.model_path):
            return f"Ready to load: {os.path.basename(self.model_path)}"
        if self.model_path:
            return f"File not found: {self.model_path}"
        return "No model path set"

    @property
    def template_name(self) -> str:
        return self._template_name

    @template_name.setter
    def template_name(self, name: str):
        if name in PROMPT_TEMPLATES:
            self._template_name = name

    @property
    def model_name(self) -> str:
        if self._model_name_hint:
            return self._model_name_hint
        return os.path.basename(self.model_path) if self.model_path else "unknown"

    # -- Model management --

    def load_model(self, path: Optional[str] = None) -> bool:
        if path:
            self.model_path = path
            self._auto_detect_template()
        if not LLAMA_AVAILABLE:
            self._load_error = "llama-cpp-python is not installed"
            return False
        if not self.model_path or not os.path.isfile(self.model_path):
            self._load_error = "Model file not found"
            return False
        with self._lock:
            if self.model is not None:
                return True
            self._loading = True
            self._load_error = ""
            try:
                from llama_cpp import Llama as _Llama
                self.model = _Llama(
                    model_path=self.model_path,
                    n_ctx=self.n_ctx,
                    n_threads=self.n_threads,
                    verbose=False,
                )
                self._model_name_hint = ""
                try:
                    metadata = getattr(self.model, 'model_meta', None)
                    if metadata and hasattr(metadata, 'get'):
                        name = metadata.get('general.name', '') or metadata.get('name', '')
                        if name:
                            self._model_name_hint = str(name)
                except Exception:
                    pass
                if not self._model_name_hint:
                    self._model_name_hint = os.path.basename(self.model_path)
                self._auto_detect_template()
                print(f"[MidiLLM] Loaded model: {self.model_path} (template: {self._template_name})")
                self._load_error = ""
                return True
            except Exception as e:
                print(f"[MidiLLM] Failed to load model: {e}")
                self._load_error = str(e)
                self.model = None
                return False
            finally:
                self._loading = False

    def unload_model(self):
        with self._lock:
            if self.model is not None:
                del self.model
                self.model = None
                import gc
                gc.collect()
                print("[MidiLLM] Model unloaded")

    def _auto_detect_template(self):
        name = os.path.basename(self.model_path).lower() if self.model_path else ""
        # Check ChatML first (more specific: qwen2, gemma, yi-, deepseek)
        if PROMPT_TEMPLATES["ChatML"]["detect"](name):
            self._template_name = "ChatML"
            return
        # Then LlamaChat (broader: llama, tinyllama, mistral, phi, qwen)
        if PROMPT_TEMPLATES["LlamaChat"]["detect"](name):
            self._template_name = "LlamaChat"
            return
        self._template_name = "LlamaChat"

    # -- Prompt building --

    def _build_prompt(self, system_prompt: str, user_msg: str) -> str:
        tmpl = PROMPT_TEMPLATES[self._template_name]
        return tmpl["wrap"](system_prompt, user_msg)

    def _get_stop_tokens(self) -> list:
        return PROMPT_TEMPLATES[self._template_name]["stop"]

    # -- Core generation with streaming --

    def _generate_text(self, full_prompt: str, temperature: float = 0.8,
                       max_tokens: int = 4096) -> Tuple[str, dict]:
        """Generate text with streaming support. Returns (text, usage_stats)."""
        if self._stream_callback:
            self._stream_callback("")
        collected = []
        usage = {"prompt_tokens": 0, "completion_tokens": 0}
        t0 = time.time()

        try:
            gen_kwargs = dict(
                max_tokens=max(min(max_tokens, self.max_tokens), 64),
                temperature=temperature,
                top_k=self.top_k, top_p=self.top_p,
                repeat_penalty=self.repeat_penalty,
                stop=self._get_stop_tokens(), echo=False, stream=True,
            )
            # Add optional sampling params if supported by llama-cpp-python
            try:
                streamer = self.model(full_prompt, **gen_kwargs)
            except TypeError:
                # Older llama-cpp-python may not support all params; strip extras
                for k in ("frequency_penalty", "presence_penalty", "min_p"):
                    gen_kwargs.pop(k, None)
                streamer = self.model(full_prompt, **gen_kwargs)
            for chunk in streamer:
                if self._abort:
                    break
                text = chunk["choices"][0]["text"]
                collected.append(text)
                if self._stream_callback:
                    self._stream_callback("".join(collected))
                if "usage" in chunk:
                    usage["prompt_tokens"] = chunk["usage"].get("prompt_tokens", 0)
                    usage["completion_tokens"] = chunk["usage"].get("completion_tokens", 0)
        except TypeError:
            fb_kwargs = dict(
                max_tokens=max(min(max_tokens, self.max_tokens), 64),
                temperature=temperature,
                top_k=self.top_k, top_p=self.top_p,
                repeat_penalty=self.repeat_penalty,
                stop=self._get_stop_tokens(), echo=False,
            )
            for k in ("frequency_penalty", "presence_penalty", "min_p"):
                fb_kwargs.pop(k, None)
            output = self.model(full_prompt, **fb_kwargs)
            collected = [output["choices"][0]["text"]]
            if self._stream_callback:
                self._stream_callback(collected[0])
            if "usage" in output:
                usage["prompt_tokens"] = output["usage"].get("prompt_tokens", 0)
                usage["completion_tokens"] = output["usage"].get("completion_tokens", 0)
        except Exception as e:
            print(f"[MidiLLM] Generation stream error: {e}")
            return "", usage

        elapsed = time.time() - t0
        text = "".join(collected)
        if not usage["completion_tokens"]:
            usage["completion_tokens"] = len(text) // 4
        usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
        return text, usage

    # -- Public API --

    def generate(self, prompt: str = "", genre: str = "pop", key: str = "C",
                 scale: str = "major", tempo: int = 120, num_bars: int = 4,
                 temperature: float = 0.8, layer: str = "",
                 stream_cb: Optional[Callable] = None) -> List[MidiNote]:
        """Generate MIDI notes. Returns list of MidiNote."""
        self._abort = False
        self._stream_callback = stream_cb

        if not self.is_loaded:
            if not self.load_model():
                notes = generate_demo_notes(genre, key, scale, tempo, num_bars)
                self.last_stats = GenerationStats(
                    mode="demo", note_count=len(notes),
                    timestamp=time.time(), model_name="demo")
                self.history.push(notes, self.last_stats)
                return notes

        sys_prompt = self._custom_system_prompt or _build_system_prompt(
            genre=genre, key=key, scale=scale, layer=layer, extra=prompt)
        # Build context from existing notes if continuing
        context_str = ""
        if hasattr(self, '_current_notes') and self._current_notes:
            analysis = analyze_notes(self._current_notes)
            context_str = (
                f"\nCurrent project context:\n"
                f"- {analysis['note_count']} notes, {analysis['total_bars']} bars\n"
                f"- Pitch range: {analysis['pitch_range']}\n"
                f"- Avg velocity: {analysis['avg_velocity']}, Density: {analysis['density_profile']}\n"
                f"- Dominant pitches: {', '.join(analysis['dominant_pitch_classes'][:4])}\n"
            )

        # Add chord progression context
        prog_hint = ""
        progs = get_genre_progressions(genre)
        if progs:
            prog_hint = f" Consider using: {progs[0]}.\n"

        user_msg = (
            f"Generate {num_bars} bars of {genre} music in the key of {key} "
            f"({scale} scale) at {tempo} BPM.\n"
            f"{'Layer: ' + layer + '. ' if layer else ''}"
            f"{context_str}"
            f"{prog_hint}"
            f"{prompt}\n\nOutput notes:\n"
        )
        full_prompt = self._build_prompt(sys_prompt, user_msg)

        t0 = time.time()
        with self._lock:
            try:
                text, usage = self._generate_text(full_prompt, temperature)
            except Exception as e:
                print(f"[MidiLLM] Generation error: {e}")
                notes = generate_demo_notes(genre, key, scale, tempo, num_bars)
                self.last_stats = GenerationStats(
                    mode="demo", note_count=len(notes),
                    timestamp=time.time(), model_name="demo")
                self.history.push(notes, self.last_stats)
                return notes

        notes = self._parse(text)
        if not notes:
            notes = generate_demo_notes(genre, key, scale, tempo, num_bars)
            mode = "demo"
        else:
            mode = "llm"

        elapsed = time.time() - t0
        self.last_stats = GenerationStats(
            prompt_tokens=usage["prompt_tokens"],
            completion_tokens=usage["completion_tokens"],
            total_tokens=usage["total_tokens"],
            generation_time=elapsed,
            tokens_per_sec=usage["completion_tokens"] / max(elapsed, 0.001),
            mode=mode, note_count=len(notes),
            model_name=self.model_name, timestamp=time.time())
        self.history.push(notes, self.last_stats)
        return notes

    def generate_continue(self, existing_notes: List[MidiNote], prompt: str = "",
                          genre: str = "pop", key: str = "C", scale: str = "major",
                          tempo: int = 120, num_bars: int = 4,
                          temperature: float = 0.8,
                          stream_cb: Optional[Callable] = None) -> List[MidiNote]:
        """Continue generation from existing notes."""
        self._abort = False
        self._stream_callback = stream_cb

        if not self.is_loaded:
            if not self.load_model():
                return []

        if existing_notes:
            last_tick = max(n.tick + n.duration for n in existing_notes)
        else:
            last_tick = 0

        sys_prompt = self._custom_system_prompt or _build_system_prompt(
            genre=genre, key=key, scale=scale, extra=prompt)
        lines = "\n".join(f"{n.tick}:{n.pitch}:{n.velocity}:{n.duration}" for n in existing_notes[-60:])
        user_msg = (
            f"Here are existing MIDI notes (last 60 shown):\n{lines}\n\n"
            f"Continue the music for {num_bars} more bars starting at tick {last_tick}. "
            f"Genre: {genre}, Key: {key} {scale}, Tempo: {tempo} BPM.\n"
            f"{prompt}\n\nNew notes (starting from tick {last_tick}):\n"
        )
        full_prompt = self._build_prompt(sys_prompt, user_msg)

        t0 = time.time()
        with self._lock:
            try:
                text, usage = self._generate_text(full_prompt, temperature)
            except Exception as e:
                print(f"[MidiLLM] Continue error: {e}")
                return []

        new_notes = self._parse(text)
        elapsed = time.time() - t0
        self.last_stats = GenerationStats(
            prompt_tokens=usage["prompt_tokens"],
            completion_tokens=usage["completion_tokens"],
            total_tokens=usage["total_tokens"],
            generation_time=elapsed,
            tokens_per_sec=usage["completion_tokens"] / max(elapsed, 0.001),
            mode="llm", note_count=len(new_notes),
            model_name=self.model_name, timestamp=time.time())
        return new_notes

    def generate_variation(self, genre: str = "pop", key: str = "C",
                             scale: str = "major", tempo: int = 120, num_bars: int = 4,
                             temperature: float = 0.8,
                             stream_cb: Optional[Callable] = None) -> List[MidiNote]:
        """Generate a variation of the last generation with slightly shifted params."""
        if not self.history:
            return self.generate(genre=genre, key=key, scale=scale, tempo=tempo,
                                num_bars=num_bars, temperature=temperature,
                                stream_cb=stream_cb)
        varied_temp = max(0.3, min(1.5, temperature + random.uniform(-0.15, 0.15)))
        return self.generate(genre=genre, key=key, scale=scale, tempo=tempo,
                            num_bars=num_bars, temperature=varied_temp,
                            stream_cb=stream_cb)

    def generate_arrangement(self, melody_notes: List[MidiNote],
                             genre: str = "pop", key: str = "C", scale: str = "major",
                             tempo: int = 120, temperature: float = 0.7,
                             stream_cb: Optional[Callable] = None) -> List[MidiNote]:
        """Generate bass and chords to accompany a given melody."""
        self._abort = False
        self._stream_callback = stream_cb

        if not melody_notes:
            return []

        if not self.is_loaded:
            if not self.load_model():
                return []

        analysis = analyze_notes(melody_notes)
        melody_lines = "\n".join(
            f"{n.tick}:{n.pitch}:{n.velocity}:{n.duration}" for n in melody_notes[-80:])

        gd = GENRE_DETAILS.get(genre, GENRE_DETAILS["pop"])
        sys_prompt = _build_system_prompt(
            genre=genre, key=key, scale=scale, layer="",
            extra="Generate BASS and CHORD accompaniment for the given melody. "
                  "Output ONLY bass and chord notes. Do NOT repeat the melody notes.")

        user_msg = (
            f"Here is a melody ({analysis['note_count']} notes, "
            f"range {analysis['pitch_range']}):\n{melody_lines}\n\n"
            f"Generate bass notes (range {gd['bass_range']}) "
            f"and chord/harmony notes (range {gd['chord_range']}) "
            f"that accompany this melody.\n"
            f"Genre: {genre}, Key: {key} {scale}, Tempo: {tempo} BPM.\n"
            f"Detected chords: {', '.join(analysis.get('detected_chords', ['unknown'])[:4])}\n\n"
            f"Output bass and chord notes in TICK:PITCH:VELOCITY:DURATION format:\n"
        )
        full_prompt = self._build_prompt(sys_prompt, user_msg)

        t0 = time.time()
        with self._lock:
            try:
                text, usage = self._generate_text(full_prompt, temperature)
            except Exception as e:
                print(f"[MidiLLM] Arrangement error: {e}")
                return []

        accompaniment = self._parse(text)
        elapsed = time.time() - t0
        self.last_stats = GenerationStats(
            prompt_tokens=usage["prompt_tokens"],
            completion_tokens=usage["completion_tokens"],
            total_tokens=usage["total_tokens"],
            generation_time=elapsed,
            tokens_per_sec=usage["completion_tokens"] / max(elapsed, 0.001),
            mode="llm", note_count=len(accompaniment),
            model_name=self.model_name, timestamp=time.time())
        self.history.push(accompaniment, self.last_stats)
        return accompaniment

    def generate_melody_from_chords(self, chord_notes: List[MidiNote],
                                     genre: str = "pop", key: str = "C", scale: str = "major",
                                     tempo: int = 120, temperature: float = 0.8,
                                     stream_cb: Optional[Callable] = None) -> List[MidiNote]:
        """Generate a melody that harmonically fits over existing chord notes."""
        self._abort = False
        self._stream_callback = stream_cb

        if not chord_notes:
            return []

        if not self.is_loaded:
            if not self.load_model():
                return []

        analysis = analyze_notes(chord_notes)
        chord_text = notes_to_text(chord_notes, 100)

        gd = GENRE_DETAILS.get(genre, GENRE_DETAILS["pop"])
        sys_prompt = _build_system_prompt(
            genre=genre, key=key, scale=scale, layer="melody",
            extra="Generate a MELODY that harmonically fits over the given chords. "
                  "Output ONLY melody notes. Do NOT repeat the chord notes. "
                  "Use chord tones on strong beats and passing tones on weak beats. "
                  "Create a singable, memorable melody with good contour.")

        user_msg = (
            f"Here are the chord/harmony notes ({analysis['note_count']} notes, "
            f"range {analysis['pitch_range']}):\n{chord_text}\n\n"
            f"Generate a melody (range {gd['melody_range']}) that fits these chords.\n"
            f"Genre: {genre}, Key: {key} {scale}, Tempo: {tempo} BPM.\n"
            f"Detected chords: {', '.join(analysis.get('detected_chords', ['unknown'])[:4])}\n\n"
            f"Output melody notes in TICK:PITCH:VELOCITY:DURATION format:\n"
        )
        full_prompt = self._build_prompt(sys_prompt, user_msg)

        t0 = time.time()
        with self._lock:
            try:
                text, usage = self._generate_text(full_prompt, temperature)
            except Exception as e:
                print(f"[MidiLLM] Melody from chords error: {e}")
                return []

        melody = self._parse(text)
        elapsed = time.time() - t0
        self.last_stats = GenerationStats(
            prompt_tokens=usage["prompt_tokens"],
            completion_tokens=usage["completion_tokens"],
            total_tokens=usage["total_tokens"],
            generation_time=elapsed,
            tokens_per_sec=usage["completion_tokens"] / max(elapsed, 0.001),
            mode="llm", note_count=len(melody),
            model_name=self.model_name, timestamp=time.time())
        self.history.push(melody, self.last_stats)
        return melody

    def generate_full_arrangement(self, seed_notes: List[MidiNote],
                                   genre: str = "pop", key: str = "C", scale: str = "major",
                                   tempo: int = 120, temperature: float = 0.75,
                                   stream_cb: Optional[Callable] = None) -> List[MidiNote]:
        """Generate a complete arrangement (bass + chords + melody) from seed notes.
        If seed has only melody, generates bass+chords. If chords, generates melody.
        If mixed, generates complementary parts for missing layers."""
        self._abort = False
        self._stream_callback = stream_cb

        if not seed_notes:
            return []

        if not self.is_loaded:
            if not self.load_model():
                return []

        layers = separate_layers(seed_notes)
        analysis = analyze_notes(seed_notes)
        gd = GENRE_DETAILS.get(genre, GENRE_DETAILS["pop"])
        seed_text = notes_to_text(seed_notes, 80)

        # Determine what we need to generate
        needs_bass = len(layers["bass"]) < 4
        needs_chords = len(layers["chords"]) < 4
        needs_melody = len(layers["melody"]) < 4

        parts_to_gen = []
        if needs_bass:
            parts_to_gen.append(f"bass notes in range {gd['bass_range']}")
        if needs_chords:
            parts_to_gen.append(f"chord/harmony notes in range {gd['chord_range']}")
        if needs_melody:
            parts_to_gen.append(f"melody notes in range {gd['melody_range']}")

        if not parts_to_gen:
            # All layers exist — do a full regen
            parts_to_gen = [f"bass ({gd['bass_range']})", f"chords ({gd['chord_range']})", f"melody ({gd['melody_range']})"]

        sys_prompt = _build_system_prompt(
            genre=genre, key=key, scale=scale,
            extra=f"Generate a COMPLETE arrangement with {', '.join(parts_to_gen)}. "
                  f"Build upon the existing musical ideas. Make it sound full and professional. "
                  f"Ensure bass supports the harmony, chords fill the mid-range, "
                  f"and the melody is singable and memorable.")

        user_msg = (
            f"Here are existing musical ideas ({analysis['note_count']} notes, "
            f"range {analysis['pitch_range']}):\n{seed_text}\n\n"
            f"Generate {', '.join(parts_to_gen)} to complete the arrangement.\n"
            f"Genre: {genre}, Key: {key} {scale}, Tempo: {tempo} BPM.\n"
            f"Detected chords: {', '.join(analysis.get('detected_chords', ['unknown'])[:4])}\n\n"
            f"Output ALL notes (existing + new) in TICK:PITCH:VELOCITY:DURATION format:\n"
        )
        full_prompt = self._build_prompt(sys_prompt, user_msg)

        t0 = time.time()
        with self._lock:
            try:
                text, usage = self._generate_text(full_prompt, temperature)
            except Exception as e:
                print(f"[MidiLLM] Full arrangement error: {e}")
                return []

        arranged = self._parse(text)
        elapsed = time.time() - t0
        self.last_stats = GenerationStats(
            prompt_tokens=usage["prompt_tokens"],
            completion_tokens=usage["completion_tokens"],
            total_tokens=usage["total_tokens"],
            generation_time=elapsed,
            tokens_per_sec=usage["completion_tokens"] / max(elapsed, 0.001),
            mode="llm", note_count=len(arranged),
            model_name=self.model_name, timestamp=time.time())
        self.history.push(arranged, self.last_stats)
        return arranged

    def ai_adjust(self, notes: List[MidiNote], instruction: str,
                  temperature: float = 0.7, mode: str = "full",
                  stream_cb: Optional[Callable] = None) -> List[MidiNote]:
        """AI-adjust selected notes based on instruction and mode."""
        self._abort = False
        self._stream_callback = stream_cb

        if not notes:
            return []

        if not self.is_loaded:
            if not self.load_model():
                return self._demo_adjust(notes, instruction, mode)

        # Build mode-aware system prompt
        mode_info = ADJUST_MODES.get(mode, ADJUST_MODES["full"])
        base_sys = self._custom_system_prompt or _build_system_prompt()
        if mode_info.get("system_extra"):
            base_sys += "\n" + mode_info["system_extra"]

        # Analyze notes for context
        analysis = analyze_notes(notes)
        lines = "\n".join(f"{n.tick}:{n.pitch}:{n.velocity}:{n.duration}" for n in notes)
        user_msg = (
            f"Here are {len(notes)} MIDI notes to adjust:\n{lines}\n\n"
            f"Note analysis: {analysis['note_count']} notes, pitch range {analysis['pitch_range']}, "
            f"avg velocity {analysis['avg_velocity']}, {analysis['total_bars']} bars, "
            f"density: {analysis['density_profile']}.\n\n"
            f"Adjustment mode: {mode_info['label']}. "
            f"Instruction: {instruction}\n\n"
            f"Output ALL adjusted notes in TICK:PITCH:VELOCITY:DURATION format "
            f"(one per line, no other text):"
        )
        full_prompt = self._build_prompt(base_sys, user_msg)

        t0 = time.time()
        with self._lock:
            try:
                text, usage = self._generate_text(full_prompt, temperature)
            except Exception as e:
                print(f"[MidiLLM] Adjust error: {e}")
                return self._demo_adjust(notes, instruction)

        parsed = self._parse(text)
        elapsed = time.time() - t0
        self.last_stats = GenerationStats(
            prompt_tokens=usage["prompt_tokens"],
            completion_tokens=usage["completion_tokens"],
            total_tokens=usage["total_tokens"],
            generation_time=elapsed,
            tokens_per_sec=usage["completion_tokens"] / max(elapsed, 0.001),
            mode="llm", note_count=len(parsed) if parsed else len(notes),
            model_name=self.model_name, timestamp=time.time())
        return parsed if parsed else self._demo_adjust(notes, instruction)

    def chat(self, message: str, stream_cb: Optional[Callable] = None,
             context: str = "") -> str:
        """Multi-turn chat with the LLM about music."""
        if not self.is_loaded:
            if not self.load_model():
                return "No LLM model loaded. Please load a model first."

        self._abort = False
        self._stream_callback = stream_cb
        self._chat_messages.append({"role": "user", "content": message})

        sys_prompt = (
            "You are a helpful music theory and composition assistant. "
            "You help users with chord progressions, scales, melody writing, "
            "arrangement tips, genre characteristics, and MIDI production advice. "
            "Be concise and practical. Use music terminology accurately. "
            "When the user asks about their current project, reference the context provided."
        )
        if context:
            sys_prompt += "\n\nCurrent project context:\n" + context

        tmpl = PROMPT_TEMPLATES[self._template_name]
        if self._template_name == "ChatML":
            parts = [f"<|im_start|>system\n{sys_prompt}<|im_end|>"]
            for msg in self._chat_messages:
                parts.append(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>")
            parts.append("<|im_start|>assistant\n")
            full_prompt = "\n".join(parts)
            stop = ["<|im_end|>", "<|im_start|>"]
        else:
            history_text = "\n".join(
                f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
                for m in self._chat_messages
            )
            full_prompt = tmpl["wrap"](sys_prompt, history_text + "\n\nAssistant:")
            stop = self._get_stop_tokens()

        t0 = time.time()
        with self._lock:
            try:
                # Override stop tokens for chat
                old_stop = self._get_stop_tokens
                self._get_stop_tokens = lambda: stop
                text, usage = self._generate_text(full_prompt, 0.7, max_tokens=2048)
                self._get_stop_tokens = old_stop
            except Exception as e:
                self._get_stop_tokens = old_stop
                self._chat_messages.pop()
                return f"Error: {e}"

        self._chat_messages.append({"role": "assistant", "content": text})
        elapsed = time.time() - t0
        self.last_stats = GenerationStats(
            prompt_tokens=usage["prompt_tokens"],
            completion_tokens=usage["completion_tokens"],
            total_tokens=usage["total_tokens"],
            generation_time=elapsed,
            tokens_per_sec=usage["completion_tokens"] / max(elapsed, 0.001),
            mode="llm", model_name=self.model_name, timestamp=time.time())
        return text

    def clear_chat(self):
        self._chat_messages.clear()

    def abort(self):
        self._abort = True

    # -- Demo adjust fallback --

    def _demo_adjust(self, notes: List[MidiNote], instruction: str,
                        mode: str = "full") -> List[MidiNote]:
        """Rule-based fallback when no LLM is available. Respects adjust mode."""
        instr = instruction.lower()
        result = []
        avg_pitch = sum(n.pitch for n in notes) / max(len(notes), 1)
        analysis = analyze_notes(notes)

        for n in notes:
            tick, pitch, vel, dur = n.tick, n.pitch, n.velocity, n.duration

            # Mode-specific adjustments
            if mode == "melody":
                # Only adjust pitches — smooth contour, stepwise motion
                if "higher" in instr:
                    pitch = min(127, pitch + random.choice([1, 2, 2, 3]))
                elif "lower" in instr:
                    pitch = max(0, pitch - random.choice([1, 2, 2, 3]))
                elif "invert" in instr:
                    pitch = int(2 * avg_pitch - pitch)
                    pitch = max(0, min(127, pitch))
                elif "smooth" in instr or "legato" in instr:
                    dur = min(dur * 2, TICKS_PER_BEAT * 4)
                else:
                    # Add passing tones occasionally
                    if random.random() < 0.15:
                        pitch += random.choice([-1, 1])
                        pitch = max(0, min(127, pitch))
                tick = n.tick  # Keep rhythm unchanged for melody mode
                dur = n.duration  # Keep duration unless explicitly changed

            elif mode == "harmony":
                # Add chord tones / extensions
                if "richer" in instr or "extend" in instr or "7th" in instr:
                    dur = max(dur, TICKS_PER_BEAT)
                if "spread" in instr:
                    if pitch > avg_pitch:
                        pitch = min(127, pitch + random.choice([0, 4, 7]))
                    else:
                        pitch = max(0, pitch - random.choice([0, 4, 7]))

            elif mode == "rhythm":
                # Only adjust timing
                if "staccato" in instr or "shorter" in instr:
                    dur = max(30, dur // 3)
                elif "legato" in instr or "longer" in instr:
                    dur = min(dur * 2, TICKS_PER_BEAT * 4)
                elif "syncop" in instr or "syncopation" in instr:
                    shift = TICKS_PER_BEAT // 4
                    tick = tick + shift
                if "swing" in instr:
                    if (tick // (TICKS_PER_BEAT // 2)) % 2 == 1:
                        tick += TICKS_PER_BEAT // 6
                if "quantize" in instr or "snap" in instr:
                    snap = TICKS_PER_BEAT // 4
                    tick = round(tick / snap) * snap
                pitch = n.pitch  # Keep pitches unchanged for rhythm mode
                vel = n.velocity

            elif mode == "velocity":
                # Only adjust velocity
                if "humanize" in instr or "natural" in instr:
                    vel = max(1, min(127, vel + random.randint(-12, 12)))
                elif "crescendo" in instr:
                    bar_pos = (tick % (TICKS_PER_BEAT * 4)) / (TICKS_PER_BEAT * 4)
                    vel = int(vel * (0.6 + 0.8 * bar_pos))
                elif "accent" in instr:
                    if tick % TICKS_PER_BEAT == 0:
                        vel = min(127, vel + 25)
                    else:
                        vel = max(1, vel - 10)
                else:
                    if "louder" in instr or "forte" in instr:
                        vel = min(127, vel + 20)
                    elif "softer" in instr or "piano" in instr:
                        vel = max(20, vel - 20)
                tick = n.tick
                pitch = n.pitch
                dur = n.duration

            elif mode == "density":
                if "sparse" in instr or "remove" in instr or "simplify" in instr:
                    if random.random() < 0.3:
                        continue  # Skip some notes
                elif "dense" in instr or "add" in instr or "fill" in instr:
                    # Keep original + add some below
                    result.append(MidiNote(tick=tick, pitch=pitch, velocity=max(1, vel - 15), duration=dur))
                    pitch = max(0, pitch - random.choice([3, 4, 5, 7]))

            else:  # full mode — apply all applicable
                if "staccato" in instr or "shorter" in instr:
                    dur = max(30, dur // 3)
                elif "legato" in instr or "longer" in instr:
                    dur = min(dur * 2, TICKS_PER_BEAT * 4)
                if "louder" in instr or "forte" in instr:
                    vel = min(127, vel + 20)
                elif "softer" in instr or "piano" in instr:
                    vel = max(20, vel - 20)
                if "higher" in instr or "up" in instr:
                    pitch = min(127, pitch + 2)
                elif "lower" in instr or "down" in instr:
                    pitch = max(0, pitch - 2)
                if "quantize" in instr or "snap" in instr:
                    snap = TICKS_PER_BEAT // 4
                    tick = round(tick / snap) * snap
                if "swing" in instr:
                    if (tick // (TICKS_PER_BEAT // 2)) % 2 == 1:
                        tick += TICKS_PER_BEAT // 6
                if "random" in instr or "humanize" in instr:
                    tick += random.randint(-20, 20)
                    vel = max(1, min(127, vel + random.randint(-10, 10)))
                if "invert" in instr:
                    pitch = int(2 * avg_pitch - pitch)
                    pitch = max(0, min(127, pitch))

            result.append(MidiNote(tick=tick, pitch=pitch, velocity=max(1, min(127, vel)), duration=max(1, dur)))

        if "retrograde" in instr or "reverse" in instr:
            if notes:
                last_tick = max(n.tick + n.duration for n in notes)
                result = []
                for n in reversed(notes):
                    new_tick = last_tick - n.tick - n.duration
                    result.append(MidiNote(tick=max(0, new_tick), pitch=n.pitch,
                                           velocity=n.velocity, duration=n.duration))
        return result

    # -- Parser --

    @staticmethod
    def _parse(text: str, max_bars: int = 64) -> List[MidiNote]:
        """Parse LLM text output into MidiNote objects.
        Validates pitch/velocity/duration ranges, filters extreme outliers,
        and clips to max_bars to prevent runaway generation artifacts."""
        notes = []
        max_tick = max_bars * TICKS_PER_BEAT * 4  # 4 beats per bar
        for line in text.strip().split("\n"):
            line = line.strip().strip("*").strip("-").strip()
            # Skip lines that are clearly not note data
            if not line or line.startswith("#") or line.startswith("//"):
                continue
            if "example" in line.lower() or "note" in line.lower() and ":" not in line:
                continue
            try:
                parts = line.replace("|", ":").replace(",", ":").replace("\t", ":").split(":")
                if len(parts) >= 4:
                    tick = int(parts[0])
                    pitch = int(parts[1])
                    velocity = int(parts[2])
                    duration = int(parts[3])
                    if (0 <= pitch <= 127 and 0 < velocity <= 127
                            and 0 < duration <= max_tick and 0 <= tick < max_tick):
                        notes.append(MidiNote(tick=tick, pitch=pitch,
                                              velocity=velocity, duration=duration))
            except (ValueError, IndexError):
                continue
        return notes


# ---------------------------------------------------------------------------
# Model download helper
# ---------------------------------------------------------------------------

def download_model(repo: str, filename: str, local_dir: str,
                   progress_cb: Optional[Callable[[int, int], None]] = None) -> str:
    """Download a GGUF model from HuggingFace. Returns the local file path."""
    if HF_AVAILABLE:
        return _download_hf(repo, filename, local_dir, progress_cb)
    else:
        return _download_url(repo, filename, local_dir, progress_cb)


def _download_hf(repo: str, filename: str, local_dir: str,
                 progress_cb: Optional[Callable] = None) -> str:
    os.makedirs(local_dir, exist_ok=True)

    class _Progress:
        def __init__(self, cb):
            self.cb = cb
        def __call__(self, progress):
            if self.cb:
                self.cb(progress.current, progress.total)

    path = hf_hub_download(
        repo_id=repo, filename=filename, local_dir=local_dir,
        local_dir_use_symlinks=False, tqdm_class=_Progress(progress_cb) if progress_cb else None,
    )
    return path


def _download_url(repo: str, filename: str, local_dir: str,
                  progress_cb: Optional[Callable] = None) -> str:
    """Fallback: construct HF URL and download with urllib."""
    os.makedirs(local_dir, exist_ok=True)
    url = f"https://huggingface.co/{repo}/resolve/main/{filename}"
    dest = os.path.join(local_dir, filename)

    def _reporthook(block_num, block_size, total_size):
        if progress_cb and total_size > 0:
            progress_cb(block_num * block_size, total_size)

    try:
        urlretrieve(url, dest, _reporthook)
        return dest
    except (URLError, Exception) as e:
        raise RuntimeError(f"Download failed: {e}\nInstall huggingface_hub for better downloads: pip install huggingface_hub")




# ---------------------------------------------------------------------------
# Biquad Filter (resonant low-pass)
# ---------------------------------------------------------------------------

class _BiquadFilter:
    """Second-order IIR biquad low-pass filter with resonance."""
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self._x1 = 0.0
        self._x2 = 0.0
        self._y1 = 0.0
        self._y2 = 0.0
        self._b0 = 1.0
        self._b1 = 0.0
        self._b2 = 0.0
        self._a1 = 0.0
        self._a2 = 0.0
    
    def set_lowpass(self, freq: float, q: float = 0.707):
        w0 = 2.0 * math.pi * freq / self.sample_rate
        cos_w0 = math.cos(w0)
        sin_w0 = math.sin(w0)
        alpha = sin_w0 / (2.0 * q)
        a0 = 1.0 + alpha
        self._b0 = ((1.0 - cos_w0) / 2.0) / a0
        self._b1 = (1.0 - cos_w0) / a0
        self._b2 = ((1.0 - cos_w0) / 2.0) / a0
        self._a1 = (-2.0 * cos_w0) / a0
        self._a2 = (1.0 - alpha) / a0
    
    def process(self, audio: np.ndarray) -> np.ndarray:
        b0, b1, b2 = self._b0, self._b1, self._b2
        a1, a2 = self._a1, self._a2
        x1 = self._x1; x2 = self._x2
        y1 = self._y1; y2 = self._y2
        out = np.empty_like(audio)
        for i in range(len(audio)):
            x0 = audio[i]
            y0 = b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
            out[i] = y0
            x2 = x1; x1 = x0; y2 = y1; y1 = y0
        self._x1 = x1; self._x2 = x2
        self._y1 = y1; self._y2 = y2
        return out
    
    def reset(self):
        self._x1 = self._x2 = self._y1 = self._y2 = 0.0

# ---------------------------------------------------------------------------
# Improved Synth Engine — pre-rendered audio + real-time mode
# ---------------------------------------------------------------------------

class SynthEngine:
    WAVEFORMS = {"sine": 0, "triangle": 1, "sawtooth": 2, "square": 3,
                 "piano": 4, "organ": 5, "pad": 6, "bass": 7, "strings": 8}

    def __init__(self, sample_rate: int = 44100, block_size: int = 512):
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.waveform = "piano"
        self.volume = 0.7
        self._lock = threading.Lock()
        self._stream: Optional[sd.OutputStream] = None
        self._audio_buffer: Optional[np.ndarray] = None
        self._buffer_pos = 0
        self._playing = False
        self._callback_data = {"phase": 0, "stop": False}

    @property
    def is_available(self) -> bool:
        return SD_AVAILABLE

    def set_waveform(self, name: str):
        if name in self.WAVEFORMS:
            self.waveform = name

    # ── Pre-rendered playback (preferred) ──────────────────────────────────

    def render_audio(self, notes: List[MidiNote], tempo: int,
                     total_ticks: int) -> np.ndarray:
        """Pre-render the entire MIDI sequence to a numpy array.
        Uses biquad low-pass filter with velocity-sensitive cutoff and
        a simple feedback delay for spatial depth."""
        tpb = TICKS_PER_BEAT
        ticks_per_sec = (tempo / 60.0) * tpb
        total_seconds = total_ticks / max(ticks_per_sec, 0.001)
        total_samples = int(total_seconds * self.sample_rate) + self.sample_rate
        audio = np.zeros(total_samples, dtype=np.float64)

        for note in notes:
            start_samp = int(note.tick / max(ticks_per_sec, 0.001) * self.sample_rate)
            dur_samp = int(note.duration / max(ticks_per_sec, 0.001) * self.sample_rate)
            release_samp = int(0.1 * self.sample_rate)
            end_samp = min(start_samp + dur_samp + release_samp, total_samples)
            if start_samp >= total_samples or end_samp <= start_samp:
                continue

            freq = 440.0 * (2.0 ** ((note.pitch - 69) / 12.0))
            vel_frac = note.velocity / 127.0
            amp = vel_frac * 0.11 * self.volume
            length = end_samp - start_samp
            t_rel = np.arange(length, dtype=np.float64) / self.sample_rate

            phase = 2.0 * np.pi * freq * (t_rel + start_samp / self.sample_rate)
            wave = self._generate_waveform(phase, freq, t_rel, dur_samp / self.sample_rate)
            env = self._adsr_envelope(length, dur_samp, release_samp)
            audio[start_samp:end_samp] += amp * env * wave

        # Velocity-sensitive biquad low-pass filter
        if notes:
            avg_vel = sum(n.velocity for n in notes) / len(notes)
            vel_factor = avg_vel / 127.0
            cutoff = 800.0 + vel_factor * 7200.0
            if self.waveform in ("bass",):
                cutoff = min(cutoff, 1200.0)
            elif self.waveform in ("pad", "strings"):
                cutoff = min(cutoff, 4000.0)
            lpf = _BiquadFilter(self.sample_rate)
            lpf.set_lowpass(cutoff, q=0.7)
            audio = lpf.process(audio)

        # Simple feedback delay for spatial depth
        delay_sec = 0.22
        delay_samples = int(delay_sec * self.sample_rate)
        feedback = 0.25
        if delay_samples < len(audio):
            delayed = np.zeros_like(audio)
            delayed[delay_samples:] = audio[:-delay_samples] * feedback
            tap2 = int(delay_samples * 1.5)
            if tap2 < len(audio):
                delayed[tap2:] += audio[:-tap2] * feedback * 0.6
            audio = audio + delayed * 0.3

        peak = np.max(np.abs(audio))
        if peak > 0.95:
            audio *= 0.95 / peak

        return np.clip(audio, -1.0, 1.0).astype(np.float32)

    def _generate_waveform(self, phase: np.ndarray, freq: float,
                            t_rel: np.ndarray, dur_s: float) -> np.ndarray:
        """Generate waveform samples based on current waveform type."""
        name = self.waveform
        if name == "sine":
            return np.sin(phase)
        elif name == "triangle":
            p = phase / (2 * np.pi) % 1.0
            return 2.0 * np.abs(2.0 * p - 1.0) - 1.0
        elif name == "sawtooth":
            p = phase / (2 * np.pi) % 1.0
            return 2.0 * p - 1.0
        elif name == "square":
            p = phase / (2 * np.pi) % 1.0
            return np.where(p < 0.5, 1.0, -1.0)
        elif name == "piano":
            # Piano-like: fundamental + harmonics with fast decay
            w = np.sin(phase) * 1.0
            w += np.sin(2 * phase) * 0.5 * np.exp(-t_rel * 4)
            w += np.sin(3 * phase) * 0.25 * np.exp(-t_rel * 6)
            w += np.sin(4 * phase) * 0.12 * np.exp(-t_rel * 10)
            # Brightness decreases over time
            brightness = np.exp(-t_rel * 3)
            return w * (0.7 + 0.3 * brightness)
        elif name == "organ":
            # Organ: drawbar-like harmonics
            w = np.sin(phase) * 0.8
            w += np.sin(2 * phase) * 0.6
            w += np.sin(3 * phase) * 0.4
            w += np.sin(4 * phase) * 0.3
            w += np.sin(6 * phase) * 0.2
            w += np.sin(8 * phase) * 0.1
            return w / 2.0
        elif name == "pad":
            # Pad: detuned for chorus effect
            w = np.sin(phase) + np.sin(phase * 1.003) + np.sin(phase * 0.997)
            w += 0.5 * np.sin(2 * phase + 0.3) + 0.5 * np.sin(2 * phase - 0.3)
            return w / 3.5
        elif name == "bass":
            # Bass: sub-oscillator + sine with fast decay
            w = np.sin(phase) * 1.0
            w += np.sin(2 * phase) * 0.6 * np.exp(-t_rel * 2)
            w += np.sin(0.5 * phase) * 0.4
            return w / 2.0
        elif name == "strings":
            # Strings: slow shimmer with vibrato
            vibrato = 1.0 + 0.003 * np.sin(2 * np.pi * 5.5 * t_rel)
            w = np.sin(phase * vibrato)
            w += np.sin(phase * vibrato * 1.002) * 0.7
            w += np.sin(phase * vibrato * 0.998) * 0.7
            w += np.sin(2 * phase * vibrato) * 0.2
            w += np.sin(3 * phase * vibrato) * 0.1 * np.exp(-t_rel * 2)
            return w / 2.7
        return np.sin(phase)

    def _adsr_envelope(self, length: int, dur_samples: int,
                       release_samples: int) -> np.ndarray:
        """Generate ADSR envelope using vectorized numpy (no Python loops)."""
        if length == 0:
            return np.zeros(0, dtype=np.float64)

        sr = self.sample_rate
        name = self.waveform

        if name in ("piano",):
            attack = int(0.005 * sr)
            decay = int(0.15 * sr)
            sustain = 0.3
        elif name in ("organ", "strings"):
            attack = int(0.02 * sr)
            decay = int(0.05 * sr)
            sustain = 0.8
        elif name in ("pad",):
            attack = int(0.15 * sr)
            decay = int(0.3 * sr)
            sustain = 0.7
        elif name in ("bass",):
            attack = int(0.003 * sr)
            decay = int(0.1 * sr)
            sustain = 0.5
        else:
            attack = int(0.01 * sr)
            decay = int(0.08 * sr)
            sustain = 0.6

        release = release_samples
        idx = np.arange(length, dtype=np.float64)
        env = np.zeros(length, dtype=np.float64)

        if attack > 0:
            mask_a = idx < attack
            env[mask_a] = idx[mask_a] / attack

        decay_end = attack + decay
        if decay > 0:
            mask_d = (idx >= attack) & (idx < decay_end)
            env[mask_d] = 1.0 - (1.0 - sustain) * ((idx[mask_d] - attack) / decay)

        mask_s = (idx >= decay_end) & (idx < dur_samples)
        env[mask_s] = sustain

        rel_end = dur_samples + release
        if release > 0:
            mask_r = (idx >= dur_samples) & (idx < rel_end)
            env[mask_r] = sustain * (1.0 - (idx[mask_r] - dur_samples) / release)

        return np.clip(env, 0.0, 1.0)

    @staticmethod
    def _lowpass(audio: np.ndarray, cutoff: float = 0.15) -> np.ndarray:
        """Simple one-pole low-pass IIR filter for warmth."""
        result = np.zeros_like(audio)
        result[0] = audio[0]
        alpha = cutoff
        for i in range(1, len(audio)):
            result[i] = alpha * audio[i] + (1 - alpha) * result[i - 1]
        return result

    def play_buffer(self, audio: np.ndarray):
        """Play a pre-rendered audio buffer."""
        if not SD_AVAILABLE:
            return
        self._stop_stream()
        self._audio_buffer = audio
        self._buffer_pos = 0
        self._playing = True
        self._callback_data = {"phase": 0, "stop": False}

        try:
            self._stream = sd.OutputStream(
                samplerate=self.sample_rate, channels=1, dtype="float32",
                blocksize=self.block_size, callback=self._buffer_callback,
            )
            self._stream.start()
        except Exception as e:
            print(f"[Synth] Playback error: {e}")
            self._playing = False

    def _buffer_callback(self, outdata, frames, time_info, status):
        if self._audio_buffer is None or self._callback_data["stop"]:
            outdata[:, 0] = 0
            return
        end = min(self._buffer_pos + frames, len(self._audio_buffer))
        chunk_len = end - self._buffer_pos
        if chunk_len <= 0:
            outdata[:, 0] = 0
            self._playing = False
            return
        outdata[:chunk_len, 0] = self._audio_buffer[self._buffer_pos:end]
        if chunk_len < frames:
            outdata[chunk_len:, 0] = 0
            self._playing = False
        self._buffer_pos = end

    @property
    def playback_position(self) -> float:
        """Current playback position in seconds."""
        if self._audio_buffer is not None and self.sample_rate > 0:
            return self._buffer_pos / self.sample_rate
        return 0.0

    @property
    def is_actively_playing(self) -> bool:
        return self._playing

    def stop(self):
        self._callback_data["stop"] = True
        self._stop_stream()
        self._audio_buffer = None
        self._buffer_pos = 0
        self._playing = False

    def _stop_stream(self):
        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None


# ---------------------------------------------------------------------------
# MIDI file I/O
# ---------------------------------------------------------------------------

def export_midi(notes: List[MidiNote], tempo: int = 120,
                filepath: Optional[str] = None) -> Optional[str]:
    if MIDO_AVAILABLE:
        return _export_mido(notes, tempo, filepath)
    else:
        return _export_raw(notes, tempo, filepath)


def _export_mido(notes: List[MidiNote], tempo: int, filepath: Optional[str]) -> str:
    mid = MidiFile(ticks_per_beat=TICKS_PER_BEAT)
    track = MidiTrack()
    mid.tracks.append(track)
    track.append(MetaMessage("set_tempo", tempo=mido.bpm2tempo(tempo), time=0))
    track.append(MetaMessage("time_signature", numerator=4, denominator=4, time=0))
    sorted_notes = sorted(notes, key=lambda n: n.tick)
    events = []
    for n in sorted_notes:
        events.append((n.tick, "note_on", n.pitch, n.velocity))
        events.append((n.tick + n.duration, "note_off", n.pitch, 0))
    events.sort(key=lambda e: (e[0], e[1] != "note_off"))
    prev = 0
    for abs_t, msg_t, pitch, vel in events:
        delta = max(0, abs_t - prev)
        prev = abs_t
        track.append(Message(msg_t, note=pitch, velocity=vel, time=int(delta)))
    if filepath is None:
        filepath = os.path.join(tempfile.gettempdir(), "generated.mid")
    mid.save(filepath)
    return filepath


def _export_raw(notes: List[MidiNote], tempo: int, filepath: Optional[str]) -> str:
    ticks_per_beat = TICKS_PER_BEAT
    sorted_notes = sorted(notes, key=lambda n: n.tick)
    events = []
    for n in sorted_notes:
        events.append((n.tick, 0x90, n.pitch, n.velocity))
        events.append((n.tick + n.duration, 0x80, n.pitch, 0))
    events.sort(key=lambda e: (e[0], e[1] != 0x80))
    track_data = bytearray()
    track_data += b"\x00\xFF\x51\x03"
    us_per_beat = int(60000000 / tempo)
    track_data += us_per_beat.to_bytes(3, "big")
    track_data += b"\x00\xFF\x58\x04\x04\x02\x18\x08"
    prev = 0
    for abs_t, status, pitch, vel in events:
        delta = max(0, abs_t - prev)
        prev = abs_t
        track_data += bytes(_encode_vlq(delta))
        track_data += bytes([status, pitch, vel])
    track_data += b"\x00\xFF\x2F\x00"
    header = b"MThd" + (6).to_bytes(4, "big") + (0).to_bytes(2, "big") + (1).to_bytes(2, "big") + ticks_per_beat.to_bytes(2, "big")
    track_header = b"MTrk" + len(track_data).to_bytes(4, "big")
    data = header + track_header + bytes(track_data)
    if filepath is None:
        filepath = os.path.join(tempfile.gettempdir(), "generated.mid")
    with open(filepath, "wb") as f:
        f.write(data)
    return filepath


def _encode_vlq(value: int) -> bytes:
    if value < 0:
        value = 0
    result = []
    result.append(value & 0x7F)
    value >>= 7
    while value:
        result.append((value & 0x7F) | 0x80)
        value >>= 7
    result.reverse()
    return bytes(result)


def import_midi(filepath: str) -> Tuple[List[MidiNote], int]:
    if MIDO_AVAILABLE:
        return _import_mido(filepath)
    else:
        return _import_raw(filepath)


def _import_mido(filepath: str) -> Tuple[List[MidiNote], int]:
    mid = MidiFile(filepath)
    tempo = 120
    notes: List[MidiNote] = []
    note_on_stack: dict = {}
    for track in mid.tracks:
        tick = 0
        for msg in track:
            tick += msg.time
            if msg.type == "set_tempo":
                tempo = int(mido.tempo2bpm(msg.tempo))
            elif msg.type == "note_on" and msg.velocity > 0:
                key = (msg.channel, msg.note)
                note_on_stack.setdefault(key, []).append((tick, msg.velocity))
            elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                key = (msg.channel, msg.note)
                if key in note_on_stack and note_on_stack[key]:
                    start_tick, vel = note_on_stack[key].pop(0)
                    dur = max(1, tick - start_tick)
                    notes.append(MidiNote(tick=start_tick, pitch=msg.note,
                                          velocity=vel, duration=dur))
    return notes, tempo


def _import_raw(filepath: str) -> Tuple[List[MidiNote], int]:
    with open(filepath, "rb") as f:
        data = f.read()
    tempo = 120
    notes: List[MidiNote] = []
    note_on_stack: dict = {}
    idx = data.find(b"MTrk")
    if idx < 0:
        return notes, tempo
    idx += 4
    track_len = int.from_bytes(data[idx:idx + 4], "big")
    idx += 4
    track_end = idx + track_len
    tick = 0
    while idx < track_end:
        delta = 0
        while idx < track_end:
            b = data[idx]; idx += 1
            delta = (delta << 7) | (b & 0x7F)
            if not (b & 0x80):
                break
        tick += delta
        if idx >= track_end:
            break
        status = data[idx]
        if status == 0xFF:
            idx += 1
            if idx < track_end:
                meta_type = data[idx]; idx += 1
                length = 0
                while idx < track_end:
                    b = data[idx]; idx += 1
                    length = (length << 7) | (b & 0x7F)
                    if not (b & 0x80):
                        break
                if meta_type == 0x51 and length == 3:
                    us_per_beat = int.from_bytes(data[idx:idx + 3], "big")
                    tempo = int(60000000 / us_per_beat) if us_per_beat else 120
                idx += length
        elif 0x80 <= status <= 0xEF:
            msg_type = status & 0xF0; idx += 1
            if idx + 1 >= track_end:
                break
            pitch = data[idx]; velocity = data[idx + 1]; idx += 2
            if msg_type == 0x90 and velocity > 0:
                note_on_stack.setdefault(pitch, []).append((tick, velocity))
            elif msg_type == 0x80 or (msg_type == 0x90 and velocity == 0):
                if pitch in note_on_stack and note_on_stack[pitch]:
                    st, vel = note_on_stack[pitch].pop(0)
                    dur = max(1, tick - st)
                    notes.append(MidiNote(tick=st, pitch=pitch, velocity=vel, duration=dur))
        elif 0xF0 <= status <= 0xF7:
            idx += 1
            while idx < track_end:
                b = data[idx]; idx += 1
                if not (b & 0x80):
                    break
        else:
            idx += 1
    return notes, tempo