#!/usr/bin/env python3
"""
MIDI LLM Studio — PySide6 Desktop Application
Piano roll editor with local LLM-powered MIDI generation, editing, and playback.
"""

import sys
import os
import math
import time
import re
import threading
from typing import List, Optional, Set, Tuple
import copy
from dataclasses import asdict
import numpy as np

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QLabel, QPushButton, QComboBox, QSpinBox, QDoubleSpinBox,
    QTextEdit, QGroupBox, QToolBar, QFileDialog, QMessageBox,
    QStatusBar, QSlider, QScrollArea, QCheckBox, QFrame, QMenu,
    QSizePolicy, QToolButton, QProgressBar, QLineEdit, QTabWidget
)
from PySide6.QtCore import (
    Qt, QTimer, QRect, QRectF, QPointF, Signal, QSize, QPoint,
    QMimeData, QElapsedTimer
)
from PySide6.QtGui import (
    QPainter, QColor, QPen, QBrush, QFont, QKeySequence,
    QPainterPath, QLinearGradient, QCursor, QAction, QRadialGradient
)

from midi_engine import (
    MidiNote, MidiLLMGenerator, SynthEngine,
    generate_demo_notes, export_midi, import_midi,
    TICKS_PER_BEAT, note_name, is_black_key, pitch_class_color_rgb,
    LLAMA_AVAILABLE, SD_AVAILABLE, MIDO_AVAILABLE, HF_AVAILABLE,
    RECOMMENDED_MODELS, download_model,
    PROMPT_TEMPLATES, GENRE_DETAILS, _build_system_prompt,
    GenerationStats, GenerationHistory,
    analyze_notes, ADJUST_MODES, FEW_SHOT_EXAMPLES,
    merge_notes, separate_layers, notes_to_text,
    CHORD_PROGRESSION_SUGGESTIONS, get_genre_progressions,
)


def pitch_class_color(pitch: int, velocity: int = 100) -> QColor:
    r, g, b = pitch_class_color_rgb(pitch, velocity)
    return QColor(r, g, b)


# ═══════════════════════════════════════════════════════════════════════════
#  Undo/Redo Manager
# ═══════════════════════════════════════════════════════════════════════════

class UndoManager:
    """Tracks note state changes for undo/redo."""
    def __init__(self, max_history: int = 100):
        self._undo_stack: list = []
        self._redo_stack: list = []
        self._max = max_history

    def save_state(self, notes: list):
        snapshot = [asdict(n) for n in notes]
        self._undo_stack.append(snapshot)
        if len(self._undo_stack) > self._max:
            self._undo_stack.pop(0)
        self._redo_stack.clear()

    def can_undo(self) -> bool:
        return len(self._undo_stack) > 0

    def can_redo(self) -> bool:
        return len(self._redo_stack) > 0

    def undo(self, current_notes: list) -> list:
        if not self._undo_stack:
            return current_notes
        self._redo_stack.append([asdict(n) for n in current_notes])
        snapshot = self._undo_stack.pop()
        return [MidiNote.from_dict(d) for d in snapshot]

    def redo(self, current_notes: list) -> list:
        if not self._redo_stack:
            return current_notes
        self._undo_stack.append([asdict(n) for n in current_notes])
        snapshot = self._redo_stack.pop()
        return [MidiNote.from_dict(d) for d in snapshot]

    def clear(self):
        self._undo_stack.clear()
        self._redo_stack.clear()


# ═══════════════════════════════════════════════════════════════════════════
#  Piano Roll Widget  (with minimap + glow + beat pulse)
# ═══════════════════════════════════════════════════════════════════════════

class PianoRollWidget(QWidget):
    noteSelected = Signal(list)
    notesChanged = Signal()
    playbackPositionChanged = Signal(int)

    PIANO_WIDTH = 64
    HEADER_HEIGHT = 28
    MINIMAP_HEIGHT = 36
    MIN_NOTE_HEIGHT = 6
    MAX_NOTE_HEIGHT = 28
    MIN_PPQ = 0.1
    MAX_PPQ = 3.0

    def __init__(self, parent=None):
        super().__init__(parent)
        self.notes: List[MidiNote] = []
        self.selected_ids: Set[str] = set()

        self.ppq = 0.6
        self.note_height = 14
        self.scroll_x = 0.0
        self.scroll_y = 0.0

        self.tool = "select"
        self._dragging = False
        self._drag_type = None
        self._drag_start = None
        self._drag_note_origins = {}
        self._select_rect = None
        self._hover_note_id = None
        self._last_draw_pitch = -1

        self.playback_tick = 0
        self.is_playing = False
        self.total_ticks = TICKS_PER_BEAT * 16
        self._active_note_ids: Set[str] = set()  # currently sounding

        # Beat pulse
        self._beat_pulse_alpha = 0.0

        # Colors
        self._bg_color = QColor(12, 12, 22)
        self._grid_color = QColor(22, 22, 38)
        self._beat_color = QColor(32, 32, 52)
        self._bar_color = QColor(42, 42, 68)
        self._white_key = QColor(240, 240, 245)
        self._black_key = QColor(40, 40, 50)
        self._white_key_alt = QColor(230, 230, 240)
        self._cursor_color = QColor(255, 80, 80, 200)
        self._select_color = QColor(100, 140, 255, 60)
        self._select_border = QColor(100, 140, 255, 180)
        self._beat_pulse_color = QColor(124, 92, 252, 40)

        # Undo/Redo
        self.undo = UndoManager()
        self._ignore_notes_changed = False

        # Snap setting (1 = 16th, 2 = 8th, 4 = quarter, 8 = half, 16 = whole)
        self.snap_divisor = 4  # default: 16th note
        self._clipboard: list = []

        # Velocity drag
        self._vel_dragging = False
        self._vel_drag_note = None
        self._vel_drag_start_y = 0.0
        self._vel_drag_orig_vel = 0

        self.setMinimumSize(600, 300)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    @property
    def snap_ticks(self) -> int:
        return TICKS_PER_BEAT // self.snap_divisor

    # ── coordinate helpers ────────────────────────────────────────────────

    def _note_area_height(self) -> int:
        return max(0, self.height() - self.HEADER_HEIGHT - self.MINIMAP_HEIGHT)

    def _visible_range(self) -> Tuple[int, int, int, int]:
        area_w = self.width() - self.PIANO_WIDTH
        area_h = self._note_area_height()
        min_tick = int(self.scroll_x / max(self.ppq, 0.001))
        max_tick = int((self.scroll_x + area_w) / max(self.ppq, 0.001)) + 1
        max_pitch = min(127, int(self.scroll_y / max(self.note_height, 1)))
        min_pitch = max(0, int((self.scroll_y + area_h) / max(self.note_height, 1)) + 1)
        return min_pitch, max_pitch, min_tick, max_t

    def _tick_to_x(self, tick: int) -> float:
        return self.PIANO_WIDTH + tick * self.ppq - self.scroll_x

    def _pitch_to_y(self, pitch: int) -> float:
        return self.HEADER_HEIGHT + (127 - pitch) * self.note_height - self.scroll_y

    def _x_to_tick(self, x: float) -> int:
        return max(0, int((x - self.PIANO_WIDTH + self.scroll_x) / max(self.ppq, 0.001)))

    def _y_to_pitch(self, y: float) -> int:
        return max(0, min(127, int(127 - (y - self.HEADER_HEIGHT + self.scroll_y) / max(self.note_height, 1))))

    # ── hit testing ───────────────────────────────────────────────────────

    def _note_at(self, pos: QPointF) -> Optional[Tuple[MidiNote, str]]:
        x, y = pos.x(), pos.y()
        margin = 5
        for n in reversed(self.notes):
            nx = self._tick_to_x(n.tick)
            ny = self._pitch_to_y(n.pitch)
            nw = max(n.duration * self.ppq, 4)
            nh = self.note_height - 1
            if nx <= x <= nx + nw and ny <= y <= ny + nh:
                # Check if in velocity bar zone (bottom 30% of note)
                vel_frac = n.velocity / 127.0
                vel_h = max(2, int(nh * 0.15 * vel_frac + nh * 0.05))
                vel_top = ny + nh - vel_h
                if nw > 8 and nh > 6 and y >= vel_top and self.tool == "select":
                    return n, "vel_bar"
                if x < nx + margin: return n, "left_edge"
                elif x > nx + nw - margin: return n, "right_edge"
                return n, "body"
        return None

    def _notes_in_rect(self, rect: QRectF) -> List[MidiNote]:
        result = []
        for n in self.notes:
            r = QRectF(self._tick_to_x(n.tick), self._pitch_to_y(n.pitch),
                       max(n.duration * self.ppq, 4), self.note_height - 1)
            if rect.intersects(r):
                result.append(n)
        return result

    # ── painting ──────────────────────────────────────────────────────────

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, False)
        w, h = self.width(), self.height()
        note_h = self._note_area_height()

        p.fillRect(0, 0, w, h, self._bg_color)

        # Note area
        p.setClipRect(self.PIANO_WIDTH, self.HEADER_HEIGHT, w - self.PIANO_WIDTH, note_h)
        self._draw_grid(p, w, note_h)
        self._draw_beat_pulse(p, w, note_h)
        self._draw_notes(p)
        self._draw_select_rect(p)
        self._draw_cursor(p, note_h)
        p.setClipping(False)

        self._draw_header(p, w)
        self._draw_piano(p, note_h)
        self._draw_minimap(p, w, h, note_h)

        # Borders
        p.setPen(QPen(QColor(50, 50, 70), 1))
        p.drawLine(self.PIANO_WIDTH, 0, self.PIANO_WIDTH, note_h + self.HEADER_HEIGHT)
        p.drawLine(0, self.HEADER_HEIGHT, w, self.HEADER_HEIGHT)
        p.drawLine(0, note_h + self.HEADER_HEIGHT, w, note_h + self.HEADER_HEIGHT)

        # Info overlay
        self._draw_info_overlay(p, w)

        p.end()

    def _draw_grid(self, p: QPainter, w: int, h: int):
        min_p, max_p, min_t, max_t = self._visible_range()
        for pitch in range(max_p, min_p - 1, -1):
            y = self._pitch_to_y(pitch)
            if y < self.HEADER_HEIGHT - 1 or y > h + self.HEADER_HEIGHT:
                continue
            black = is_black_key(pitch)
            if black:
                p.fillRect(int(self.PIANO_WIDTH), int(y), w - self.PIANO_WIDTH, int(self.note_height),
                           QColor(16, 16, 28))
            color = self._grid_color if not black else QColor(18, 18, 30)
            p.setPen(QPen(color, 1))
            p.drawLine(self.PIANO_WIDTH, int(y), w, int(y))

        sub_beat = TICKS_PER_BEAT // 4
        eighth = TICKS_PER_BEAT // 2
        for tick in range(min_t, max_t + 1):
            x = self._tick_to_x(tick)
            if x < self.PIANO_WIDTH - 1 or x > w:
                continue
            if tick % (TICKS_PER_BEAT * 4) == 0:
                p.setPen(QPen(self._bar_color, 1))
                p.drawLine(int(x), self.HEADER_HEIGHT, int(x), h + self.HEADER_HEIGHT)
            elif tick % TICKS_PER_BEAT == 0:
                p.setPen(QPen(self._beat_color, 1))
                p.drawLine(int(x), self.HEADER_HEIGHT, int(x), h + self.HEADER_HEIGHT)
            elif tick % eighth == 0:
                p.setPen(QPen(QColor(28, 28, 46), 1))
                p.drawLine(int(x), self.HEADER_HEIGHT, int(x), h + self.HEADER_HEIGHT)
            elif tick % sub_beat == 0:
                # Sixteenth notes: short tick marks + subtle dots
                p.setPen(QPen(QColor(30, 30, 48), 1))
                p.drawLine(int(x), self.HEADER_HEIGHT, int(x), self.HEADER_HEIGHT + 6)
                p.setPen(QPen(QColor(35, 35, 55), 1))
                for dy in range(0, h, 40):
                    p.drawPoint(int(x), self.HEADER_HEIGHT + 10 + dy)

        # Note overlap indicators — subtle red dots where notes overlap on same pitch
        overlap_map = {}
        for n in self.notes:
            if n.pitch < min_p or n.pitch > max_p:
                continue
            if n.tick + n.duration < min_t or n.tick > max_t:
                continue
            if n.pitch not in overlap_map:
                overlap_map[n.pitch] = []
            overlap_map[n.pitch].append((n.tick, n.tick + n.duration))
        for pitch, intervals in overlap_map.items():
            if len(intervals) < 2:
                continue
            intervals.sort()
            y = self._pitch_to_y(pitch)
            mid_y = int(y + self.note_height / 2)
            for i in range(len(intervals) - 1):
                a_start, a_end = intervals[i]
                b_start, b_end = intervals[i + 1]
                overlap_start = max(a_start, b_start)
                overlap_end = min(a_end, b_end)
                if overlap_start < overlap_end:
                    ox = self._tick_to_x(overlap_start)
                    oex = self._tick_to_x(overlap_end)
                    if oex > self.PIANO_WIDTH and ox < w:
                        p.setPen(Qt.NoPen)
                        p.setBrush(QColor(255, 60, 60, 30))
                        p.drawRect(QRectF(max(self.PIANO_WIDTH, ox), int(y),
                                         oex - ox, int(self.note_height)))

    def _draw_beat_pulse(self, p: QPainter, w: int, h: int):
        """Draw a translucent column highlight on the current beat."""
        if self._beat_pulse_alpha <= 0:
            return
        tpb = TICKS_PER_BEAT
        beat_start = (self.playback_tick // tpb) * tpb
        x1 = self._tick_to_x(beat_start)
        x2 = self._tick_to_x(beat_start + tpb)
        color = QColor(self._beat_pulse_color)
        color.setAlpha(int(self._beat_pulse_alpha * 60))
        p.fillRect(QRectF(max(self.PIANO_WIDTH, x1), self.HEADER_HEIGHT,
                          x2 - x1, h), color)

    def _draw_notes(self, p: QPainter):
        min_p, max_p, min_t, max_t = self._visible_range()

        for n in self.notes:
            if n.pitch < min_p or n.pitch > max_p:
                continue
            if n.tick + n.duration < min_t or n.tick > max_t:
                continue

            x = self._tick_to_x(n.tick)
            y = self._pitch_to_y(n.pitch)
            nw = max(n.duration * self.ppq, 3)
            nh = self.note_height - 1

            is_active = n.id in self._active_note_ids
            is_selected = n.id in self.selected_ids
            is_hovered = n.id == self._hover_note_id

            base_color = pitch_class_color(n.pitch, n.velocity)
            vel_frac = n.velocity / 127.0

            # Note shadow for depth
            if nh >= 10 and nw > 6:
                shadow_color = QColor(0, 0, 0, 40)
                p.setPen(Qt.NoPen)
                p.setBrush(QBrush(shadow_color))
                p.drawRoundedRect(QRectF(x + 2, y + 2, nw, nh), 2, 2)

            if is_active:
                # Multi-layer glow for active notes
                for gs, ga in [(6, 20), (3, 40)]:
                    gc = QColor(base_color)
                    gc.setAlpha(ga)
                    p.setPen(Qt.NoPen)
                    p.setBrush(QBrush(gc))
                    p.drawRoundedRect(QRectF(x - gs, y - gs, nw + gs * 2, nh + gs * 2), 4, 4)
                base_color = base_color.lighter(150)

            if is_selected:
                p.setPen(QPen(QColor(255, 255, 255, 220), 2))
            elif is_hovered:
                p.setPen(QPen(base_color.lighter(160), 1))
            else:
                p.setPen(QPen(base_color.darker(130), 1))

            # Gradient fill with velocity-dependent brightness
            grad = QLinearGradient(x, y, x, y + nh)
            bright = 1.0 + 0.3 * vel_frac
            if is_active:
                bright += 0.2
            grad.setColorAt(0, base_color.lighter(int(120 * bright)))
            grad.setColorAt(0.4, base_color.lighter(int(105 * bright)))
            grad.setColorAt(1, base_color.darker(int(max(80, 120 - vel_frac * 40))))
            p.setBrush(QBrush(grad))
            p.drawRoundedRect(QRectF(x, y, nw, nh), 2, 2)

            # Velocity bar proportional to velocity
            if nw > 8 and nh > 6:
                vel_bar_h = max(2, int(nh * 0.15 * vel_frac + nh * 0.05))
                vel_color = QColor(255, 255, 255, int(30 + 100 * vel_frac))
                p.setPen(Qt.NoPen)
                p.setBrush(QBrush(vel_color))
                p.drawRoundedRect(QRectF(x + 1, y + nh - vel_bar_h, nw - 2, vel_bar_h - 1), 1, 1)

            # Bright left edge accent
            if nw > 12 and nh > 8:
                accent = QColor(255, 255, 255, int(25 + 30 * vel_frac))
                p.setPen(Qt.NoPen)
                p.setBrush(QBrush(accent))
                p.drawRoundedRect(QRectF(x + 1, y + 1, 2, nh - 2), 1, 1)

            # Note name label
            if nw > 44 and nh > 10:
                p.setPen(QColor(0, 0, 0, 200))
                p.setFont(QFont("Segoe UI", 7))
                p.drawText(QRectF(x + 5, y + 1, nw - 9, nh - 1),
                           Qt.AlignVCenter | Qt.AlignLeft, note_name(n.pitch))

    def _draw_select_rect(self, p: QPainter):
        if self._select_rect:
            p.setPen(QPen(self._select_border, 1, Qt.DashLine))
            p.setBrush(QBrush(self._select_color))
            p.drawRect(self._select_rect)

    def _draw_cursor(self, p: QPainter, h: int):
        x = self._tick_to_x(self.playback_tick)
        if self.PIANO_WIDTH <= x <= self.width():
            # Wide glow column
            glow_w = 30
            grad = QLinearGradient(x - glow_w, 0, x + glow_w, 0)
            grad.setColorAt(0, QColor(255, 60, 60, 0))
            grad.setColorAt(0.35, QColor(255, 60, 60, 15))
            grad.setColorAt(0.5, QColor(255, 80, 80, 35))
            grad.setColorAt(0.65, QColor(255, 60, 60, 15))
            grad.setColorAt(1, QColor(255, 60, 60, 0))
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(grad))
            p.drawRect(QRectF(x - glow_w, self.HEADER_HEIGHT, glow_w * 2, h))

            # Line with shadow
            p.setPen(QPen(QColor(0, 0, 0, 60), 3))
            p.drawLine(int(x) + 1, self.HEADER_HEIGHT, int(x) + 1, h + self.HEADER_HEIGHT)
            p.setPen(QPen(self._cursor_color, 2))
            p.drawLine(int(x), self.HEADER_HEIGHT, int(x), h + self.HEADER_HEIGHT)

            # Triangle with border
            path = QPainterPath()
            path.moveTo(x - 7, self.HEADER_HEIGHT - 1)
            path.lineTo(x + 7, self.HEADER_HEIGHT - 1)
            path.lineTo(x, self.HEADER_HEIGHT + 11)
            path.closeSubpath()
            p.setBrush(QBrush(self._cursor_color))
            p.setPen(QPen(QColor(255, 120, 120), 1))
            p.drawPath(path)

            # Position label in header
            bar = self.playback_tick // (TICKS_PER_BEAT * 4) + 1
            beat = (self.playback_tick // TICKS_PER_BEAT) % 4 + 1
            p.setPen(QColor(255, 200, 200, 220))
            p.setFont(QFont("Segoe UI", 7, QFont.Bold))
            p.drawText(int(x) + 10, self.HEADER_HEIGHT - 4, f"{bar}.{beat}")

    def _draw_header(self, p: QPainter, w: int):
        p.fillRect(0, 0, w, self.HEADER_HEIGHT, QColor(18, 18, 30))
        p.setPen(QColor(140, 140, 170))
        p.setFont(QFont("Segoe UI", 8))
        min_t, max_t = self._visible_range()[2], self._visible_range()[3]
        tpb = TICKS_PER_BEAT
        for tick in range(min_t, max_t + 1):
            x = self._tick_to_x(tick)
            if x < self.PIANO_WIDTH or x > w:
                continue
            if tick % (tpb * 4) == 0:
                bar = tick // (tpb * 4) + 1
                p.setPen(QColor(200, 200, 230))
                p.setFont(QFont("Segoe UI", 8, QFont.Bold))
                label = f"{bar}"
                tw = p.fontMetrics().horizontalAdvance(label)
                p.fillRect(int(x + 2), 1, tw + 6, self.HEADER_HEIGHT - 2, QColor(28, 28, 48))
                p.drawText(int(x + 5), 17, label)
            elif tick % tpb == 0:
                beat = (tick // tpb) % 4 + 1
                p.setPen(QColor(80, 80, 110))
                p.setFont(QFont("Segoe UI", 7))
                p.drawText(int(x + 3), 16, str(beat))
            elif tick % (tpb // 2) == 0 and self.ppq > 0.4:
                p.setPen(QColor(50, 50, 75))
                p.setFont(QFont("Segoe UI", 6))
                if self.ppq > 0.8:
                    p.drawText(int(x + 2), 15, "+")

    def _draw_piano(self, p: QPainter, h: int):
        min_p, max_p = self._visible_range()[0], self._visible_range()[1]
        active_pitches = {n.pitch for n in self._active_note_ids}
        has_notes = {n.pitch for n in self.notes}
        pitch_max_vel = {}
        for n in self.notes:
            if n.pitch not in pitch_max_vel or n.velocity > pitch_max_vel[n.pitch]:
                pitch_max_vel[n.pitch] = n.velocity

        for pitch in range(max_p, min_p - 1, -1):
            y = self._pitch_to_y(pitch)
            if y < self.HEADER_HEIGHT or y > h + self.HEADER_HEIGHT:
                continue
            nh = self.note_height
            black = is_black_key(pitch)

            if black:
                color = QColor(35, 35, 48)
                if pitch in active_pitches:
                    color = QColor(100, 80, 180)
                elif pitch in has_notes:
                    pc = pitch_class_color(pitch, pitch_max_vel.get(pitch, 80))
                    color = QColor(35 + pc.red() // 8, 35 + pc.green() // 8, 48 + pc.blue() // 8)
                p.fillRect(0, int(y), self.PIANO_WIDTH, int(nh), color)
            else:
                row = pitch // 12
                color = self._white_key if row % 2 == 0 else self._white_key_alt
                if pitch in active_pitches:
                    color = QColor(180, 200, 255)
                elif pitch in has_notes:
                    pc = pitch_class_color(pitch, pitch_max_vel.get(pitch, 80))
                    color = QColor(
                        min(255, 220 + pc.red() // 16),
                        min(255, 225 + pc.green() // 16),
                        min(255, 240 + pc.blue() // 16))
                p.fillRect(0, int(y), self.PIANO_WIDTH, int(nh), color)
                p.setPen(QPen(QColor(180, 180, 195), 1))
                p.drawLine(0, int(y), self.PIANO_WIDTH, int(y))

                if pitch % 12 == 0:
                    p.setPen(QColor(60, 60, 80))
                    p.setFont(QFont("Segoe UI", 7, QFont.Bold))
                    p.drawText(4, int(y + nh - 3), note_name(pitch))
                elif nh > 10:
                    p.setPen(QColor(130, 130, 150))
                    p.setFont(QFont("Segoe UI", 6))
                    p.drawText(6, int(y + nh - 3), note_name(pitch))

            # Velocity indicator bar on right edge
            if pitch in pitch_max_vel and nh > 6 and pitch not in active_pitches:
                vf = pitch_max_vel[pitch] / 127.0
                bar_w = max(2, int(4 * vf))
                pc = pitch_class_color(pitch, pitch_max_vel[pitch])
                pc.setAlpha(int(80 + 120 * vf))
                p.setPen(Qt.NoPen)
                p.setBrush(QBrush(pc))
                p.drawRect(self.PIANO_WIDTH - bar_w - 1, int(y + 1), bar_w, int(nh - 2))

        p.setPen(QPen(QColor(50, 50, 70), 1))
        p.drawLine(self.PIANO_WIDTH - 1, self.HEADER_HEIGHT, self.PIANO_WIDTH - 1, h + self.HEADER_HEIGHT)

    def _draw_minimap(self, p: QPainter, w: int, h: int, note_area_h: int):
        """Draw a minimap overview strip at the bottom."""
        mm_y = note_area_h + self.HEADER_HEIGHT
        mm_h = self.MINIMAP_HEIGHT

        # Background
        p.fillRect(0, mm_y, w, mm_h, QColor(10, 10, 18))

        if not self.notes:
            p.setPen(QColor(50, 50, 70))
            p.setFont(QFont("Segoe UI", 8))
            p.drawText(QRectF(self.PIANO_WIDTH, mm_y, w - self.PIANO_WIDTH, mm_h),
                       Qt.AlignCenter, "No notes")
            return

        min_tick = min(n.tick for n in self.notes)
        max_tick = max(n.tick + n.duration for n in self.notes)
        min_pitch = min(n.pitch for n in self.notes)
        max_pitch = max(n.pitch for n in self.notes)
        tick_range = max(max_tick - min_tick, TICKS_PER_BEAT)
        pitch_range = max(max_pitch - min_pitch, 12)

        mm_w = w - self.PIANO_WIDTH
        x_scale = mm_w / tick_range
        y_scale = mm_h / pitch_range

        # Draw notes as tiny colored rectangles
        for n in self.notes:
            nx = self.PIANO_WIDTH + (n.tick - min_tick) * x_scale
            ny = mm_y + (max_pitch - n.pitch) * y_scale
            nw = max(n.duration * x_scale, 1)
            nh = max(y_scale, 1)
            color = pitch_class_color(n.pitch, n.velocity)
            color.setAlpha(180)
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(color))
            p.drawRect(QRectF(nx, ny, nw, nh))

        # Viewport indicator
        vis_min_t = self.scroll_x / max(self.ppq, 0.001)
        vis_max_t = (self.scroll_x + mm_w / x_scale * (mm_w / (w - self.PIANO_WIDTH))) if x_scale > 0 else vis_min_t
        # Simpler: map visible tick range to minimap
        vis_x1 = self.PIANO_WIDTH + (vis_min_t - min_tick) * x_scale
        area_w = w - self.PIANO_WIDTH
        vis_x2 = vis_x1 + (area_w / max(self.ppq, 0.001)) * x_scale
        vis_x1 = max(self.PIANO_WIDTH, min(w, vis_x1))
        vis_x2 = max(self.PIANO_WIDTH, min(w, vis_x2))

        p.setPen(QPen(QColor(255, 255, 255, 80), 1))
        p.setBrush(QColor(255, 255, 255, 15))
        p.drawRect(QRectF(vis_x1, mm_y, vis_x2 - vis_x1, mm_h))

        # Playback position on minimap
        px = self.PIANO_WIDTH + (self.playback_tick - min_tick) * x_scale
        if self.PIANO_WIDTH <= px <= w:
            p.setPen(QPen(QColor(255, 80, 80, 200), 1))
            p.drawLine(int(px), mm_y, int(px), mm_y + mm_h)

    def _draw_info_overlay(self, p: QPainter, w: int):
        """Draw note count, selection, zoom, and pitch range in top-right."""
        total = len(self.notes)
        sel = len(self.selected_ids)
        if total == 0 and not self.is_playing:
            return
        parts = [f"{total} notes"]
        if sel:
            parts.append(f"{sel} sel")
        if self.is_playing:
            parts.append("▶ Playing")
        parts.append(f"zoom {self.ppq:.1f}x")
        text = "  |  ".join(parts)

        p.setPen(QColor(120, 120, 160, 180))
        p.setFont(QFont("Segoe UI", 8))
        fm = p.fontMetrics()
        tw = fm.horizontalAdvance(text) + 16
        p.fillRect(w - tw - 8, self.HEADER_HEIGHT + 4, tw, 18, QColor(12, 12, 22, 200))
        p.drawText(w - tw, self.HEADER_HEIGHT + 16, text)

        # Pitch range on right edge
        if self.notes:
            min_p = min(n.pitch for n in self.notes)
            max_p = max(n.pitch for n in self.notes)
            p.setPen(QColor(80, 80, 110, 120))
            p.setFont(QFont("Segoe UI", 7))
            range_text = f"{note_name(max_p)} - {note_name(min_p)}"
            p.save()
            p.translate(w - 4, self.HEADER_HEIGHT + self._note_area_height() // 2)
            p.rotate(-90)
            p.drawText(QRectF(-50, -8, 100, 16), Qt.AlignCenter, range_text)
            p.restore()

    # ── update active notes ───────────────────────────────────────────────

    def update_active_notes(self, current_tick: int):
        """Update which notes are currently sounding for glow effect."""
        new_active = set()
        for n in self.notes:
            if n.tick <= current_tick < n.tick + n.duration:
                new_active.add(n.id)
        if new_active != self._active_note_ids:
            self._active_note_ids = new_active
            self.update()

    def trigger_beat_pulse(self):
        self._beat_pulse_alpha = 1.0
        QTimer.singleShot(150, self._fade_beat_pulse)

    def _fade_beat_pulse(self):
        self._beat_pulse_alpha = 0.0
        self.update()

    # ── mouse events ──────────────────────────────────────────────────────

    def mousePressEvent(self, event):
        pos = event.position()
        x, y = pos.x(), pos.y()

        # Click on minimap -> seek
        note_h = self._note_area_height()
        mm_y = note_h + self.HEADER_HEIGHT
        if y >= mm_y and x >= self.PIANO_WIDTH:
            if self.notes:
                min_tick = min(n.tick for n in self.notes)
                max_tick = max(n.tick + n.duration for n in self.notes)
                mm_w = self.width() - self.PIANO_WIDTH
                tick_range = max(max_tick - min_tick, TICKS_PER_BEAT)
                clicked_tick = min_tick + (x - self.PIANO_WIDTH) / mm_w * tick_range
                self.playbackPositionChanged.emit(int(clicked_tick))
            return

        if x < self.PIANO_WIDTH and y > self.HEADER_HEIGHT and y < mm_y:
            self._piano_key_pressed.emit(self._y_to_pitch(y))
            return
        if x < self.PIANO_WIDTH or y < self.HEADER_HEIGHT:
            return

        if event.button() == Qt.LeftButton:
            if self.tool == "draw":
                pitch = self._y_to_pitch(y)
                tick = self._x_to_tick(x)
                snap = self.snap_ticks
                tick = (tick // snap) * snap
                n = MidiNote(tick=tick, pitch=pitch, velocity=100, duration=snap)
                self.undo.save_state(self.notes)
                self.notes.append(n)
                self.selected_ids = {n.id}
                self._last_draw_pitch = pitch
                self._dragging = True
                self._drag_type = "draw"
                self.notesChanged.emit()
                self.noteSelected.emit(list(self.selected_ids))
                self.update()
                return

            if self.tool == "erase":
                hit = self._note_at(pos)
                if hit:
                    note, _ = hit
                    self.undo.save_state(self.notes)
                    self.notes = [n for n in self.notes if n.id != note.id]
                    self.selected_ids.discard(note.id)
                    self.notesChanged.emit()
                    self.noteSelected.emit(list(self.selected_ids))
                    self.update()

            elif self.tool == "select":
                hit = self._note_at(pos)
                if hit:
                    note, zone = hit
                    if event.modifiers() & (Qt.ControlModifier | Qt.ShiftModifier):
                        if note.id in self.selected_ids:
                            self.selected_ids.discard(note.id)
                        else:
                            self.selected_ids.add(note.id)
                    else:
                        if note.id not in self.selected_ids:
                            self.selected_ids = {note.id}
                    self.noteSelected.emit(list(self.selected_ids))
                    self.undo.save_state(self.notes)
                    self._dragging = True
                    if zone == "vel_bar":
                        self._drag_type = "vel_drag"
                        self._vel_drag_note = note
                        self._vel_drag_start_y = pos.y()
                        self._vel_drag_orig_vel = note.velocity
                    else:
                        self._drag_type = "resize_left" if zone == "left_edge" else (
                            "resize_right" if zone == "right_edge" else "move")
                    self._drag_start = pos
                    self._drag_note_origins = {
                        n.id: (n.tick, n.pitch, n.duration)
                        for n in self.notes if n.id in self.selected_ids
                    }
                else:
                    if not (event.modifiers() & (Qt.ControlModifier | Qt.ShiftModifier)):
                        self.selected_ids.clear()
                    self._dragging = True
                    self._drag_type = "select_rect"
                    self._select_rect = QRectF(x, y, 0, 0)
                    self._drag_start = pos
                    self.noteSelected.emit(list(self.selected_ids))
                self.update()

        elif event.button() == Qt.MiddleButton:
            self._dragging = True
            self._drag_type = "pan"
            self._drag_start = pos
            self.setCursor(Qt.ClosedHandCursor)

        elif event.button() == Qt.RightButton:
            hit = self._note_at(pos)
            if hit:
                note, _ = hit
                self.selected_ids = {note.id}
                self.noteSelected.emit(list(self.selected_ids))
                self._show_context_menu(pos, note)
                self.update()

    def mouseMoveEvent(self, event):
        pos = event.position()
        x, y = pos.x(), pos.y()
        mm_y = self._note_area_height() + self.HEADER_HEIGHT

        if x >= self.PIANO_WIDTH and self.HEADER_HEIGHT <= y < mm_y:
            hit = self._note_at(pos)
            if self.tool == "select":
                if hit:
                    _, z = hit
                    if z == "vel_bar":
                        self.setCursor(Qt.SizeVerCursor)
                    elif z != "body":
                        self.setCursor(Qt.SizeHorCursor)
                    else:
                        self.setCursor(Qt.SizeAllCursor)
                else:
                    self.setCursor(Qt.ArrowCursor)
            elif self.tool == "draw":
                self.setCursor(Qt.CrossCursor)
            elif self.tool == "erase":
                self.setCursor(Qt.PointingHandCursor)
        elif x < self.PIANO_WIDTH and self.HEADER_HEIGHT < y < mm_y:
            self.setCursor(Qt.PointingHandCursor)
        elif y >= mm_y and x >= self.PIANO_WIDTH:
            self.setCursor(Qt.PointingHandCursor)
        else:
            self.setCursor(Qt.ArrowCursor)

        if not self._dragging:
            hit = self._note_at(pos) if (x >= self.PIANO_WIDTH and self.HEADER_HEIGHT <= y < mm_y) else None
            new_hover = hit[0].id if hit else None
            if new_hover != self._hover_note_id:
                self._hover_note_id = new_hover
                self.update()
            return

        if not self._drag_start:
            return

        dx = pos.x() - self._drag_start.x()
        dy = pos.y() - self._drag_start.y()

        if self._drag_type == "move":
            tick_d = int(dx / max(self.ppq, 0.001))
            pitch_d = int(-dy / max(self.note_height, 1))
            snap = self.snap_ticks
            tick_d = (tick_d // snap) * snap
            for n in self.notes:
                if n.id in self._drag_note_origins:
                    ot, op, od = self._drag_note_origins[n.id]
                    n.tick = max(0, ot + tick_d)
                    n.pitch = max(0, min(127, op + pitch_d))
            self.notesChanged.emit()
            self.update()

        elif self._drag_type == "resize_right":
            td = int(dx / max(self.ppq, 0.001))
            snap = self.snap_ticks
            td = (td // snap) * snap
            for n in self.notes:
                if n.id in self._drag_note_origins:
                    n.duration = max(snap, self._drag_note_origins[n.id][2] + td)
            self.notesChanged.emit()
            self.update()

        elif self._drag_type == "resize_left":
            td = int(dx / max(self.ppq, 0.001))
            snap = self.snap_ticks
            td = (td // snap) * snap
            for n in self.notes:
                if n.id in self._drag_note_origins:
                    ot, _, od = self._drag_note_origins[n.id]
                    nt = max(0, ot + td)
                    n.duration = max(snap, od - (nt - ot))
                    n.tick = nt
            self.notesChanged.emit()
            self.update()

        elif self._drag_type == "select_rect":
            sx, sy = self._drag_start.x(), self._drag_start.y()
            self._select_rect = QRectF(min(sx, x), min(sy, y), abs(dx), abs(dy))
            self.selected_ids = {n.id for n in self._notes_in_rect(self._select_rect)}
            self.noteSelected.emit(list(self.selected_ids))
            self.update()

        elif self._drag_type == "draw":
            pitch = self._y_to_pitch(y)
            if pitch != self._last_draw_pitch:
                tick = self._x_to_tick(x)
                snap = self.snap_ticks
                tick = (tick // snap) * snap
                n = MidiNote(tick=tick, pitch=pitch, velocity=100, duration=snap)
                self.notes.append(n)
                self._last_draw_pitch = pitch
                self.notesChanged.emit()
                self.update()

        elif self._drag_type == "vel_drag":
            if self._vel_drag_note and self._vel_drag_note.id in self.selected_ids:
                dy = self._vel_drag_start_y - pos.y()
                vel_change = int(dy / 1.5)
                new_vel = max(1, min(127, self._vel_drag_orig_vel + vel_change))
                for n in self.notes:
                    if n.id in self.selected_ids:
                        n.velocity = new_vel
                self.noteSelected.emit(self.get_selected_notes())
                self.notesChanged.emit()
                self.update()

        elif self._drag_type == "pan":
            dx = pos.x() - self._drag_start.x()
            dy = pos.y() - self._drag_start.y()
            self.scroll_x = max(0, self.scroll_x - dx)
            self.scroll_y = max(0, self.scroll_y - dy)
            self._drag_start = pos
            self.update()

    def mouseReleaseEvent(self, event):
        if self._drag_type == "pan":
            self.setCursor(Qt.ArrowCursor)
        self._dragging = False
        self._drag_type = None
        self._drag_start = None
        self._drag_note_origins = {}
        self._select_rect = None
        self._last_draw_pitch = -1
        self.update()

    def wheelEvent(self, event):
        delta = event.angleDelta()
        if event.modifiers() & Qt.ControlModifier:
            self.ppq = max(self.MIN_PPQ, min(self.MAX_PPQ, self.ppq * (1.15 if delta.y() > 0 else 1/1.15)))
        elif event.modifiers() & Qt.ShiftModifier:
            self.scroll_x = max(0, self.scroll_x - delta.y() * 0.5 * self.ppq)
        else:
            self.scroll_y = max(0, self.scroll_y + delta.y() * 0.5)
        self.update()

    def keyPressEvent(self, event):
        key = event.key()
        mod = event.modifiers()
        if key in (Qt.Key_Delete, Qt.Key_Backspace):
            if self.selected_ids:
                self.undo.save_state(self.notes)
                self.notes = [n for n in self.notes if n.id not in self.selected_ids]
                self.selected_ids.clear()
                self.notesChanged.emit()
                self.noteSelected.emit([])
                self.update()
        elif key == Qt.Key_A and mod & Qt.ControlModifier and not (mod & Qt.ShiftModifier):
            self.selected_ids = {n.id for n in self.notes}
            self.noteSelected.emit(list(self.selected_ids))
            self.update()
        elif key == Qt.Key_D and mod & Qt.ControlModifier:
            self.undo.save_state(self.notes)
            new_notes = [MidiNote(tick=n.tick + TICKS_PER_BEAT * 4, pitch=n.pitch,
                                   velocity=n.velocity, duration=n.duration) for n in self.notes if n.id in self.selected_ids]
            self.notes.extend(new_notes)
            self.selected_ids = {n.id for n in new_notes}
            self.notesChanged.emit()
            self.noteSelected.emit(list(self.selected_ids))
            self.update()
        elif key == Qt.Key_Z and mod & Qt.ControlModifier:
            if mod & Qt.ShiftModifier:
                self.redo_notes()
            else:
                self.undo_notes()
        elif key == Qt.Key_Y and mod & Qt.ControlModifier:
            self.redo_notes()
        elif key == Qt.Key_C and mod & Qt.ControlModifier:
            self._copy_selected()
        elif key == Qt.Key_X and mod & Qt.ControlModifier:
            self._copy_selected(cut=True)
        elif key == Qt.Key_V and mod & Qt.ControlModifier:
            self._paste_notes()
        elif key == Qt.Key_T and mod & Qt.ControlModifier:
            self._transpose_dialog()
        elif key == Qt.Key_Up and mod & (Qt.ControlModifier | Qt.ShiftModifier):
            self._transpose_selected(1)
        elif key == Qt.Key_Down and mod & (Qt.ControlModifier | Qt.ShiftModifier):
            self._transpose_selected(-1)
        else:
            super().keyPressEvent(event)

    def _show_context_menu(self, pos, note: MidiNote):
        menu = QMenu(self)
        menu.setStyleSheet("QMenu{background:#1e1e32;color:#c8c8d8;border:1px solid #333;padding:4px;}"
                          "QMenu::item{padding:6px 24px;}QMenu::item:selected{background:#3a3a5c;}")
        actions = {menu.addAction("Delete Note"): "del",
                   menu.addAction("Duplicate"): "dup",
                   menu.addAction("Velocity +10"): "vu",
                   menu.addAction("Velocity -10"): "vd"}
        action = menu.exec(pos.toPoint())
        if action == actions["del"]:
            self.notes = [n for n in self.notes if n.id != note.id]
            self.selected_ids.clear()
        elif action == actions["dup"]:
            nn = MidiNote(tick=note.tick + TICKS_PER_BEAT, pitch=note.pitch,
                          velocity=note.velocity, duration=note.duration)
            self.notes.append(nn)
            self.selected_ids = {nn.id}
        elif action == actions["vu"]:
            for n in self.notes:
                if n.id in self.selected_ids: n.velocity = min(127, n.velocity + 10)
        elif action == actions["vd"]:
            for n in self.notes:
                if n.id in self.selected_ids: n.velocity = max(1, n.velocity - 10)
        self.notesChanged.emit()
        self.noteSelected.emit(list(self.selected_ids))
        self.update()

    # ── Undo/Redo ────────────────────────────────────────────
    def undo_notes(self):
        if self.undo.can_undo():
            self.notes = self.undo.undo(self.notes)
            self.selected_ids.clear()
            self._update_total_ticks()
            self.notesChanged.emit()
            self.noteSelected.emit([])
            self.update()

    def redo_notes(self):
        if self.undo.can_redo():
            self.notes = self.undo.redo(self.notes)
            self.selected_ids.clear()
            self._update_total_ticks()
            self.notesChanged.emit()
            self.noteSelected.emit([])
            self.update()

    # ── Copy/Paste ──────────────────────────────────────────────
    def _copy_selected(self, cut: bool = False):
        sel = self.get_selected_notes()
        if not sel:
            return
        self._clipboard = [asdict(n) for n in sel]
        if cut:
            self.undo.save_state(self.notes)
            self.notes = [n for n in self.notes if n.id not in self.selected_ids]
            self.selected_ids.clear()
            self.notesChanged.emit()
            self.noteSelected.emit([])
            self.update()

    def _paste_notes(self):
        if not self._clipboard:
            return
        self.undo.save_state(self.notes)
        # Find the earliest tick to offset paste position
        min_tick = min(d["tick"] for d in self._clipboard)
        # Paste after the last note or at playback position
        if self.notes:
            paste_at = max(n.tick + n.duration for n in self.notes)
            paste_at = max(paste_at, self.playback_tick)
        else:
            paste_at = self.playback_tick
        offset = paste_at - min_tick
        new_notes = []
        for d in self._clipboard:
            nn = MidiNote(tick=d["tick"] + offset, pitch=d["pitch"],
                          velocity=d["velocity"], duration=d["duration"])
            new_notes.append(nn)
        self.notes.extend(new_notes)
        self.selected_ids = {n.id for n in new_notes}
        self._update_total_ticks()
        self.notesChanged.emit()
        self.noteSelected.emit(list(self.selected_ids))
        self.update()

    # ── Transpose ───────────────────────────────────────────────
    def _transpose_selected(self, semitones: int):
        if not self.selected_ids or semitones == 0:
            return
        self.undo.save_state(self.notes)
        for n in self.notes:
            if n.id in self.selected_ids:
                n.pitch = max(0, min(127, n.pitch + semitones))
        self.notesChanged.emit()
        self.noteSelected.emit(self.get_selected_notes())
        self.update()

    def _transpose_dialog(self):
        if not self.selected_ids:
            return
        from PySide6.QtWidgets import QInputDialog
        semitones, ok = QInputDialog.getInt(
            self, "Transpose", "Semitones (+/-):", 0, -60, 60, 1)
        if ok and semitones != 0:
            self._transpose_selected(semitones)

    # ── Improved context menu ───────────────────────────────────
    def _show_context_menu(self, pos, note: MidiNote):
        menu = QMenu(self)
        menu.setStyleSheet("QMenu{background:#1e1e32;color:#c8c8d8;border:1px solid #333;padding:4px;}"
                          "QMenu::item{padding:6px 24px;}QMenu::item:selected{background:#3a3a5c;}"
                          "QMenu::separator{height:1px;background:#333;margin:4px 8px;}")

        act_del = menu.addAction("Delete Note")
        act_dup = menu.addAction("Duplicate")
        menu.addSeparator()
        act_vu = menu.addAction("Velocity +10")
        act_vd = menu.addAction("Velocity -10")
        menu.addSeparator()
        act_tu = menu.addAction("Transpose +1")
        act_td = menu.addAction("Transpose -1")
        act_td5 = menu.addAction("Transpose +5 (4th)")
        act_to = menu.addAction("Transpose +7 (5th)")
        act_to12 = menu.addAction("Transpose +12 (octave)")
        menu.addSeparator()
        act_copy = menu.addAction("Copy")
        act_cut = menu.addAction("Cut")
        menu.addSeparator()
        act_qu = menu.addAction("Quantize")

        action = menu.exec(pos.toPoint())
        if action == act_del:
            self.undo.save_state(self.notes)
            self.notes = [n for n in self.notes if n.id not in self.selected_ids]
            self.selected_ids.clear()
        elif action == act_dup:
            self.undo.save_state(self.notes)
            nn = MidiNote(tick=note.tick + TICKS_PER_BEAT, pitch=note.pitch,
                          velocity=note.velocity, duration=note.duration)
            self.notes.append(nn)
            self.selected_ids = {nn.id}
        elif action == act_vu:
            self.undo.save_state(self.notes)
            for n in self.notes:
                if n.id in self.selected_ids: n.velocity = min(127, n.velocity + 10)
        elif action == act_vd:
            self.undo.save_state(self.notes)
            for n in self.notes:
                if n.id in self.selected_ids: n.velocity = max(1, n.velocity - 10)
        elif action == act_tu:
            self._transpose_selected(1)
            return
        elif action == act_td:
            self._transpose_selected(-1)
            return
        elif action == act_td5:
            self._transpose_selected(5)
            return
        elif action == act_to:
            self._transpose_selected(7)
            return
        elif action == act_to12:
            self._transpose_selected(12)
            return
        elif action == act_copy:
            self._copy_selected()
            return
        elif action == act_cut:
            self._copy_selected(cut=True)
            return
        elif action == act_qu:
            snap = self.snap_ticks
            self.undo.save_state(self.notes)
            for n in self.notes:
                if n.id in self.selected_ids:
                    n.tick = round(n.tick / snap) * snap
            self.notesChanged.emit()
            self.noteSelected.emit(self.get_selected_notes())
            self.update()
            return
        self.notesChanged.emit()
        self.noteSelected.emit(list(self.selected_ids))
        self.update()

    _piano_key_pressed = Signal(int)

    def set_notes(self, notes: List[MidiNote]):
        self.notes = notes
        self.selected_ids.clear()
        self.undo.clear()
        self._clipboard.clear()
        self._update_total_ticks()
        self.update()

    def _update_total_ticks(self):
        if self.notes:
            mt = max(n.tick + n.duration for n in self.notes)
            self.total_ticks = max(TICKS_PER_BEAT * 16, ((mt // (TICKS_PER_BEAT * 4)) + 2) * TICKS_PER_BEAT * 4)
        else:
            self.total_ticks = TICKS_PER_BEAT * 16

    def get_selected_notes(self) -> List[MidiNote]:
        return [n for n in self.notes if n.id in self.selected_ids]

    def delete_selected(self):
        self.notes = [n for n in self.notes if n.id not in self.selected_ids]
        self.selected_ids.clear()
        self.notesChanged.emit()
        self.noteSelected.emit([])
        self.update()

    def scroll_to_tick(self, tick: int):
        x = self.PIANO_WIDTH + tick * self.ppq - self.scroll_x
        w = self.width() - self.PIANO_WIDTH
        if x > w or x < 0:
            self.scroll_x = tick * self.ppq - w // 3
        self.update()


# ═══════════════════════════════════════════════════════════════════════════
#  Transport Bar
# ═══════════════════════════════════════════════════════════════════════════

class TransportBar(QFrame):
    playToggled = Signal(bool)
    tempoChanged = Signal(int)
    positionChanged = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet("TransportBar{background:#14142a;border-top:1px solid #2a2a44;}")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(8, 4, 8, 4)
        lay.setSpacing(6)

        btn_css = ("QPushButton{background:#22223a;color:#c8c8d8;border:1px solid #3a3a5c;"
                    "border-radius:4px;font-size:14px;}"
                    "QPushButton:hover{background:#2e2e4a;}"
                    "QPushButton:checked{background:#4a3a7a;border-color:#7c5cfc;}")

        self.btn_stop = QPushButton("⏹"); self.btn_play = QPushButton("▶"); self.btn_loop = QPushButton("🔁")
        for b in (self.btn_stop, self.btn_play, self.btn_loop):
            b.setFixedSize(36, 28); b.setStyleSheet(btn_css)
        self.btn_stop.clicked.connect(lambda: self.playToggled.emit(False))
        self.btn_play.clicked.connect(self._toggle_play)
        self.btn_loop.setCheckable(True)
        lay.addWidget(self.btn_stop); lay.addWidget(self.btn_play); lay.addWidget(self.btn_loop)

        self.pos_label = QLabel("1:1:0")
        self.pos_label.setFixedWidth(80)
        self.pos_label.setStyleSheet("color:#a0a0c0;font-family:monospace;font-size:12px;")
        lay.addWidget(self.pos_label)

        self.pos_slider = QSlider(Qt.Horizontal)
        self.pos_slider.setRange(0, TICKS_PER_BEAT * 64)
        self.pos_slider.setValue(0)
        self.pos_slider.setStyleSheet(
            "QSlider::groove:horizontal{background:#22223a;height:4px;border-radius:2px;}"
            "QSlider::handle:horizontal{background:#7c5cfc;width:12px;height:12px;margin:-4px 0;border-radius:6px;}")
        self.pos_slider.sliderMoved.connect(lambda v: self.positionChanged.emit(v))
        lay.addWidget(self.pos_slider, 1)

        lay.addWidget(QLabel("BPM:"))
        self.tempo_spin = QSpinBox()
        self.tempo_spin.setRange(20, 300); self.tempo_spin.setValue(120)
        self.tempo_spin.setFixedWidth(70)
        self.tempo_spin.setStyleSheet("QSpinBox{background:#22223a;color:#c8c8d8;border:1px solid #3a3a5c;"
                                       "border-radius:3px;padding:2px;}")
        self.tempo_spin.valueChanged.connect(self.tempoChanged.emit)
        lay.addWidget(self.tempo_spin)

        lay.addWidget(QLabel("Wave:"))
        self.wave_combo = QComboBox()
        self.wave_combo.addItems(["piano", "bass", "strings", "sine", "triangle", "sawtooth", "square", "organ", "pad"])
        self.wave_combo.setFixedWidth(90)
        self.wave_combo.setStyleSheet(
            "QComboBox{background:#22223a;color:#c8c8d8;border:1px solid #3a3a5c;border-radius:3px;padding:2px 6px;}"
            "QComboBox::drop-down{border:none;}"
            "QComboBox QAbstractItemView{background:#1e1e32;color:#c8c8d8;selection-background-color:#3a3a5c;}")
        lay.addWidget(self.wave_combo)

        # Volume
        lay.addWidget(QLabel("Vol:"))
        self.vol_slider = QSlider(Qt.Horizontal)
        self.vol_slider.setRange(0, 100); self.vol_slider.setValue(70)
        self.vol_slider.setFixedWidth(80)
        self.vol_slider.setStyleSheet(
            "QSlider::groove:horizontal{background:#22223a;height:4px;border-radius:2px;}"
            "QSlider::handle:horizontal{background:#5c8cfc;width:10px;height:10px;margin:-3px 0;border-radius:5px;}")
        lay.addWidget(self.vol_slider)

    def _toggle_play(self):
        playing = self.btn_play.text() == "⏸"
        self.btn_play.setText("⏸" if not playing else "▶")
        self.playToggled.emit(not playing)

    def set_playing(self, playing: bool):
        self.btn_play.setText("⏸" if playing else "▶")

    def update_position(self, tick: int, total_ticks: int):
        self.pos_slider.blockSignals(True)
        self.pos_slider.setMaximum(max(total_ticks, 1))
        self.pos_slider.setValue(tick)
        self.pos_slider.blockSignals(False)
        bar = tick // (TICKS_PER_BEAT * 4) + 1
        beat = (tick // TICKS_PER_BEAT) % 4 + 1
        sub = (tick % TICKS_PER_BEAT) // (TICKS_PER_BEAT // 4)
        self.pos_label.setText(f"{bar}:{beat}:{sub}")


# ═══════════════════════════════════════════════════════════════════════════
#  Model Manager Panel
# ═══════════════════════════════════════════════════════════════════════════

class ModelManagerPanel(QGroupBox):
    modelLoaded = Signal()
    modelStatusChanged = Signal(str)

    def __init__(self, llm: MidiLLMGenerator, parent=None):
        super().__init__("🤖 Model Manager", parent)
        self.llm = llm
        self._download_thread = None
        self._download_cancel = False
        self.setStyleSheet(self._group_style())
        lay = QVBoxLayout(self)
        lay.setSpacing(6)

        # Status indicator
        status_row = QHBoxLayout()
        self.status_dot = QLabel("●")
        self.status_dot.setFixedWidth(16)
        self.status_label = QLabel("Not loaded")
        self.status_label.setStyleSheet("color: #a0a0c8;")
        status_row.addWidget(self.status_dot)
        status_row.addWidget(self.status_label, 1)
        lay.addLayout(status_row)

        # Model path
        lay.addWidget(QLabel("Model Path:"))
        path_row = QHBoxLayout()
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("/path/to/model.gguf or use Browse...")
        self.path_edit.setStyleSheet(self._text_style())
        path_row.addWidget(self.path_edit)
        self.btn_browse = QPushButton("Browse")
        self.btn_browse.setStyleSheet(self._btn_small())
        self.btn_browse.clicked.connect(self._browse)
        path_row.addWidget(self.btn_browse)
        lay.addLayout(path_row)

        # Load / Unload buttons
        btn_row = QHBoxLayout()
        self.btn_load = QPushButton("Load Model")
        self.btn_load.setStyleSheet(self._btn_primary())
        self.btn_load.clicked.connect(self._load)
        self.btn_unload = QPushButton("Unload")
        self.btn_unload.setStyleSheet(self._btn_small())
        self.btn_unload.clicked.connect(self._unload)
        self.btn_unload.setEnabled(False)
        btn_row.addWidget(self.btn_load)
        btn_row.addWidget(self.btn_unload)
        lay.addLayout(btn_row)

        # Error display
        self.error_label = QLabel("")
        self.error_label.setWordWrap(True)
        self.error_label.setStyleSheet("color: #ff6b6b; font-size: 11px;")
        self.error_label.setVisible(False)
        lay.addWidget(self.error_label)

        # Model parameters
        params_row = QHBoxLayout()
        params_row.addWidget(QLabel("n_ctx:"))
        self.ctx_spin = QSpinBox()
        self.ctx_spin.setRange(512, 32768); self.ctx_spin.setValue(4096)
        self.ctx_spin.setSingleStep(512)
        self.ctx_spin.setStyleSheet(self._spin_style())
        self.ctx_spin.setToolTip("Context window size in tokens")
        self.ctx_spin.valueChanged.connect(lambda v: setattr(self.llm, 'n_ctx', v))
        params_row.addWidget(self.ctx_spin)
        params_row.addWidget(QLabel("Threads:"))
        self.threads_spin = QSpinBox()
        self.threads_spin.setRange(1, 32); self.threads_spin.setValue(os.cpu_count() or 4)
        self.threads_spin.setStyleSheet(self._spin_style())
        self.threads_spin.setToolTip("Number of CPU threads for inference")
        self.threads_spin.valueChanged.connect(lambda v: setattr(self.llm, 'n_threads', v))
        params_row.addWidget(self.threads_spin)
        lay.addLayout(params_row)

        # ── Download section ──
        lay.addWidget(self._separator())

        dl_header = QHBoxLayout()
        dl_header.addWidget(QLabel("⬇ Quick Download"))
        dl_header.addStretch()
        self.hf_status = QLabel("huggingface_hub: " + ("✅" if HF_AVAILABLE else "❌"))
        self.hf_status.setStyleSheet("color: #7070a0; font-size: 10px;")
        dl_header.addWidget(self.hf_status)
        lay.addLayout(dl_header)

        self.dl_combo = QComboBox()
        for m in RECOMMENDED_MODELS:
            self.dl_combo.addItem(f"{m['name']}  ({m['size_mb']})")
        self.dl_combo.setStyleSheet(self._combo_style())
        lay.addWidget(self.dl_combo)

        self.dl_desc = QLabel("")
        self.dl_desc.setWordWrap(True)
        self.dl_desc.setStyleSheet("color: #8080a0; font-size: 11px;")
        self.dl_combo.currentIndexChanged.connect(self._update_desc)
        lay.addWidget(self.dl_desc)
        self._update_desc(0)

        dl_btn_row = QHBoxLayout()
        self.btn_download = QPushButton("⬇ Download")
        self.btn_download.setStyleSheet(self._btn_secondary())
        self.btn_download.clicked.connect(self._start_download)
        self.btn_cancel_dl = QPushButton("Cancel")
        self.btn_cancel_dl.setStyleSheet(self._btn_small())
        self.btn_cancel_dl.clicked.connect(self._cancel_download)
        self.btn_cancel_dl.setEnabled(False)
        dl_btn_row.addWidget(self.btn_download)
        dl_btn_row.addWidget(self.btn_cancel_dl)
        lay.addLayout(dl_btn_row)

        # Save dir
        sd_row = QHBoxLayout()
        sd_row.addWidget(QLabel("Save to:"))
        self.save_dir_edit = QLineEdit(os.path.expanduser("~/.cache/midi-llm-studio/models"))
        self.save_dir_edit.setStyleSheet(self._text_style())
        sd_row.addWidget(self.save_dir_edit)
        btn_sd = QPushButton("📁")
        btn_sd.setFixedWidth(30)
        btn_sd.setStyleSheet(self._btn_small())
        btn_sd.clicked.connect(self._browse_save_dir)
        sd_row.addWidget(btn_sd)
        lay.addLayout(sd_row)

        # Progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setStyleSheet(
            "QProgressBar{background:#1a1a30;border:1px solid #2a2a44;border-radius:4px;"
            "color:#c8c8d8;text-align:center;height:18px;}"
            "QProgressBar::chunk{background:qlineargradient(x1:0,y1:0,x2:1,y2:0,"
            "stop:0 #5c3cd0,stop:1 #7c5cfc);border-radius:3px;}")
        self.progress_bar.setVisible(False)
        lay.addWidget(self.progress_bar)

        self.dl_status_label = QLabel("")
        self.dl_status_label.setStyleSheet("color: #7090d0; font-size: 11px;")
        lay.addWidget(self.dl_status_label)

        # Model info section
        info_lay = QVBoxLayout()
        info_hdr = QHBoxLayout()
        info_hdr.addWidget(QLabel("\u2139 Model Info"))
        info_hdr.addStretch()
        lay.addLayout(info_hdr)
        self.info_text = QLabel("No model loaded")
        self.info_text.setWordWrap(True)
        self.info_text.setStyleSheet(
            "color:#8080a8;font-size:11px;padding:4px 6px;"
            "background:#12122a;border:1px solid #222244;border-radius:4px;")
        lay.addWidget(self.info_text)

        # Scan local button
        scan_row = QHBoxLayout()
        self.btn_scan = QPushButton("\U0001f50d Scan Folder for .gguf")
        self.btn_scan.setStyleSheet(self._btn_small())
        self.btn_scan.clicked.connect(self._scan_folder)
        scan_row.addWidget(self.btn_scan)
        scan_row.addStretch()
        lay.addLayout(scan_row)

        # Custom download fields
        lay.addWidget(self._separator())
        cd_hdr = QHBoxLayout()
        cd_hdr.addWidget(QLabel("\U0001f517 Custom Download"))
        cd_hdr.addStretch()
        lay.addLayout(cd_hdr)

        lay.addWidget(QLabel("HuggingFace Repo:"))
        self.custom_repo = QLineEdit()
        self.custom_repo.setPlaceholderText("user/model-name")
        self.custom_repo.setStyleSheet(self._text_style())
        lay.addWidget(self.custom_repo)

        lay.addWidget(QLabel("Filename:"))
        self.custom_file = QLineEdit()
        self.custom_file.setPlaceholderText("model-file.Q4_K_M.gguf")
        self.custom_file.setStyleSheet(self._text_style())
        lay.addWidget(self.custom_file)

        self.btn_custom_dl = QPushButton("\u2b07 Download Custom Model")
        self.btn_custom_dl.setStyleSheet(self._btn_secondary())
        self.btn_custom_dl.clicked.connect(self._start_custom_download)
        lay.addWidget(self.btn_custom_dl)

        # Update initial status
        self._refresh_status()

    def _refresh_status(self):
        if self.llm.is_loaded:
            self.status_dot.setStyleSheet("color: #4cff4c; font-size: 16px;")
            self.status_label.setText(self.llm.status_text)
            self.btn_load.setEnabled(False)
            self.btn_unload.setEnabled(True)
            self.path_edit.setText(self.llm.model_path)
            self._update_model_info()
        elif self.llm.is_available:
            self.status_dot.setStyleSheet("color: #ffc44c; font-size: 16px;")
            self.status_label.setText(self.llm.status_text)
            self.btn_load.setEnabled(True)
            self.btn_unload.setEnabled(False)
            if not self.path_edit.text():
                self.path_edit.setText(self.llm.model_path)
            self._update_file_info(self.llm.model_path)
        else:
            self.status_dot.setStyleSheet("color: #ff4c4c; font-size: 16px;")
            self.status_label.setText(self.llm.status_text)
            self.btn_load.setEnabled(bool(self.path_edit.text()))
            self.btn_unload.setEnabled(False)
            self.info_text.setText("No model loaded")

        if not LLAMA_AVAILABLE:
            self.btn_load.setEnabled(False)
            self.error_label.setText("Install llama-cpp-python:\npip install llama-cpp-python")
            self.error_label.setVisible(True)

        self.modelStatusChanged.emit(self.llm.status_text)

    def _update_model_info(self):
        path = self.llm.model_path
        lines = []
        try:
            size_mb = os.path.getsize(path) / (1024 * 1024)
            lines.append(f"File: {os.path.basename(path)}")
            lines.append(f"Size: {size_mb:.1f} MB")
        except OSError:
            pass
        try:
            m = self.llm.model
            if m is not None:
                n_ctx = getattr(m, "n_ctx", None)
                n_embd = getattr(m, "n_embd", None)
                n_layers = getattr(m, "n_layers", None)
                if n_ctx:
                    lines.append(f"Context: {n_ctx} tokens")
                if n_embd and n_layers:
                    params_b = (n_embd * n_embd + n_embd * n_layers * 4 * n_embd) / 1e9
                    lines.append(f"Embed: {n_embd}, Layers: {n_layers}")
                    lines.append(f"~{params_b:.1f}B parameters")
        except Exception:
            pass
        self.info_text.setText("\n".join(lines) if lines else "Model loaded (info unavailable)")

    def _update_file_info(self, path):
        if path and os.path.isfile(path):
            try:
                size_mb = os.path.getsize(path) / (1024 * 1024)
                self.info_text.setText(
                    f"File: {os.path.basename(path)}\n"
                    f"Size: {size_mb:.1f} MB\n\n"
                    f"Click \'Load Model\' to use")
            except OSError:
                self.info_text.setText("File info unavailable")

    def _scan_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder to Scan for Models")
        if not folder:
            return
        found = []
        for root, dirs, files in os.walk(folder):
            for f in files:
                if f.lower().endswith(".gguf"):
                    full = os.path.join(root, f)
                    try:
                        size_mb = os.path.getsize(full) / (1024 * 1024)
                        found.append((f, full, size_mb))
                    except OSError:
                        found.append((f, full, 0))
        if not found:
            self.info_text.setText(f"No .gguf files found in\n{folder}")
            self.info_text.setStyleSheet(
                "color:#ff6b6b;font-size:11px;padding:4px 6px;"
                "background:#12122a;border:1px solid #222244;border-radius:4px;")
            return
        lines = [f"Found {len(found)} model(s):\n"]
        for name, fp, size in sorted(found, key=lambda x: -x[2])[:10]:
            lines.append(f"  {name} ({size:.0f} MB)")
        if len(found) > 10:
            lines.append(f"  ... and {len(found) - 10} more")
        self.info_text.setText("\n".join(lines))
        self.info_text.setStyleSheet(
            "color:#80c8a8;font-size:11px;padding:4px 6px;"
            "background:#12122a;border:1px solid #222244;border-radius:4px;")
        if found:
            best = max(found, key=lambda x: x[2])
            self.path_edit.setText(best[1])
            self.llm.model_path = best[1]
            self._refresh_status()

    def _start_custom_download(self):
        repo = self.custom_repo.text().strip()
        filename = self.custom_file.text().strip()
        if not repo or not filename:
            QMessageBox.warning(self, "Custom Download", "Please enter both repo and filename.")
            return
        if "/" not in repo:
            QMessageBox.warning(self, "Custom Download", "Repo should be in format: user/model-name")
            return
        save_dir = self.save_dir_edit.text().strip()
        if not save_dir:
            QMessageBox.warning(self, "Download", "Please set a save directory.")
            return

        self._download_cancel = False
        self.btn_custom_dl.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.dl_status_label.setText(f"Downloading {filename}...")
        self.dl_status_label.setStyleSheet("color: #7090d0; font-size: 11px;")
        self.error_label.setVisible(False)

        def _do_dl():
            def progress_cb(current, total):
                if self._download_cancel:
                    raise InterruptedError("Cancelled")
                pct = int(current / max(total, 1) * 100)
                QTimer.singleShot(0, lambda: self._on_dl_progress(pct, current, total))
            try:
                p = download_model(repo, filename, save_dir, progress_cb)
                QTimer.singleShot(0, lambda: self._on_dl_complete(True, p))
            except InterruptedError:
                QTimer.singleShot(0, lambda: self._on_dl_complete(False, "Cancelled"))
            except Exception as e:
                QTimer.singleShot(0, lambda: self._on_dl_complete(False, str(e)))

        self._download_thread = threading.Thread(target=_do_dl, daemon=True)
        self._download_thread.start()

    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select GGUF Model", "",
            "GGUF Model Files (*.gguf *.bin);;All Files (*)")
        if path:
            self.path_edit.setText(path)
            self.llm.model_path = path
            self.error_label.setVisible(False)
            self._refresh_status()

    def _browse_save_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select Download Folder")
        if d:
            self.save_dir_edit.setText(d)

    def _load(self):
        path = self.path_edit.text().strip()
        if not path:
            return
        self.llm.model_path = path
        self.btn_load.setEnabled(False)
        self.btn_load.setText("Loading...")
        self.error_label.setVisible(False)
        QApplication.processEvents()

        def _do_load():
            ok = self.llm.load_model(path)
            # Must use signals or direct call since we're in a thread
            # We'll use a timer trick
            QTimer.singleShot(0, lambda: self._on_load_complete(ok))

        threading.Thread(target=_do_load, daemon=True).start()

    def _on_load_complete(self, ok: bool):
        self.btn_load.setText("Load Model")
        if ok:
            self.modelLoaded.emit()
        else:
            self.error_label.setText(f"Failed: {self.llm._load_error}")
            self.error_label.setVisible(True)
        self._refresh_status()

    def _unload(self):
        self.llm.unload_model()
        self.error_label.setVisible(False)
        self._refresh_status()

    def _update_desc(self, idx):
        if 0 <= idx < len(RECOMMENDED_MODELS):
            self.dl_desc.setText(RECOMMENDED_MODELS[idx]["desc"])

    def _start_download(self):
        idx = self.dl_combo.currentIndex()
        if idx < 0 or idx >= len(RECOMMENDED_MODELS):
            return
        model = RECOMMENDED_MODELS[idx]
        save_dir = self.save_dir_edit.text().strip()
        if not save_dir:
            QMessageBox.warning(self, "Download", "Please set a save directory.")
            return

        self._download_cancel = False
        self.btn_download.setEnabled(False)
        self.btn_cancel_dl.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.dl_status_label.setText(f"Downloading {model['name']}...")
        self.error_label.setVisible(False)

        def _do_download():
            def progress_cb(current, total):
                if self._download_cancel:
                    raise InterruptedError("Cancelled")
                pct = int(current / max(total, 1) * 100)
                QTimer.singleShot(0, lambda: self._on_dl_progress(pct, current, total))

            try:
                path = download_model(model["repo"], model["file"], save_dir, progress_cb)
                QTimer.singleShot(0, lambda: self._on_dl_complete(True, path))
            except InterruptedError:
                QTimer.singleShot(0, lambda: self._on_dl_complete(False, "Cancelled"))
            except Exception as e:
                QTimer.singleShot(0, lambda: self._on_dl_complete(False, str(e)))

        self._download_thread = threading.Thread(target=_do_download, daemon=True)
        self._download_thread.start()

    def _on_dl_progress(self, pct, current, total):
        self.progress_bar.setValue(pct)
        mb = current / (1024 * 1024)
        total_mb = total / (1024 * 1024)
        self.dl_status_label.setText(f"Downloading... {mb:.1f} / {total_mb:.1f} MB ({pct}%)")

    def _on_dl_complete(self, ok, info):
        self.progress_bar.setVisible(False)
        self.btn_download.setEnabled(True)
        self.btn_cancel_dl.setEnabled(False)
        if ok:
            self.path_edit.setText(info)
            self.llm.model_path = info
            self.dl_status_label.setText(f"✅ Downloaded: {os.path.basename(info)}")
            self.dl_status_label.setStyleSheet("color: #4cff4c; font-size: 11px;")
            self._refresh_status()
            # Auto-offer to load
            if self.llm.is_available:
                self.dl_status_label.setText(f"✅ Downloaded! Click 'Load Model' to use it.")
        else:
            self.dl_status_label.setText(f"❌ {info}")
            self.dl_status_label.setStyleSheet("color: #ff6b6b; font-size: 11px;")

    def _cancel_download(self):
        self._download_cancel = True
        self.btn_cancel_dl.setEnabled(False)

    @staticmethod
    def _separator():
        f = QFrame()
        f.setFrameShape(QFrame.HLine)
        f.setStyleSheet("color: #2a2a44;")
        return f

    @staticmethod
    def _group_style():
        return ("QGroupBox{color:#a0a0c8;font-weight:bold;border:1px solid #2a2a44;"
                "border-radius:6px;margin-top:8px;padding-top:16px;}"
                "QGroupBox::title{subcontrol-origin:margin;left:10px;}")

    @staticmethod
    def _text_style():
        return ("QLineEdit{background:#1a1a30;color:#c8c8d8;border:1px solid #2a2a44;"
                "border-radius:4px;padding:4px;font-size:12px;}")

    @staticmethod
    def _combo_style():
        return ("QComboBox{background:#1a1a30;color:#c8c8d8;border:1px solid #2a2a44;"
                "border-radius:4px;padding:4px 8px;}"
                "QComboBox::drop-down{border:none;}"
                "QComboBox QAbstractItemView{background:#1e1e32;color:#c8c8d8;selection-background-color:#3a3a5c;}")

    @staticmethod
    def _btn_primary():
        return ("QPushButton{background:#5c3cd0;color:white;border:none;border-radius:6px;"
                "padding:8px;font-size:13px;font-weight:bold;}"
                "QPushButton:hover{background:#6c4ce0;}"
                "QPushButton:disabled{background:#2a2a44;color:#555;}")

    @staticmethod
    def _btn_secondary():
        return ("QPushButton{background:#22223a;color:#b0b0d0;border:1px solid #3a3a5c;"
                "border-radius:4px;padding:8px;font-size:12px;}"
                "QPushButton:hover{background:#2e2e4a;}"
                "QPushButton:disabled{color:#555;}")

    @staticmethod
    def _btn_small():
        return ("QPushButton{background:#22223a;color:#b0b0d0;border:1px solid #3a3a5c;"
                "border-radius:4px;padding:5px 10px;font-size:11px;}"
                "QPushButton:hover{background:#2e2e4a;}")


# ═══════════════════════════════════════════════════════════════════════════
#  Generation / AI Panel
# ═══════════════════════════════════════════════════════════════════════════

class GenerationPanel(QScrollArea):
    generateRequested = Signal(dict)
    adjustRequested = Signal(str)
    adjustRequestedWithMode = Signal(str, str)  # instruction, mode
    continueRequested = Signal(dict)
    chatRequested = Signal(str)
    historyRecalled = Signal(int)
    generationAborted = Signal()
    analyzeRequested = Signal()
    variationRequested = Signal()
    arrangeRequested = Signal()
    chatGenerateRequested = Signal(str)
    melodyFromChordsRequested = Signal()
    fullArrangementRequested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFixedWidth(310)
        self.setStyleSheet("QScrollArea{background:#14142a;border:none;border-right:1px solid #2a2a44;}")

        container = QWidget()
        lay = QVBoxLayout(container)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(8)
        self.setWidget(container)

        title = QLabel("MIDI LLM Studio")
        title.setStyleSheet("color:#e0e0f0;font-size:16px;font-weight:bold;padding:4px;")
        lay.addWidget(title)

        self.tabs = QTabWidget()
        self.tabs.setStyleSheet(
            "QTabWidget::pane{border:1px solid #2a2a44;border-radius:4px;background:#14142a;}"
            "QTabBar::tab{background:#1a1a30;color:#8888aa;padding:6px 10px;border:1px solid #2a2a44;"
            "border-bottom:none;border-top-left-radius:4px;border-top-right-radius:4px;font-size:11px;}"
            "QTabBar::tab:selected{background:#14142a;color:#c8c8d8;border-bottom:2px solid #7c5cfc;}"
        )

        # ===== Generate Tab =====
        gen_widget = QWidget()
        gl = QVBoxLayout(gen_widget)
        gl.setSpacing(5)

        gl.addWidget(QLabel("Prompt:"))
        self.prompt_edit = QTextEdit()
        self.prompt_edit.setPlaceholderText("Describe the music you want...")
        self.prompt_edit.setMaximumHeight(60)
        self.prompt_edit.setStyleSheet(self._text_style())
        gl.addWidget(self.prompt_edit)

        # Prompt presets
        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("Preset:"))
        self.preset_combo = QComboBox()
        self.preset_combo.addItems([
            "Custom", "Upbeat Energetic", "Dark Minor", "Dreamy Ambient",
            "Funky Groove", "Classical Elegance", "Trap Bounce",
            "Jazz Smooth", "Epic Cinematic", "Simple Melody",
            "Reggaeton Vibes", "R&B Soul", "Breakbeat",
        ])
        self.preset_combo.setStyleSheet(self._combo_style())
        preset_row.addWidget(self.preset_combo)
        gl.addLayout(preset_row)

        gl.addWidget(QLabel("Genre:"))
        self.genre_combo = QComboBox()
        self.genre_combo.addItems([
            "pop", "jazz", "blues", "lofi", "edm", "classical", "ambient", "hiphop",
            "reggaeton", "rnb", "rock", "latin", "funk", "soul",
        ])
        self.genre_combo.setStyleSheet(self._combo_style())
        gl.addWidget(self.genre_combo)

        row_key = QHBoxLayout()
        row_key.addWidget(QLabel("Key:"))
        self.key_combo = QComboBox()
        self.key_combo.addItems(["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"])
        self.key_combo.setStyleSheet(self._combo_style())
        row_key.addWidget(self.key_combo)
        row_key.addWidget(QLabel("Scale:"))
        self.scale_combo = QComboBox()
        self.scale_combo.addItems(["major", "minor", "dorian", "mixolydian", "pentatonic", "blues",
                                    "harmonic minor", "melodic minor"])
        self.scale_combo.setStyleSheet(self._combo_style())
        row_key.addWidget(self.scale_combo)
        gl.addLayout(row_key)

        row_params = QHBoxLayout()
        row_params.addWidget(QLabel("Bars:"))
        self.bars_spin = QSpinBox()
        self.bars_spin.setRange(1, 64); self.bars_spin.setValue(4)
        self.bars_spin.setStyleSheet(self._spin_style())
        row_params.addWidget(self.bars_spin)
        row_params.addWidget(QLabel("BPM:"))
        self.tempo_spin = QSpinBox()
        self.tempo_spin.setRange(40, 300); self.tempo_spin.setValue(120)
        self.tempo_spin.setSingleStep(5)
        self.tempo_spin.setStyleSheet(self._spin_style())
        self.tempo_spin.setToolTip("Tempo in BPM")
        row_params.addWidget(self.tempo_spin)
        gl.addLayout(row_params)

        tmpl_row = QHBoxLayout()
        tmpl_row.addWidget(QLabel("Template:"))
        self.template_combo = QComboBox()
        self.template_combo.addItems(list(PROMPT_TEMPLATES.keys()))
        self.template_combo.setStyleSheet(self._combo_style())
        tmpl_row.addWidget(self.template_combo)
        gl.addLayout(tmpl_row)

        gl.addWidget(QLabel("Temperature:"))
        temp_row = QHBoxLayout()
        self.temp_slider = QSlider(Qt.Horizontal)
        self.temp_slider.setRange(10, 150); self.temp_slider.setValue(80)
        self.temp_slider.setStyleSheet(self._slider_style())
        temp_row.addWidget(self.temp_slider)
        self.temp_label = QLabel("0.80")
        self.temp_label.setFixedWidth(35)
        self.temp_label.setStyleSheet("color:#888;")
        self.temp_slider.valueChanged.connect(lambda v: self.temp_label.setText(f"{v/100:.2f}"))
        temp_row.addWidget(self.temp_label)
        gl.addLayout(temp_row)

        # Advanced sampling params (collapsible)
        self.sampling_group = QGroupBox("Advanced Sampling")
        self.sampling_group.setStyleSheet(
            "QGroupBox{color:#8080a8;font-size:10px;border:1px solid #222244;"
            "border-radius:4px;margin-top:6px;padding-top:12px;}"
            "QGroupBox::title{subcontrol-origin:margin;left:8px;}")
        sg_lay = QVBoxLayout(self.sampling_group)
        sg_lay.setSpacing(3)
        sg_lay.setContentsMargins(6, 14, 6, 6)

        # Top-K
        tk_row = QHBoxLayout()
        tk_row.addWidget(QLabel("Top-K:"))
        self.topk_slider = QSlider(Qt.Horizontal)
        self.topk_slider.setRange(1, 100); self.topk_slider.setValue(40)
        self.topk_slider.setStyleSheet(self._slider_style())
        tk_row.addWidget(self.topk_slider)
        self.topk_label = QLabel("40")
        self.topk_label.setFixedWidth(28)
        self.topk_label.setStyleSheet("color:#888;font-size:10px;")
        self.topk_slider.valueChanged.connect(lambda v: self.topk_label.setText(str(v)))
        tk_row.addWidget(self.topk_label)
        sg_lay.addLayout(tk_row)

        # Top-P
        tp_row = QHBoxLayout()
        tp_row.addWidget(QLabel("Top-P:"))
        self.topp_slider = QSlider(Qt.Horizontal)
        self.topp_slider.setRange(10, 100); self.topp_slider.setValue(95)
        self.topp_slider.setStyleSheet(self._slider_style())
        tp_row.addWidget(self.topp_slider)
        self.topp_label = QLabel("0.95")
        self.topp_label.setFixedWidth(28)
        self.topp_label.setStyleSheet("color:#888;font-size:10px;")
        self.topp_slider.valueChanged.connect(lambda v: self.topp_label.setText(f"{v/100:.2f}"))
        tp_row.addWidget(self.topp_label)
        sg_lay.addLayout(tp_row)

        # Repeat Penalty
        rp_row = QHBoxLayout()
        rp_row.addWidget(QLabel("Rep Pen:"))
        self.rep_penalty_slider = QSlider(Qt.Horizontal)
        self.rep_penalty_slider.setRange(100, 200); self.rep_penalty_slider.setValue(110)
        self.rep_penalty_slider.setStyleSheet(self._slider_style())
        rp_row.addWidget(self.rep_penalty_slider)
        self.rep_label = QLabel("1.10")
        self.rep_label.setFixedWidth(28)
        self.rep_label.setStyleSheet("color:#888;font-size:10px;")
        self.rep_penalty_slider.valueChanged.connect(lambda v: self.rep_label.setText(f"{v/100:.2f}"))
        rp_row.addWidget(self.rep_label)
        sg_lay.addLayout(rp_row)

        # Max Tokens
        mt_row = QHBoxLayout()
        mt_row.addWidget(QLabel("Max Tok:"))
        self.max_tok_spin = QSpinBox()
        self.max_tok_spin.setRange(256, 16384); self.max_tok_spin.setValue(4096)
        self.max_tok_spin.setSingleStep(512)
        self.max_tok_spin.setStyleSheet(self._spin_style())
        mt_row.addWidget(self.max_tok_spin)
        sg_lay.addLayout(mt_row)

        # Frequency Penalty
        fp_row = QHBoxLayout()
        fp_row.addWidget(QLabel("Freq Pen:"))
        self.freq_pen_slider = QSlider(Qt.Horizontal)
        self.freq_pen_slider.setRange(0, 200); self.freq_pen_slider.setValue(0)
        self.freq_pen_slider.setStyleSheet(self._slider_style())
        fp_row.addWidget(self.freq_pen_slider)
        self.freq_pen_label = QLabel("0.00")
        self.freq_pen_label.setFixedWidth(28)
        self.freq_pen_label.setStyleSheet("color:#888;font-size:10px;")
        self.freq_pen_slider.valueChanged.connect(lambda v: self.freq_pen_label.setText(f"{v/100:.2f}"))
        fp_row.addWidget(self.freq_pen_label)
        sg_lay.addLayout(fp_row)

        # Presence Penalty
        pp_row = QHBoxLayout()
        pp_row.addWidget(QLabel("Pres Pen:"))
        self.pres_pen_slider = QSlider(Qt.Horizontal)
        self.pres_pen_slider.setRange(0, 200); self.pres_pen_slider.setValue(0)
        self.pres_pen_slider.setStyleSheet(self._slider_style())
        pp_row.addWidget(self.pres_pen_slider)
        self.pres_pen_label = QLabel("0.00")
        self.pres_pen_label.setFixedWidth(28)
        self.pres_pen_label.setStyleSheet("color:#888;font-size:10px;")
        self.pres_pen_slider.valueChanged.connect(lambda v: self.pres_pen_label.setText(f"{v/100:.2f}"))
        pp_row.addWidget(self.pres_pen_label)
        sg_lay.addLayout(pp_row)

        # Min-P
        mp_row = QHBoxLayout()
        mp_row.addWidget(QLabel("Min-P:"))
        self.min_p_slider = QSlider(Qt.Horizontal)
        self.min_p_slider.setRange(0, 100); self.min_p_slider.setValue(5)
        self.min_p_slider.setStyleSheet(self._slider_style())
        mp_row.addWidget(self.min_p_slider)
        self.min_p_label = QLabel("0.05")
        self.min_p_label.setFixedWidth(28)
        self.min_p_label.setStyleSheet("color:#888;font-size:10px;")
        self.min_p_slider.valueChanged.connect(lambda v: self.min_p_label.setText(f"{v/100:.2f}"))
        mp_row.addWidget(self.min_p_label)
        sg_lay.addLayout(mp_row)

        self.sampling_group.setVisible(False)
        btn_toggle_sampling = QPushButton("Advanced Sampling")
        btn_toggle_sampling.setStyleSheet(self._quick_btn_style())
        btn_toggle_sampling.setCheckable(True)
        btn_toggle_sampling.toggled.connect(lambda c: self.sampling_group.setVisible(c))
        gl.addWidget(btn_toggle_sampling)
        gl.addWidget(self.sampling_group)

        gen_btn_row = QHBoxLayout()
        self.btn_generate = QPushButton("Generate MIDI")
        self.btn_generate.setStyleSheet(self._btn_primary())
        self.btn_generate.clicked.connect(self._on_generate)
        gen_btn_row.addWidget(self.btn_generate)
        self.btn_cancel = QPushButton("Stop")
        self.btn_cancel.setStyleSheet(
            "QPushButton{background:#3a1a1a;color:#ff6b6b;border:1px solid #5a2a2a;"
            "border-radius:4px;padding:6px;font-weight:bold;}"
            "QPushButton:hover{background:#4a2020;}"
            "QPushButton:disabled{color:#555;border-color:#333;}")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self._on_cancel)
        gen_btn_row.addWidget(self.btn_cancel)
        gl.addLayout(gen_btn_row)

        layer_label = QLabel("Generate Layer:")
        layer_label.setStyleSheet("color:#a0a0c0;font-size:11px;margin-top:4px;")
        gl.addWidget(layer_label)
        layer_row = QHBoxLayout()
        for layer_name, layer_color in [("Bass", "#4c8cff"), ("Chords", "#4cff8c"), ("Melody", "#ff8c4c")]:
            btn = QPushButton(layer_name)
            btn.setStyleSheet(
                f"QPushButton{{background:{layer_color}22;color:{layer_color};border:1px solid {layer_color}44;"
                f"border-radius:3px;padding:4px 8px;font-size:11px;}}"
                f"QPushButton:hover{{background:{layer_color}44;}}")
            btn.clicked.connect(lambda checked, l=layer_name.lower(): self._on_generate_layer(l))
            layer_row.addWidget(btn)
        gl.addLayout(layer_row)

        extend_row = QHBoxLayout()
        self.btn_extend = QPushButton("Extend")
        self.btn_extend.setStyleSheet(self._btn_secondary())
        self.btn_extend.clicked.connect(self._on_extend)
        extend_row.addWidget(self.btn_extend)
        self.btn_variation = QPushButton("Variation")
        self.btn_variation.setStyleSheet(self._btn_secondary())
        self.btn_variation.clicked.connect(self._on_variation)
        self.btn_variation.setToolTip("Re-generate with slightly varied parameters")
        extend_row.addWidget(self.btn_variation)
        gl.addLayout(extend_row)

        self.btn_arrange = QPushButton("Auto-Arrange (Bass+Chords)")
        self.btn_arrange.setStyleSheet(
            "QPushButton{background:#1a3a2a;color:#4cff8c;border:1px solid #2a5a3a;"
            "border-radius:4px;padding:6px;font-size:11px;font-weight:bold;}"
            "QPushButton:hover{background:#2a4a3a;}"
            "QPushButton:disabled{background:#1a1a30;color:#555;border-color:#333;}")
        self.btn_arrange.clicked.connect(self._on_arrange)
        self.btn_arrange.setToolTip("Generate bass and chord accompaniment for selected melody notes")
        gl.addWidget(self.btn_arrange)

        # New generation buttons
        new_gen_row = QHBoxLayout()
        self.btn_melody_from_chords = QPushButton("Melody from Chords")
        self.btn_melody_from_chords.setStyleSheet(
            "QPushButton{background:#2a2a1a;color:#ffc44c;border:1px solid #4a4a2a;"
            "border-radius:4px;padding:5px;font-size:10px;font-weight:bold;}"
            "QPushButton:hover{background:#3a3a2a;}"
            "QPushButton:disabled{background:#1a1a30;color:#555;border-color:#333;}")
        self.btn_melody_from_chords.clicked.connect(self._on_melody_from_chords)
        self.btn_melody_from_chords.setToolTip("Generate a melody that harmonically fits over existing chords")
        new_gen_row.addWidget(self.btn_melody_from_chords)

        self.btn_full_arrange = QPushButton("Full Arrangement")
        self.btn_full_arrange.setStyleSheet(
            "QPushButton{background:#1a2a3a;color:#4cc4ff;border:1px solid #2a4a5a;"
            "border-radius:4px;padding:5px;font-size:10px;font-weight:bold;}"
            "QPushButton:hover{background:#2a3a4a;}"
            "QPushButton:disabled{background:#1a1a30;color:#555;border-color:#333;}")
        self.btn_full_arrange.clicked.connect(self._on_full_arrange)
        self.btn_full_arrange.setToolTip("Generate a complete arrangement analyzing existing notes and filling gaps")
        new_gen_row.addWidget(self.btn_full_arrange)
        gl.addLayout(new_gen_row)

        self.stats_label = QLabel("")
        self.stats_label.setWordWrap(True)
        self.stats_label.setStyleSheet("color:#7090d0;font-size:10px;padding:2px 4px;"
                                       "background:#12122a;border-radius:3px;")
        gl.addWidget(self.stats_label)

        # Generation progress bar (hidden by default)
        self.gen_progress = QProgressBar()
        self.gen_progress.setRange(0, 100)
        self.gen_progress.setValue(0)
        self.gen_progress.setTextVisible(True)
        self.gen_progress.setFixedHeight(16)
        self.gen_progress.setStyleSheet(
            "QProgressBar{background:#1a1a30;border:1px solid #2a2a44;border-radius:4px;"
            "color:#c8c8d8;text-align:center;font-size:9px;}"
            "QProgressBar::chunk{background:qlineargradient(x1:0,y1:0,x2:1,y2:0,"
            "stop:0 #5c3cd0,stop:1 #7c5cfc);border-radius:3px;}")
        self.gen_progress.setVisible(False)
        gl.addWidget(self.gen_progress)

        self.gen_elapsed_label = QLabel("")
        self.gen_elapsed_label.setStyleSheet("color:#6060a0;font-size:9px;")
        self.gen_elapsed_label.setVisible(False)
        gl.addWidget(self.gen_elapsed_label)

        gl.addStretch()
        self.tabs.addTab(gen_widget, "Generate")

        # ===== AI Adjust Tab =====
        adj_widget = QWidget()
        al = QVBoxLayout(adj_widget)
        al.setSpacing(6)

        # Adjust mode selector
        mode_label = QLabel("Adjust Mode:")
        mode_label.setStyleSheet("color:#a0a0c8;font-size:11px;font-weight:bold;")
        al.addWidget(mode_label)

        self.adjust_mode_combo = QComboBox()
        for mode_key, mode_info in ADJUST_MODES.items():
            self.adjust_mode_combo.addItem(f"{mode_info['label']}  -  {mode_info['desc']}", mode_key)
        self.adjust_mode_combo.setStyleSheet(self._combo_style())
        al.addWidget(self.adjust_mode_combo)

        # Note analysis display
        self.analysis_label = QLabel("Select notes to analyze")
        self.analysis_label.setWordWrap(True)
        self.analysis_label.setStyleSheet(
            "color:#8080a0;font-size:10px;padding:4px 6px;"
            "background:#0e0e1e;border:1px solid #1e1e34;border-radius:4px;"
            "font-family:monospace;")
        self.analysis_label.setMaximumHeight(60)
        al.addWidget(self.analysis_label)

        btn_analyze = QPushButton("Analyze Selection")
        btn_analyze.setStyleSheet(self._quick_btn_style())
        btn_analyze.clicked.connect(self.analyzeRequested.emit)
        al.addWidget(btn_analyze)

        # Instruction input
        self.adjust_edit = QTextEdit()
        self.adjust_edit.setPlaceholderText(
            "e.g. 'Add swing feel', 'Make more legato', 'Humanize dynamics', "
            "'Add passing tones', 'Smooth the contour'...")
        self.adjust_edit.setMaximumHeight(50)
        self.adjust_edit.setStyleSheet(self._text_style())
        al.addWidget(self.adjust_edit)

        # Main adjust button
        self.btn_adjust = QPushButton("AI Adjust Selected")
        self.btn_adjust.setStyleSheet(self._btn_primary())
        self.btn_adjust.clicked.connect(self._on_adjust_with_mode)
        al.addWidget(self.btn_adjust)

        # Quick presets organized by category
        al.addWidget(QLabel("Quick Adjust:"))
        quick_lay1 = QHBoxLayout()
        for text, instr in [("Staccato", "make staccato"), ("Legato", "make legato"),
                             ("Quantize", "quantize to grid"), ("Swing", "add swing feel")]:
            btn = QPushButton(text)
            btn.setStyleSheet(self._quick_btn_style())
            btn.clicked.connect(lambda checked, i=instr: self._quick_adjust(i))
            quick_lay1.addWidget(btn)
        al.addLayout(quick_lay1)

        quick_lay2 = QHBoxLayout()
        for text, instr in [("Humanize", "humanize timing and velocity"),
                             ("Crescendo", "add crescendo"), ("Accent", "accent downbeats"),
                             ("Invert", "invert pitch contour")]:
            btn = QPushButton(text)
            btn.setStyleSheet(self._quick_btn_style())
            btn.clicked.connect(lambda checked, i=instr: self._quick_adjust(i))
            quick_lay2.addWidget(btn)
        al.addLayout(quick_lay2)

        quick_lay3 = QHBoxLayout()
        for text in ["Dense+", "Sparse-", "Smooth", "Retrograde"]:
            btn = QPushButton(text)
            btn.setStyleSheet(self._quick_btn_style())
            btn.clicked.connect(lambda checked, t=text: self._quick_adjust(t))
            quick_lay3.addWidget(btn)
        al.addLayout(quick_lay3)

        # Streaming output display
        al.addWidget(QLabel("LLM Streaming Output:"))
        self.adjust_output = QTextEdit()
        self.adjust_output.setReadOnly(True)
        self.adjust_output.setMaximumHeight(100)
        self.adjust_output.setStyleSheet(
            "QTextEdit{background:#0a0a18;color:#7090d0;border:1px solid #1a1a30;"
            "border-radius:3px;padding:3px;font-family:monospace;font-size:10px;}")
        al.addWidget(self.adjust_output)

        al.addStretch()
        self.tabs.addTab(adj_widget, "AI Adjust")

        # ===== AI Chat Tab =====
        chat_widget = QWidget()
        cl = QVBoxLayout(chat_widget)
        cl.setSpacing(5)

        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setStyleSheet(
            "QTextEdit{background:#0a0a18;color:#c8c8d8;border:1px solid #2a2a44;"
            "border-radius:4px;padding:6px;font-size:12px;}")
        self.chat_display.setPlaceholderText("Chat with the AI about music theory, composition, and MIDI...")
        cl.addWidget(self.chat_display)

        chat_input_row = QHBoxLayout()
        self.chat_input = QLineEdit()
        self.chat_input.setPlaceholderText("Ask about chords, scales, arrangement...")
        self.chat_input.setStyleSheet(self._text_style())
        self.chat_input.returnPressed.connect(self._on_chat_send)
        chat_input_row.addWidget(self.chat_input)
        self.btn_chat_send = QPushButton("Send")
        self.btn_chat_send.setStyleSheet(self._btn_secondary())
        self.btn_chat_send.clicked.connect(self._on_chat_send)
        chat_input_row.addWidget(self.btn_chat_send)
        cl.addLayout(chat_input_row)

        chat_btn_row = QHBoxLayout()
        btn_clear_chat = QPushButton("Clear")
        btn_clear_chat.setStyleSheet(self._quick_btn_style())
        btn_clear_chat.clicked.connect(self._on_clear_chat)
        chat_btn_row.addWidget(btn_clear_chat)
        self.btn_chat_gen = QPushButton("Generate from Chat")
        self.btn_chat_gen.setStyleSheet(
            "QPushButton{background:#2a2a5a;color:#c0c0e0;border:1px solid #3a3a6a;"
            "border-radius:3px;padding:3px 6px;font-size:9px;font-weight:bold;}"
            "QPushButton:hover{background:#3a3a7a;}")
        self.btn_chat_gen.setToolTip("Use last AI response as a generation prompt")
        self.btn_chat_gen.clicked.connect(self._on_chat_generate)
        chat_btn_row.addWidget(self.btn_chat_gen)
        cl.addLayout(chat_btn_row)

        chat_sug_row1 = QHBoxLayout()
        for q in ["Chords in C minor?", "Jazz voicings", "Lofi tips", "Pentatonic scale"]:
            btn = QPushButton(q)
            btn.setStyleSheet("QPushButton{background:#1a1a30;color:#7c5cfc;border:1px solid #2a2a44;"
                             "border-radius:3px;padding:3px 6px;font-size:9px;}"
                             "QPushButton:hover{background:#2a2a4a;}")
            btn.clicked.connect(lambda checked, t=q: self._chat_suggestion(t))
            chat_sug_row1.addWidget(btn)
        cl.addLayout(chat_sug_row1)

        chat_sug_row2 = QHBoxLayout()
        for q in ["EDM build-up", "Modal interchange", "Voice leading", "Groove patterns"]:
            btn = QPushButton(q)
            btn.setStyleSheet("QPushButton{background:#1a1a30;color:#7c5cfc;border:1px solid #2a2a44;"
                             "border-radius:3px;padding:3px 6px;font-size:9px;}"
                             "QPushButton:hover{background:#2a2a4a;}")
            btn.clicked.connect(lambda checked, t=q: self._chat_suggestion(t))
            chat_sug_row2.addWidget(btn)
        cl.addLayout(chat_sug_row2)

        cl.addStretch()
        self.tabs.addTab(chat_widget, "AI Chat")

        # ===== System Prompt Tab =====
        sys_widget = QWidget()
        sl = QVBoxLayout(sys_widget)
        sl.setSpacing(5)

        sl.addWidget(QLabel("Custom System Prompt (overrides auto):"))
        self.sys_prompt_edit = QTextEdit()
        self.sys_prompt_edit.setPlaceholderText(
            "Leave empty to use auto-generated genre-aware prompt.\n"
            "Enter custom prompt to override...")
        self.sys_prompt_edit.setStyleSheet(self._text_style())
        self.sys_prompt_edit.setMaximumHeight(100)
        sl.addWidget(self.sys_prompt_edit)

        btn_row_sys = QHBoxLayout()
        self.btn_apply_sys = QPushButton("Apply")
        self.btn_apply_sys.setStyleSheet(self._btn_secondary())
        btn_row_sys.addWidget(self.btn_apply_sys)
        self.btn_reset_sys = QPushButton("Reset to Auto")
        self.btn_reset_sys.setStyleSheet(self._quick_btn_style())
        self.btn_reset_sys.clicked.connect(lambda: self.sys_prompt_edit.clear())
        btn_row_sys.addWidget(self.btn_reset_sys)
        self.btn_preview_sys = QPushButton("Preview Auto Prompt")
        self.btn_preview_sys.setStyleSheet(self._quick_btn_style())
        self.btn_preview_sys.clicked.connect(self._on_preview_sys_prompt)
        btn_row_sys.addWidget(self.btn_preview_sys)
        sl.addLayout(btn_row_sys)

        self.sys_preview = QTextEdit()
        self.sys_preview.setReadOnly(True)
        self.sys_preview.setStyleSheet(
            "QTextEdit{background:#0a0a18;color:#8080a0;border:1px solid #1a1a30;"
            "border-radius:3px;padding:3px;font-size:10px;}")
        self.sys_preview.setVisible(False)
        sl.addWidget(self.sys_preview)

        sl.addStretch()
        self.tabs.addTab(sys_widget, "Prompt")

        # ===== History Tab =====
        hist_widget = QWidget()
        hl = QVBoxLayout(hist_widget)
        hl.setSpacing(5)

        hl.addWidget(QLabel("Generation History:"))
        self.history_list = QTextEdit()
        self.history_list.setReadOnly(True)
        self.history_list.setStyleSheet(
            "QTextEdit{background:#0a0a18;color:#a0a0c0;border:1px solid #2a2a44;"
            "border-radius:4px;padding:4px;font-size:11px;}")
        hl.addWidget(self.history_list)

        hist_btn_row = QHBoxLayout()
        btn_recall = QPushButton("Recall Latest")
        btn_recall.setStyleSheet(self._btn_secondary())
        btn_recall.clicked.connect(lambda: self.historyRecalled.emit(-1))
        hist_btn_row.addWidget(btn_recall)
        btn_clear_hist = QPushButton("Clear")
        btn_clear_hist.setStyleSheet(self._quick_btn_style())
        btn_clear_hist.clicked.connect(lambda: self.history_list.setPlainText("No generations yet."))
        hist_btn_row.addWidget(btn_clear_hist)
        hl.addLayout(hist_btn_row)

        hl.addStretch()
        self.tabs.addTab(hist_widget, "History")

        # ===== Notes Tab =====
        prop_widget = QWidget()
        pl = QVBoxLayout(prop_widget)
        pl.setSpacing(4)

        self.prop_pitch = self._prop_row(pl, "Pitch:")
        self.prop_velocity = self._prop_row(pl, "Velocity:")
        self.prop_start = self._prop_row(pl, "Start (tick):")
        self.prop_duration = self._prop_row(pl, "Duration (tick):")

        self.btn_apply = QPushButton("Apply Changes")
        self.btn_apply.setStyleSheet(self._btn_secondary())
        pl.addWidget(self.btn_apply)

        self.btn_delete = QPushButton("Delete Selected")
        self.btn_delete.setStyleSheet(
            "QPushButton{background:#3a1a1a;color:#ff6b6b;border:1px solid #5a2a2a;"
            "border-radius:4px;padding:6px;font-weight:bold;}"
            "QPushButton:hover{background:#4a2020;}")
        pl.addWidget(self.btn_delete)

        pl.addStretch()
        self.tabs.addTab(prop_widget, "Notes")

        lay.addWidget(self.tabs)

    def _get_preset_prompt(self):
        idx = self.preset_combo.currentIndex()
        prompts = {
            0: "",  # Custom
            1: "Create an upbeat, energetic track with driving rhythm and bright melodies. Use lively rhythms with eighth note patterns.",
            2: "Create a dark, moody minor key piece. Use low register, minor chords, and mysterious atmosphere.",
            3: "Create a dreamy, atmospheric ambient piece. Use long sustained notes, wide voicings, and slow tempo.",
            4: "Create a funky groove with syncopated bass line, tight rhythm, and percussive chord stabs.",
            5: "Create an elegant classical piece with clear voice leading, arpeggiated patterns, and refined dynamics.",
            6: "Create a trap-style beat with heavy 808 bass, hi-hat rolls, and dark minor melody.",
            7: "Create a smooth jazz piece with extended chords (7ths, 9ths), swing feel, and walking bass.",
            8: "Create an epic cinematic piece with wide orchestral range, building intensity, and dramatic dynamics.",
            9: "Create a simple, clean melody with minimal accompaniment. Focus on a memorable, singable tune.",
            10: "Create a reggaeton-inspired rhythm with syncopated patterns, minor key melody, and bouncy bass.",
            11: "Create an R&B soul ballad with lush 7th and 9th chords, smooth vocal-like melody, and gentle groove.",
            12: "Create a breakbeat with choppy drums, funky bass line, and staccato synth stabs.",
        }
        return prompts.get(idx, "")

    def _on_generate(self):
        preset_prompt = self._get_preset_prompt()
        custom_prompt = self.prompt_edit.toPlainText()
        combined = (preset_prompt + "\n" + custom_prompt).strip()
        self.generateRequested.emit({
            "prompt": combined,
            "genre": self.genre_combo.currentText(),
            "key": self.key_combo.currentText(),
            "scale": self.scale_combo.currentText(),
            "tempo": self.tempo_spin.value(),
            "num_bars": self.bars_spin.value(),
            "temperature": self.temp_slider.value() / 100.0,
            "template": self.template_combo.currentText(),
            "top_k": self.topk_slider.value(),
            "top_p": self.topp_slider.value() / 100.0,
            "repeat_penalty": self.rep_penalty_slider.value() / 100.0,
            "max_tokens": self.max_tok_spin.value(),
            "frequency_penalty": self.freq_pen_slider.value() / 100.0,
            "presence_penalty": self.pres_pen_slider.value() / 100.0,
            "min_p": self.min_p_slider.value() / 100.0,
        })

    def _on_generate_layer(self, layer: str):
        self.generateRequested.emit({
            "prompt": self.prompt_edit.toPlainText(),
            "genre": self.genre_combo.currentText(),
            "key": self.key_combo.currentText(),
            "scale": self.scale_combo.currentText(),
            "tempo": self.tempo_spin.value(),
            "num_bars": self.bars_spin.value(),
            "temperature": self.temp_slider.value() / 100.0,
            "template": self.template_combo.currentText(),
            "layer": layer,
            "top_k": self.topk_slider.value(),
            "top_p": self.topp_slider.value() / 100.0,
            "repeat_penalty": self.rep_penalty_slider.value() / 100.0,
            "max_tokens": self.max_tok_spin.value(),
            "frequency_penalty": self.freq_pen_slider.value() / 100.0,
            "presence_penalty": self.pres_pen_slider.value() / 100.0,
            "min_p": self.min_p_slider.value() / 100.0,
        })

    def _on_extend(self):
        self.continueRequested.emit({
            "prompt": self.prompt_edit.toPlainText(),
            "genre": self.genre_combo.currentText(),
            "key": self.key_combo.currentText(),
            "scale": self.scale_combo.currentText(),
            "tempo": self.tempo_spin.value(),
            "num_bars": self.bars_spin.value(),
            "temperature": self.temp_slider.value() / 100.0,
            "template": self.template_combo.currentText(),
            "top_k": self.topk_slider.value(),
            "top_p": self.topp_slider.value() / 100.0,
            "repeat_penalty": self.rep_penalty_slider.value() / 100.0,
            "max_tokens": self.max_tok_spin.value(),
            "frequency_penalty": self.freq_pen_slider.value() / 100.0,
            "presence_penalty": self.pres_pen_slider.value() / 100.0,
            "min_p": self.min_p_slider.value() / 100.0,
        })

    def _on_cancel(self):
        self.generationAborted.emit()

    def _quick_adjust(self, instruction: str):
        """Quick adjust using current mode from combo."""
        mode = self.adjust_mode_combo.currentData() or "full"
        self.adjustRequestedWithMode.emit(instruction, mode)

    def _on_adjust_with_mode(self):
        mode = self.adjust_mode_combo.currentData() or "full"
        self.adjustRequestedWithMode.emit(self.adjust_edit.toPlainText(), mode)

    def _on_adjust(self):
        self.adjustRequested.emit(self.adjust_edit.toPlainText())

    def _on_chat_send(self):
        text = self.chat_input.text().strip()
        if not text:
            return
        self.chat_input.clear()
        self.chatRequested.emit(text)

    def _chat_suggestion(self, text: str):
        self.chat_input.setText(text)
        self._on_chat_send()

    def _on_clear_chat(self):
        self.chat_display.clear()

    def _on_preview_sys_prompt(self):
        prompt = _build_system_prompt(
            genre=self.genre_combo.currentText(),
            key=self.key_combo.currentText(),
            scale=self.scale_combo.currentText(),
            extra=self.prompt_edit.toPlainText())
        self.sys_preview.setPlainText(prompt)
        self.sys_preview.setVisible(not self.sys_preview.isVisible())

    def _on_variation(self):
        self.variationRequested.emit({})

    def _on_arrange(self):
        self.arrangeRequested.emit()

    def _on_chat_generate(self):
        text = self.chat_display.toPlainText()
        if "AI:" in text:
            last_ai = text.rsplit("AI:", 1)[-1].strip()
            if last_ai:
                self.chatGenerateRequested.emit(last_ai)

    def _on_melody_from_chords(self):
        self.melodyFromChordsRequested.emit()

    def _on_full_arrange(self):
        self.fullArrangementRequested.emit()

    def set_generating(self, is_generating: bool):
        self.btn_generate.setEnabled(not is_generating)
        self.btn_generate.setText("Generating..." if is_generating else "Generate MIDI")
        self.btn_cancel.setEnabled(is_generating)
        self.btn_extend.setEnabled(not is_generating)
        self.btn_variation.setEnabled(not is_generating)
        self.btn_arrange.setEnabled(not is_generating)
        self.btn_melody_from_chords.setEnabled(not is_generating)
        self.btn_full_arrange.setEnabled(not is_generating)
        # Show/hide progress bar
        self.gen_progress.setVisible(is_generating)
        self.gen_elapsed_label.setVisible(is_generating)
        if not is_generating:
            self.gen_progress.setValue(0)
            self.gen_elapsed_label.setText("")

    def update_stats(self, stats):
        self.stats_label.setText(stats.summary)

    def update_adjust_output(self, text: str):
        self.adjust_output.setPlainText(text)

    def update_analysis(self, notes):
        """Update the note analysis display."""
        if not notes:
            self.analysis_label.setText("Select notes to analyze")
            return
        a = analyze_notes(notes)
        chords_str = ""
        if a.get("detected_chords"):
            chords_str = f"\nChords: {', '.join(a['detected_chords'][:5])}"
        time_str = f" | {a.get('time_sig_feel', '4/4')}" if a.get('time_sig_feel') else ""
        self.analysis_label.setText(
            f"Notes: {a['note_count']} | Bars: {a['total_bars']}{time_str} | "
            f"Range: {a['pitch_range']}\n"
            f"Vel: {a['min_velocity']}-{a['max_velocity']} (avg {a['avg_velocity']}) | "
            f"Density: {a['density_profile']} | "
            f"Max sim: {a['max_simultaneous']}\n"
            f"Notes/bar: {a['notes_per_bar']} | "
            f"Dominant: {', '.join(a['dominant_pitch_classes'][:5])}"
            f"{chords_str}"
        )

    def append_chat_message(self, role: str, text: str):
        color = "#80c0ff" if role == "user" else "#c080ff"
        label = "You" if role == "user" else "AI"
        # Simple markdown-like formatting: bold **text**, code `text`
        formatted = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        formatted = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', formatted)
        formatted = re.sub(r'`(.+?)`', r'<span style="color:#7cfc7c;font-family:monospace;">\1</span>', formatted)
        formatted = formatted.replace("\n", "<br>")
        self.chat_display.append(
            f'<span style="color:{color};font-weight:bold;">{label}:</span> '
            f'<span style="color:#c8c8d8;">{formatted}</span>')
        sb = self.chat_display.verticalScrollBar()
        sb.setValue(sb.maximum())

    def update_history(self, history):
        if len(history) == 0:
            self.history_list.setPlainText("No generations yet.")
            return
        lines = []
        for i, (notes, stats) in enumerate(history):
            import time as _time
            t = _time.strftime("%H:%M:%S", _time.localtime(stats.timestamp))
            lines.append(f"[{i+1}] {t} | {stats.summary}")
        self.history_list.setPlainText("\n".join(lines))

    def update_properties(self, notes):
        if len(notes) == 1:
            n = notes[0]
            self.prop_pitch.setValue(n.pitch); self.prop_pitch.setEnabled(True)
            self.prop_velocity.setValue(n.velocity); self.prop_velocity.setEnabled(True)
            self.prop_start.setValue(n.tick); self.prop_start.setEnabled(True)
            self.prop_duration.setValue(n.duration); self.prop_duration.setEnabled(True)
            self.btn_apply.setEnabled(True)
        elif notes:
            self.prop_pitch.setEnabled(False); self.prop_start.setEnabled(False)
            self.prop_duration.setEnabled(False)
            self.prop_velocity.setValue(sum(n.velocity for n in notes) // len(notes))
            self.prop_velocity.setEnabled(True); self.btn_apply.setEnabled(True)
            self.prop_pitch.clear(); self.prop_start.clear(); self.prop_duration.clear()
        else:
            for s in (self.prop_pitch, self.prop_velocity, self.prop_start, self.prop_duration):
                s.clear(); s.setEnabled(False)
            self.btn_apply.setEnabled(False)

    @staticmethod
    def _quick_btn_style():
        return ("QPushButton{background:#1a1a30;color:#a0a0c0;border:1px solid #2a2a44;"
                "border-radius:3px;padding:4px 6px;font-size:10px;}"
                "QPushButton:hover{background:#2a2a4a;}")

    @staticmethod
    def _text_style():
        return ("QTextEdit{background:#1a1a30;color:#c8c8d8;border:1px solid #2a2a44;"
                "border-radius:4px;padding:4px;font-size:12px;}"
                "QLineEdit{background:#1a1a30;color:#c8c8d8;border:1px solid #2a2a44;"
                "border-radius:4px;padding:4px;font-size:12px;}")

    @staticmethod
    def _combo_style():
        return ("QComboBox{background:#1a1a30;color:#c8c8d8;border:1px solid #2a2a44;"
                "border-radius:4px;padding:4px 8px;}"
                "QComboBox::drop-down{border:none;}"
                "QComboBox QAbstractItemView{background:#1a1a30;color:#c8c8d8;"
                "selection-background-color:#7c5cfc;border:1px solid #2a2a44;}")

    @staticmethod
    def _spin_style():
        return ("QSpinBox{background:#1a1a30;color:#c8c8d8;border:1px solid #2a2a44;"
                "border-radius:4px;padding:3px;}")

    @staticmethod
    def _slider_style():
        return ("QSlider::groove:horizontal{background:#2a2a44;height:6px;border-radius:3px;}"
                "QSlider::handle:horizontal{background:#7c5cfc;width:14px;height:14px;"
                "margin:-4px 0;border-radius:7px;}"
                "QSlider::sub-page:horizontal{background:#5c3cd0;border-radius:3px;}")

    @staticmethod
    def _btn_primary():
        return ("QPushButton{background:#5c3cd0;color:white;border:none;"
                "border-radius:5px;padding:8px;font-weight:bold;font-size:13px;}"
                "QPushButton:hover{background:#7c5cfc;}"
                "QPushButton:disabled{background:#2a2a44;color:#555;}")

    @staticmethod
    def _btn_secondary():
        return ("QPushButton{background:#2a2a5a;color:#c0c0e0;border:1px solid #3a3a6a;"
                "border-radius:4px;padding:6px;}"
                "QPushButton:hover{background:#3a3a7a;}"
                "QPushButton:disabled{background:#1a1a30;color:#444;}")

    def _prop_row(self, layout, label):
        layout.addWidget(QLabel(label))
        spin = QSpinBox()
        spin.setRange(0, 99999)
        spin.setStyleSheet(self._spin_style())
        layout.addWidget(spin)
        return spin


class ToolBar(QToolBar):
    toolChanged = Signal(str)
    zoomRequested = Signal(float)
    fitRequested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMovable(False)
        self.setIconSize(QSize(20, 20))
        self.setStyleSheet(
            "QToolBar{background:#14142a;border-bottom:1px solid #2a2a44;spacing:2px;padding:2px;}"
            "QToolButton{background:transparent;color:#a0a0c0;border:1px solid transparent;"
            "border-radius:4px;padding:5px 10px;font-size:12px;}"
            "QToolButton:hover{background:#22223a;}"
            "QToolButton:checked{background:#3a3a5c;color:#e0e0f0;border-color:#7c5cfc;}")

        self._tools = {}
        for name, label in [("select", "🖱 Select"), ("draw", "✏️ Draw"), ("erase", "🗑 Erase")]:
            btn = QToolButton(); btn.setText(label); btn.setCheckable(True)
            btn.clicked.connect(lambda _, n=name: self._select_tool(n))
            self.addWidget(btn); self._tools[name] = btn
        self._tools["select"].setChecked(True)

        self.addSeparator()
        for text, factor in [("🔍+", 1.3), ("🔍−", 1/1.3)]:
            btn = QToolButton(); btn.setText(text)
            btn.clicked.connect(lambda _, f=factor: self.zoomRequested.emit(f))
            self.addWidget(btn)
        fit = QToolButton(); fit.setText("Fit"); fit.clicked.connect(self.fitRequested.emit)
        self.addWidget(fit)

        self.addSeparator()

        # Snap setting
        snap_lbl = QLabel("  Snap:")
        snap_lbl.setStyleSheet("color:#8888aa;font-size:11px;padding:0 4px;")
        self.addWidget(snap_lbl)
        self.snap_combo = QComboBox()
        self.snap_combo.addItems(["16th", "8th", "1/4", "1/2", "1 bar"])
        self.snap_combo.setCurrentIndex(0)
        self.snap_combo.setFixedWidth(65)
        self.snap_combo.setStyleSheet(
            "QComboBox{background:#1a1a30;color:#c8c8d8;border:1px solid #3a3a5c;"
            "border-radius:3px;padding:2px 4px;font-size:11px;}"
            "QComboBox::drop-down{border:none;}"
            "QComboBox QAbstractItemView{background:#1e1e32;color:#c8c8d8;selection-background-color:#3a3a5c;}")
        self.addWidget(self.snap_combo)

    def _select_tool(self, name: str):
        for t, b in self._tools.items(): b.setChecked(t == name)
        self.toolChanged.emit(name)

    def set_tool(self, name: str):
        self._select_tool(name)


# ═══════════════════════════════════════════════════════════════════════════
#  Main Window
# ═══════════════════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MIDI LLM Studio")
        self.resize(1300, 780)
        self.setStyleSheet(
            "QMainWindow{background:#0a0a14;}"
            "QMenuBar{background:#14142a;color:#c8c8d8;border-bottom:1px solid #2a2a44;}"
            "QMenuBar::item:selected{background:#2e2e4a;}"
            "QMenu{background:#1a1a2e;color:#c8c8d8;border:1px solid #2a2a44;}"
            "QMenu::item:selected{background:#3a3a5c;}"
            "QStatusBar{background:#0f0f1e;color:#7070a0;border-top:1px solid #2a2a44;}"
            "QLabel{color:#a0a0c8;}")

        self.llm = MidiLLMGenerator()
        self.synth = SynthEngine()
        self.tempo = 120
        self._is_playing = False
        self._play_start_time = 0.0
        self._play_start_tick = 0
        self._last_beat = -1

        self.piano_roll = PianoRollWidget()
        self.gen_panel = GenerationPanel()
        self.model_panel = ModelManagerPanel(self.llm)
        self.transport = TransportBar()

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        self.toolbar = ToolBar()
        self.addToolBar(self.toolbar)

        # Left panel with tabs for Gen + Model Manager
        left_panel = QWidget()
        left_panel.setFixedWidth(290)
        left_lay = QVBoxLayout(left_panel)
        left_lay.setContentsMargins(0, 0, 0, 0)
        left_lay.setSpacing(0)

        panel_tabs = QTabWidget()
        panel_tabs.setStyleSheet(
            "QTabWidget::pane{border:none;}"
            "QTabBar::tab{background:#18182e;color:#7777aa;padding:8px 14px;"
            "border-bottom:2px solid transparent;font-size:11px;}"
            "QTabBar::tab:selected{color:#c8c8d8;border-bottom-color:#7c5cfc;}"
            "QTabBar::tab:hover{color:#b0b0d0;}")

        gen_scroll = QScrollArea()
        gen_scroll.setWidgetResizable(True)
        gen_scroll.setWidget(self.gen_panel)
        gen_scroll.setStyleSheet("QScrollArea{border:none;background:#14142a;}")

        model_scroll = QScrollArea()
        model_scroll.setWidgetResizable(True)
        model_scroll.setWidget(self.model_panel)
        model_scroll.setStyleSheet("QScrollArea{border:none;background:#14142a;}")

        panel_tabs.addTab(gen_scroll, "🎵 Create")
        panel_tabs.addTab(model_scroll, "🤖 Models")

        left_lay.addWidget(panel_tabs)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(self.piano_roll)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([290, 1010])
        main_layout.addWidget(splitter, 1)
        main_layout.addWidget(self.transport)

        self._build_menus()
        self._connect_signals()

        self._play_timer = QTimer()
        self._play_timer.setInterval(16)
        self._play_timer.timeout.connect(self._on_playback_tick)

        self.statusBar().showMessage("Ready — Generate MIDI or import a file to start")

    def _build_menus(self):
        mb = self.menuBar()
        fm = mb.addMenu("File")
        fm.addAction("New", self._new_project, "Ctrl+N")
        fm.addAction("Import MIDI...", self._import_midi, "Ctrl+O")
        fm.addAction("Export MIDI...", self._export_midi, "Ctrl+S")
        fm.addAction("Export WAV...", self._export_wav, "Ctrl+Shift+S")
        fm.addSeparator()
        fm.addAction("Exit", self.close, "Ctrl+Q")

        em = mb.addMenu("Edit")
        em.addAction("Undo", self.piano_roll.undo_notes, "Ctrl+Z")
        em.addAction("Redo", self.piano_roll.redo_notes, "Ctrl+Shift+Z")
        em.addSeparator()
        em.addAction("Cut", lambda: self.piano_roll._copy_selected(cut=True), "Ctrl+X")
        em.addAction("Copy", self.piano_roll._copy_selected, "Ctrl+C")
        em.addAction("Paste", self.piano_roll._paste_notes, "Ctrl+V")
        em.addSeparator()
        em.addAction("Select All", self._select_all, "Ctrl+A")
        em.addAction("Delete Selected", self.piano_roll.delete_selected, "Delete")
        em.addAction("Duplicate", self._duplicate_selected, "Ctrl+D")
        em.addSeparator()
        em.addAction("Quantize", self._quantize_selected)
        em.addAction("Transpose...", self._transpose_dialog, "Ctrl+T")

        vm = mb.addMenu("View")
        vm.addAction("Zoom In", lambda: self._zoom(1.3), "Ctrl+=")
        vm.addAction("Zoom Out", lambda: self._zoom(1/1.3), "Ctrl+-")
        vm.addAction("Fit to Content", self._fit_content)

        hm = mb.addMenu("Help")
        hm.addAction("Keyboard Shortcuts", self._show_shortcuts)
        hm.addAction("About", self._show_about)

    def _connect_signals(self):
        self.piano_roll.noteSelected.connect(self._on_selection_changed)
        self.piano_roll.notesChanged.connect(self._on_notes_changed)
        self.piano_roll._piano_key_pressed.connect(self._preview_note)
        self.piano_roll.playbackPositionChanged.connect(self._seek_to_tick)

        self.gen_panel.generateRequested.connect(self._on_generate)
        self.gen_panel.adjustRequested.connect(self._on_ai_adjust)
        self.gen_panel.adjustRequestedWithMode.connect(self._on_ai_adjust_with_mode)
        self.gen_panel.analyzeRequested.connect(self._on_analyze_selection)
        self.gen_panel.continueRequested.connect(self._on_continue)
        self.gen_panel.chatRequested.connect(self._on_chat)
        self.gen_panel.historyRecalled.connect(self._on_history_recall)
        self.gen_panel.generationAborted.connect(self._on_abort)
        self.gen_panel.variationRequested.connect(self._on_variation)
        self.gen_panel.arrangeRequested.connect(self._on_arrange)
        self.gen_panel.chatGenerateRequested.connect(self._on_chat_generate)
        self.gen_panel.melodyFromChordsRequested.connect(self._on_melody_from_chords)
        self.gen_panel.fullArrangementRequested.connect(self._on_full_arrangement)
        self.gen_panel.btn_delete.clicked.connect(self.piano_roll.delete_selected)
        self.gen_panel.btn_apply.clicked.connect(self._apply_note_properties)

        self.transport.playToggled.connect(self._toggle_playback)
        self.transport.tempoChanged.connect(self._on_tempo_changed)
        self.gen_panel.tempo_spin.valueChanged.connect(self._on_panel_tempo_changed)
        self.transport.positionChanged.connect(self._seek_to_tick)
        self.transport.wave_combo.currentTextChanged.connect(self.synth.set_waveform)
        self.transport.vol_slider.valueChanged.connect(lambda v: setattr(self.synth, 'volume', v / 100.0))

        self.toolbar.toolChanged.connect(lambda t: setattr(self.piano_roll, 'tool', t))
        self.toolbar.zoomRequested.connect(self._zoom)
        self.toolbar.fitRequested.connect(self._fit_content)

        # Snap combo
        snap_map = {0: 4, 1: 2, 2: 1, 3: 8, 4: 16}  # 16th=4, 8th=2, quarter=1, half=8, bar=16
        self.toolbar.snap_combo.currentIndexChanged.connect(
            lambda idx: setattr(self.piano_roll, 'snap_divisor', snap_map.get(idx, 4)))

        # Template sync
        self.gen_panel.template_combo.currentTextChanged.connect(
            lambda t: setattr(self.llm, 'template_name', t))

    # ── File operations ───────────────────────────────────────────────────

    def _new_project(self):
        self._stop_playback()
        self.piano_roll.set_notes([])
        self.statusBar().showMessage("New project")

    def _import_midi(self):
        path, _ = QFileDialog.getOpenFileName(self, "Import MIDI", "", "MIDI (*.mid *.midi);;All (*)")
        if not path: return
        try:
            notes, tempo = import_midi(path)
            self.piano_roll.set_notes(notes)
            self.tempo = tempo
            self.transport.tempo_spin.setValue(tempo)
            self.statusBar().showMessage(f"Imported {len(notes)} notes from {os.path.basename(path)}")
        except Exception as e:
            QMessageBox.critical(self, "Import Error", str(e))

    def _export_midi(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export MIDI", "generated.mid", "MIDI (*.mid);;All (*)")
        if not path: return
        try:
            export_midi(self.piano_roll.notes, self.tempo, path)
            self.statusBar().showMessage(f"Exported to {path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))

    # ── Generation ────────────────────────────────────────────────────────

    def _on_generate(self, params: dict):
        self._stop_playback()
        custom = self.gen_panel.sys_prompt_edit.toPlainText().strip()
        self.llm._custom_system_prompt = custom if custom else ""
        if "template" in params:
            self.llm.template_name = params["template"]
        # Apply sampling parameters
        if "top_k" in params:
            self.llm.top_k = params["top_k"]
        if "top_p" in params:
            self.llm.top_p = params["top_p"]
        if "repeat_penalty" in params:
            self.llm.repeat_penalty = params["repeat_penalty"]
        if "max_tokens" in params:
            self.llm.max_tokens = params["max_tokens"]
        # Set current notes context for awareness
        self.llm._current_notes = self.piano_roll.notes
        self.statusBar().showMessage("Generating MIDI...")
        self.gen_panel.set_generating(True)
        self.gen_panel.update_adjust_output("")
        QApplication.processEvents()

        _gen_start = time.time()
        _expected_dur = 15.0  # rough expected seconds for progress bar

        def _stream_cb(text):
            def _update():
                self.gen_panel.update_adjust_output(text)
                elapsed = time.time() - _gen_start
                note_count = text.count("\n")
                # Update progress bar (estimate based on note count vs expected)
                expected_notes = params.get("num_bars", 4) * 10  # ~10 notes per bar expected
                progress = min(95, int((note_count / max(expected_notes, 1)) * 100))
                self.gen_panel.gen_progress.setValue(progress)
                self.gen_panel.gen_elapsed_label.setText(f"Elapsed: {elapsed:.1f}s")
                self.statusBar().showMessage(
                    f"Generating... {note_count} potential notes | {elapsed:.1f}s")
            QTimer.singleShot(0, _update)

        def _gen():
            notes = self.llm.generate(
                prompt=params["prompt"], genre=params["genre"],
                key=params["key"], scale=params["scale"],
                tempo=self.tempo, num_bars=params["num_bars"],
                temperature=params["temperature"],
                layer=params.get("layer", ""),
                stream_cb=_stream_cb)
            QTimer.singleShot(0, lambda: self._on_gen_done(notes, params))

        threading.Thread(target=_gen, daemon=True).start()

    def _on_gen_done(self, notes, params):
        self.gen_panel.gen_progress.setValue(100)
        self.gen_panel.gen_elapsed_label.setText("")
        self.gen_panel.set_generating(False)
        layer = params.get("layer", "")
        if layer:
            self.piano_roll.notes.extend(notes)
            self.piano_roll.notesChanged.emit()
            self.piano_roll.update()
            self.statusBar().showMessage(f"Added {len(notes)} {layer} notes")
        else:
            self.piano_roll.set_notes(notes)
            mode = self.llm.last_stats.mode if self.llm.last_stats else ("LLM" if self.llm.is_loaded else "Demo")
            self.statusBar().showMessage(
                f"Generated {len(notes)} notes ({mode}) - {params['genre']} in {params['key']} {params['scale']}")
        if self.llm.last_stats:
            self.gen_panel.update_stats(self.llm.last_stats)
        self.gen_panel.update_history(self.llm.history)

    def _on_continue(self, params: dict):
        self._stop_playback()
        custom = self.gen_panel.sys_prompt_edit.toPlainText().strip()
        self.llm._custom_system_prompt = custom if custom else ""
        if "template" in params:
            self.llm.template_name = params["template"]
        self.statusBar().showMessage("Extending MIDI...")
        self.gen_panel.set_generating(True)
        self.gen_panel.update_adjust_output("")
        QApplication.processEvents()

        def _stream_cb(text):
            QTimer.singleShot(0, lambda: self.gen_panel.update_adjust_output(text))

        def _cont():
            new_notes = self.llm.generate_continue(
                existing_notes=self.piano_roll.notes,
                prompt=params["prompt"], genre=params["genre"],
                key=params["key"], scale=params["scale"],
                tempo=self.tempo, num_bars=params["num_bars"],
                temperature=params["temperature"],
                stream_cb=_stream_cb)
            QTimer.singleShot(0, lambda: self._on_continue_done(new_notes))

        threading.Thread(target=_cont, daemon=True).start()

    def _on_continue_done(self, new_notes):
        self.gen_panel.set_generating(False)
        if new_notes:
            self.piano_roll.notes.extend(new_notes)
            self.piano_roll.notesChanged.emit()
            self.piano_roll.update()
            self.piano_roll.total_ticks = max(
                (n.tick + n.duration for n in self.piano_roll.notes), default=TICKS_PER_BEAT * 16)
            self.statusBar().showMessage(f"Extended with {len(new_notes)} notes")
        else:
            self.statusBar().showMessage("Extend produced no notes")
        if self.llm.last_stats:
            self.gen_panel.update_stats(self.llm.last_stats)
        self.gen_panel.update_history(self.llm.history)

    def _on_chat(self, message: str):
        self.gen_panel.append_chat_message("user", message)
        self.gen_panel.chat_display.append(
            '<span style="color:#c080ff;font-weight:bold;">AI:</span> '
            '<span style="color:#666;" id="thinking">thinking</span>')
        sb = self.gen_panel.chat_display.verticalScrollBar()
        sb.setValue(sb.maximum())
        self.gen_panel.btn_chat_send.setEnabled(False)
        # Animated thinking dots
        self._thinking_dots = 0
        self._thinking_timer = QTimer()
        self._thinking_timer.setInterval(500)
        self._thinking_timer.timeout.connect(self._animate_thinking)
        self._thinking_timer.start()

        # Build context from current project notes
        context = ""
        if self.piano_roll.notes:
            a = analyze_notes(self.piano_roll.notes)
            context = (
                f"- Project has {a['note_count']} notes, {a['total_bars']} bars\n"
                f"- Range: {a['pitch_range']}, Avg velocity: {a['avg_velocity']}\n"
                f"- Density: {a['density_profile']}, Genre feel: {a.get('time_sig_feel', '4/4')}\n"
            )
            if a.get("detected_chords"):
                context += f"- Detected chords: {', '.join(a['detected_chords'][:5])}\n"

        def _stream_cb(text):
            QTimer.singleShot(0, lambda: self._update_chat_stream(text))

        def _chat():
            reply = self.llm.chat(message, stream_cb=_stream_cb, context=context)
            QTimer.singleShot(0, lambda: self._on_chat_done(reply))

        threading.Thread(target=_chat, daemon=True).start()

    def _update_chat_stream(self, text: str):
        # Show streaming progress - only update the last AI block
        cursor = self.gen_panel.chat_display.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        # Check if last block is an AI block we can update
        doc = self.gen_panel.chat_display.document()
        if doc.blockCount() > 0:
            last_block = doc.findBlockByNumber(doc.blockCount() - 1)
            last_text = last_block.text()
            if last_text.startswith("AI: "):
                # Replace the last AI line with updated text
                cursor.select(cursor.SelectionType.BlockUnderCursor)
                # Escape HTML
                safe_text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                cursor.insertHtml(
                    '<span style="color:#c080ff;font-weight:bold;">AI:</span> '
                    f'<span style="color:#c8c8d8;">{safe_text}</span>')
                sb = self.gen_panel.chat_display.verticalScrollBar()
                sb.setValue(sb.maximum())
                return
        # Fallback: append new block (shouldn't happen during streaming)
        safe_text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        self.gen_panel.chat_display.append(
            '<span style="color:#c080ff;font-weight:bold;">AI:</span> '
            f'<span style="color:#c8c8d8;">{safe_text}</span>')
        sb = self.gen_panel.chat_display.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _animate_thinking(self):
        self._thinking_dots = (self._thinking_dots + 1) % 4
        dots = "." * self._thinking_dots
        cursor = self.gen_panel.chat_display.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        doc = self.gen_panel.chat_display.document()
        if doc.blockCount() > 0:
            last_block = doc.findBlockByNumber(doc.blockCount() - 1)
            last_text = last_block.text()
            if "thinking" in last_text.lower() or dots:
                cursor.select(cursor.SelectionType.BlockUnderCursor)
                cursor.insertHtml(
                    '<span style="color:#c080ff;font-weight:bold;">AI:</span> '
                    f'<span style="color:#666;">thinking{dots}</span>')
                sb = self.gen_panel.chat_display.verticalScrollBar()
                sb.setValue(sb.maximum())

    def _on_chat_done(self, reply: str):
        if hasattr(self, '_thinking_timer') and self._thinking_timer.isActive():
            self._thinking_timer.stop()
        self.gen_panel.btn_chat_send.setEnabled(True)
        if self.llm.last_stats:
            self.gen_panel.update_stats(self.llm.last_stats)

    def _on_history_recall(self, index: int):
        if index == -1:
            entry = self.llm.history.latest()
        else:
            entry = self.llm.history[index] if index < len(self.llm.history) else None
        if entry:
            notes, stats = entry
            import copy
            self.piano_roll.set_notes(copy.deepcopy(notes))
            self.statusBar().showMessage(f"Recalled generation: {stats.summary}")

    def _on_variation(self, params: dict):
        self._stop_playback()
        self.llm._current_notes = self.piano_roll.notes
        self.gen_panel.set_generating(True)
        self.gen_panel.update_adjust_output("")
        self.statusBar().showMessage("Generating variation...")

        def _stream_cb(text):
            QTimer.singleShot(0, lambda: self.gen_panel.update_adjust_output(text))

        def _gen():
            notes = self.llm.generate_variation(
                genre=self.gen_panel.genre_combo.currentText(),
                key=self.gen_panel.key_combo.currentText(),
                scale=self.gen_panel.scale_combo.currentText(),
                tempo=self.gen_panel.tempo_spin.value(),
                num_bars=self.gen_panel.bars_spin.value(),
                temperature=self.gen_panel.temp_slider.value() / 100.0,
                stream_cb=_stream_cb)
            QTimer.singleShot(0, lambda: self._on_gen_done(notes, {
                "genre": self.gen_panel.genre_combo.currentText(),
                "key": self.gen_panel.key_combo.currentText(),
                "scale": self.gen_panel.scale_combo.currentText(),
            }))

        threading.Thread(target=_gen, daemon=True).start()

    def _on_arrange(self):
        melody = self.piano_roll.get_selected_notes()
        if not melody:
            melody = self.piano_roll.notes
        if not melody:
            self.statusBar().showMessage("No notes to arrange")
            return
        self.piano_roll.undo.save_state(self.piano_roll.notes)
        self._stop_playback()
        self.llm._current_notes = self.piano_roll.notes
        self.gen_panel.set_generating(True)
        self.gen_panel.update_adjust_output("")
        self.gen_panel.btn_arrange.setEnabled(False)
        self.statusBar().showMessage("Auto-arranging: generating bass + chords...")

        def _stream_cb(text):
            QTimer.singleShot(0, lambda: self.gen_panel.update_adjust_output(text))

        def _arr():
            accompaniment = self.llm.generate_arrangement(
                melody_notes=melody,
                genre=self.gen_panel.genre_combo.currentText(),
                key=self.gen_panel.key_combo.currentText(),
                scale=self.gen_panel.scale_combo.currentText(),
                tempo=self.gen_panel.tempo_spin.value(),
                temperature=self.gen_panel.temp_slider.value() / 100.0 * 0.9,
                stream_cb=_stream_cb)
            QTimer.singleShot(0, lambda: self._on_arrange_done(accompaniment))

        threading.Thread(target=_arr, daemon=True).start()

    def _on_arrange_done(self, accompaniment):
        self.gen_panel.set_generating(False)
        self.gen_panel.btn_arrange.setEnabled(True)
        if accompaniment:
            self.piano_roll.notes.extend(accompaniment)
            self.piano_roll.notesChanged.emit()
            self.piano_roll._update_total_ticks()
            self.piano_roll.update()
            self.statusBar().showMessage(f"Added {len(accompaniment)} accompaniment notes (bass + chords)")
        else:
            self.statusBar().showMessage("Auto-arrange produced no notes")
        if self.llm.last_stats:
            self.gen_panel.update_stats(self.llm.last_stats)
        self.gen_panel.update_history(self.llm.history)

    def _on_chat_generate(self, prompt_text: str):
        if not prompt_text.strip():
            return
        self.gen_panel.prompt_edit.setPlainText(prompt_text[:500])
        self.gen_panel.tabs.setCurrentIndex(0)
        self._on_generate({
            "prompt": prompt_text,
            "genre": self.gen_panel.genre_combo.currentText(),
            "key": self.gen_panel.key_combo.currentText(),
            "scale": self.gen_panel.scale_combo.currentText(),
            "tempo": self.gen_panel.tempo_spin.value(),
            "num_bars": self.gen_panel.bars_spin.value(),
            "temperature": self.gen_panel.temp_slider.value() / 100.0,
            "template": self.gen_panel.template_combo.currentText(),
            "top_k": self.gen_panel.topk_slider.value(),
            "top_p": self.gen_panel.topp_slider.value() / 100.0,
            "repeat_penalty": self.gen_panel.rep_penalty_slider.value() / 100.0,
            "max_tokens": self.gen_panel.max_tok_spin.value(),
            "frequency_penalty": self.gen_panel.freq_pen_slider.value() / 100.0,
            "presence_penalty": self.gen_panel.pres_pen_slider.value() / 100.0,
            "min_p": self.gen_panel.min_p_slider.value() / 100.0,
        })

    def _on_panel_tempo_changed(self, tempo: int):
        """Sync tempo from generation panel to transport and internal state."""
        self.tempo = tempo
        self.transport.tempo_spin.setValue(tempo)

    def _on_melody_from_chords(self):
        notes = self.piano_roll.get_selected_notes()
        if not notes:
            notes = self.piano_roll.notes
        if not notes:
            self.statusBar().showMessage("No notes to generate melody from")
            return
        self.piano_roll.undo.save_state(self.piano_roll.notes)
        self._stop_playback()
        self.gen_panel.set_generating(True)
        self.gen_panel.update_adjust_output("")
        self.statusBar().showMessage("Generating melody from chords...")

        def _stream_cb(text):
            QTimer.singleShot(0, lambda: self.gen_panel.update_adjust_output(text))

        def _gen():
            melody = self.llm.generate_melody_from_chords(
                chord_notes=notes,
                genre=self.gen_panel.genre_combo.currentText(),
                key=self.gen_panel.key_combo.currentText(),
                scale=self.gen_panel.scale_combo.currentText(),
                tempo=self.gen_panel.tempo_spin.value(),
                temperature=self.gen_panel.temp_slider.value() / 100.0,
                stream_cb=_stream_cb)
            QTimer.singleShot(0, lambda: self._on_melody_from_chords_done(melody))

        threading.Thread(target=_gen, daemon=True).start()

    def _on_melody_from_chords_done(self, melody):
        self.gen_panel.set_generating(False)
        if melody:
            self.piano_roll.notes.extend(melody)
            self.piano_roll.notesChanged.emit()
            self.piano_roll._update_total_ticks()
            self.piano_roll.update()
            self.statusBar().showMessage(f"Added {len(melody)} melody notes from chords")
        else:
            self.statusBar().showMessage("Melody generation produced no notes")
        if self.llm.last_stats:
            self.gen_panel.update_stats(self.llm.last_stats)
        self.gen_panel.update_history(self.llm.history)

    def _on_full_arrangement(self):
        notes = self.piano_roll.get_selected_notes()
        if not notes:
            notes = self.piano_roll.notes
        if not notes:
            self.statusBar().showMessage("No notes to arrange")
            return
        self.piano_roll.undo.save_state(self.piano_roll.notes)
        self._stop_playback()
        self.gen_panel.set_generating(True)
        self.gen_panel.update_adjust_output("")
        self.statusBar().showMessage("Generating full arrangement...")

        def _stream_cb(text):
            QTimer.singleShot(0, lambda: self.gen_panel.update_adjust_output(text))

        def _gen():
            arranged = self.llm.generate_full_arrangement(
                seed_notes=notes,
                genre=self.gen_panel.genre_combo.currentText(),
                key=self.gen_panel.key_combo.currentText(),
                scale=self.gen_panel.scale_combo.currentText(),
                tempo=self.gen_panel.tempo_spin.value(),
                temperature=self.gen_panel.temp_slider.value() / 100.0 * 0.9,
                stream_cb=_stream_cb)
            QTimer.singleShot(0, lambda: self._on_full_arrange_done(arranged))

        threading.Thread(target=_gen, daemon=True).start()

    def _on_full_arrange_done(self, arranged):
        self.gen_panel.set_generating(False)
        if arranged:
            self.piano_roll.set_notes(arranged)
            self.piano_roll._update_total_ticks()
            self.piano_roll.update()
            self.statusBar().showMessage(f"Full arrangement: {len(arranged)} notes")
        else:
            self.statusBar().showMessage("Full arrangement produced no notes")
        if self.llm.last_stats:
            self.gen_panel.update_stats(self.llm.last_stats)
        self.gen_panel.update_history(self.llm.history)

    def _on_abort(self):
        self.llm.abort()
        self.gen_panel.set_generating(False)
        self.gen_panel.gen_progress.setValue(0)
        self.gen_elapsed_label.setText("")
        self.statusBar().showMessage("Generation aborted")

    def _on_ai_adjust_with_mode(self, instruction: str, mode: str):
        """AI adjust with explicit mode selection."""
        selected = self.piano_roll.get_selected_notes()
        if not instruction.strip():
            self.statusBar().showMessage("Enter an instruction"); return
        if not selected:
            self.statusBar().showMessage("Select notes first"); return
        # Save state for undo before AI modification
        self.piano_roll.undo.save_state(self.piano_roll.notes)
        self.gen_panel.btn_adjust.setEnabled(False)
        self.gen_panel.update_adjust_output("")
        mode_label = ADJUST_MODES.get(mode, {}).get("label", mode)
        self.statusBar().showMessage(f"AI adjusting ({mode_label})...")

        self.llm._current_notes = self.piano_roll.notes

        def _stream_cb(text):
            QTimer.singleShot(0, lambda: self.gen_panel.update_adjust_output(text))

        def _adj():
            adjusted = self.llm.ai_adjust(selected, instruction, mode=mode, stream_cb=_stream_cb)
            QTimer.singleShot(0, lambda: self._on_adj_done(adjusted))

        threading.Thread(target=_adj, daemon=True).start()

    def _on_ai_adjust(self, instruction: str):
        selected = self.piano_roll.get_selected_notes()
        if not instruction.strip():
            self.statusBar().showMessage("Enter an instruction"); return
        if not selected:
            self.statusBar().showMessage("Select notes first"); return
        # Save state for undo before AI modification
        self.piano_roll.undo.save_state(self.piano_roll.notes)
        self.gen_panel.btn_adjust.setEnabled(False)
        self.gen_panel.update_adjust_output("")
        self.statusBar().showMessage("AI adjusting...")

        self.llm._current_notes = self.piano_roll.notes

        def _stream_cb(text):
            QTimer.singleShot(0, lambda: self.gen_panel.update_adjust_output(text))

        def _adj():
            adjusted = self.llm.ai_adjust(selected, instruction, stream_cb=_stream_cb)
            QTimer.singleShot(0, lambda: self._on_adj_done(adjusted))

        threading.Thread(target=_adj, daemon=True).start()

    def _on_analyze_selection(self):
        selected = self.piano_roll.get_selected_notes()
        if not selected:
            self.gen_panel.update_analysis(self.piano_roll.notes)
            self.statusBar().showMessage("Analyzed all notes")
        else:
            self.gen_panel.update_analysis(selected)
            self.statusBar().showMessage(f"Analyzed {len(selected)} selected notes")

    def _on_adj_done(self, adjusted):
        self.gen_panel.btn_adjust.setEnabled(True)
        sel_ids = {n.id for n in self.piano_roll.get_selected_notes()}
        new_notes = [n for n in self.piano_roll.notes if n.id not in sel_ids]
        new_notes.extend(adjusted)
        self.piano_roll.set_notes(new_notes)
        self.piano_roll.selected_ids = {n.id for n in adjusted}
        self.statusBar().showMessage(f"Adjusted {len(adjusted)} notes")
        if self.llm.last_stats:
            self.gen_panel.update_stats(self.llm.last_stats)

    # ── Playback (pre-rendered) ───────────────────────────────────────────

    def _toggle_playback(self, play: bool):
        if play:
            self._start_playback()
        else:
            self._stop_playback()

    def _start_playback(self):
        self._is_playing = True
        self.transport.set_playing(True)
        self._play_start_tick = self.piano_roll.playback_tick

        if SD_AVAILABLE and self.piano_roll.notes:
            self.statusBar().showMessage("Rendering audio...")
            QApplication.processEvents()

            def _render():
                audio = self.synth.render_audio(
                    self.piano_roll.notes, self.tempo, self.piano_roll.total_ticks)
                QTimer.singleShot(0, lambda: self._on_render_done(audio))

            threading.Thread(target=_render, daemon=True).start()
        else:
            self._play_start_time = time.time()
            self._last_beat = -1
            self._play_timer.start()
            if not SD_AVAILABLE:
                self.statusBar().showMessage("Playing (visual only — install sounddevice for audio)")

    def _on_render_done(self, audio):
        self.synth.play_buffer(audio)
        self._play_start_time = time.time()
        self._last_beat = -1
        self._play_timer.start()
        self.statusBar().showMessage("▶ Playing")

    def _stop_playback(self):
        self._is_playing = False
        self.transport.set_playing(False)
        self._play_timer.stop()
        self.synth.stop()
        self.piano_roll.playback_tick = 0
        self.piano_roll._active_note_ids.clear()
        self.transport.update_position(0, self.piano_roll.total_ticks)
        self.piano_roll.update()
        self.statusBar().showMessage("Stopped")

    def _on_playback_tick(self):
        if not self._is_playing: return
        tpb = TICKS_PER_BEAT
        ticks_per_sec = (self.tempo / 60.0) * tpb

        elapsed = time.time() - self._play_start_time
        current_tick = self._play_start_tick + int(elapsed * ticks_per_sec)

        # Beat pulse
        current_beat = current_tick // tpb
        if current_beat != self._last_beat:
            self._last_beat = current_beat
            self.piano_roll.trigger_beat_pulse()

        loop = self.transport.btn_loop.isChecked()
        if current_tick >= self.piano_roll.total_ticks:
            if loop:
                current_tick %= self.piano_roll.total_ticks
                self._play_start_tick = current_tick
                self._play_start_time = time.time()
                # Re-render for loop
                if SD_AVAILABLE and self.piano_roll.notes:
                    audio = self.synth.render_audio(
                        self.piano_roll.notes, self.tempo, self.piano_roll.total_ticks)
                    self.synth.play_buffer(audio)
            else:
                self._stop_playback()
                return

        self.piano_roll.playback_tick = current_tick
        self.piano_roll.update_active_notes(current_tick)
        self.transport.update_position(current_tick, self.piano_roll.total_ticks)
        self.piano_roll.scroll_to_tick(current_tick)

    def _seek_to_tick(self, tick: int):
        tick = max(0, min(tick, self.piano_roll.total_ticks))
        was_playing = self._is_playing
        if was_playing:
            self._stop_playback()
        self.piano_roll.playback_tick = tick
        self.transport.update_position(tick, self.piano_roll.total_ticks)
        self.piano_roll.update()
        if was_playing:
            self._start_playback()

    def _on_tempo_changed(self, tempo: int):
        self.tempo = tempo

    # ── Edit ───────────────────────────────────────────────────────────────

    def _on_selection_changed(self, selected_ids):
        self.gen_panel.update_properties(
            [n for n in self.piano_roll.notes if n.id in selected_ids])
        # Auto-update analysis for selected notes
        selected = self.piano_roll.get_selected_notes()
        if selected:
            self.gen_panel.update_analysis(selected)

    def _on_notes_changed(self):
        self.statusBar().showMessage(f"{len(self.piano_roll.notes)} notes")

    def _select_all(self):
        self.piano_roll.selected_ids = {n.id for n in self.piano_roll.notes}
        self.piano_roll.noteSelected.emit(list(self.piano_roll.selected_ids))
        self.piano_roll.update()

    def _duplicate_selected(self):
        notes = self.piano_roll.get_selected_notes()
        if not notes: return
        new = [MidiNote(tick=n.tick + TICKS_PER_BEAT * 4, pitch=n.pitch,
                        velocity=n.velocity, duration=n.duration) for n in notes]
        self.piano_roll.notes.extend(new)
        self.piano_roll.selected_ids = {n.id for n in new}
        self.piano_roll.notesChanged.emit()
        self.piano_roll.noteSelected.emit(list(self.piano_roll.selected_ids))
        self.piano_roll.update()

    def _quantize_selected(self):
        snap = TICKS_PER_BEAT // 4
        for n in self.piano_roll.notes:
            if n.id in self.piano_roll.selected_ids:
                n.tick = round(n.tick / snap) * snap
        self.piano_roll.notesChanged.emit()
        self.piano_roll.update()
        self.statusBar().showMessage("Quantized")

    def _apply_note_properties(self):
        selected = self.piano_roll.get_selected_notes()
        if not selected: return
        vel = self.gen_panel.prop_velocity.value()
        for n in selected:
            if len(selected) == 1:
                n.pitch = self.gen_panel.prop_pitch.value()
                n.tick = self.gen_panel.prop_start.value()
                n.duration = max(1, self.gen_panel.prop_duration.value())
            n.velocity = max(1, min(127, vel))
        self.piano_roll.notesChanged.emit()
        self.piano_roll.update()

    def _preview_note(self, pitch: int):
        if not SD_AVAILABLE: return
        def _play():
            try:
                sr = 44100; dur = 0.25
                freq = 440.0 * (2.0 ** ((pitch - 69) / 12.0))
                n_samp = int(sr * dur)
                t = np.arange(n_samp, dtype=np.float64) / sr
                phase = 2.0 * np.pi * freq * t
                audio = np.sin(phase) * 1.0
                audio += np.sin(2 * phase) * 0.4
                audio += np.sin(3 * phase) * 0.15
                audio *= np.exp(-t * 5)
                audio *= 0.2 * self.synth.volume
                env = np.ones(n_samp, dtype=np.float64)
                a = int(0.005 * sr); r = int(0.08 * sr)
                if a > 0: env[:a] = np.linspace(0, 1, a)
                if r > 0 and r < n_samp: env[-r:] = np.linspace(1, 0, r)
                sd.play((audio * env).astype(np.float32), samplerate=sr)
            except Exception: pass
        threading.Thread(target=_play, daemon=True).start()

    # ── View ───────────────────────────────────────────────────────────────

    def _zoom(self, factor: float):
        self.piano_roll.ppq = max(PianoRollWidget.MIN_PPQ,
                                   min(PianoRollWidget.MAX_PPQ, self.piano_roll.ppq * factor))
        self.piano_roll.update()

    def _fit_content(self):
        notes = self.piano_roll.notes
        if not notes: return
        min_p = min(n.pitch for n in notes); max_p = max(n.pitch for n in notes)
        min_t = min(n.tick for n in notes); max_t = max(n.tick + n.duration for n in notes)
        aw = self.piano_roll.width() - PianoRollWidget.PIANO_WIDTH
        ah = self.piano_roll._note_area_height()
        pr = max(max_p - min_p + 2, 12); tr = max(max_t - min_t, TICKS_PER_BEAT * 4)
        self.piano_roll.note_height = max(PianoRollWidget.MIN_NOTE_HEIGHT,
                                           min(PianoRollWidget.MAX_NOTE_HEIGHT, ah / pr))
        self.piano_roll.ppq = max(PianoRollWidget.MIN_PPQ,
                                   min(PianoRollWidget.MAX_PPQ, aw / tr))
        self.piano_roll.scroll_x = max(0, min_t * self.piano_roll.ppq - 20)
        self.piano_roll.scroll_y = (127 - max_p - 1) * self.piano_roll.note_height
        self.piano_roll.update()

    # ── WAV Export ──────────────────────────────────────────────
    def _export_wav(self):
        if not SD_AVAILABLE:
            QMessageBox.warning(self, "WAV Export", "sounddevice is not installed.\npip install sounddevice")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export WAV", "output.wav", "WAV (*.wav);;All (*)")
        if not path: return
        self.statusBar().showMessage("Rendering WAV...")
        QApplication.processEvents()

        def _render_wav():
            try:
                audio = self.synth.render_audio(
                    self.piano_roll.notes, self.tempo, self.piano_roll.total_ticks)
                import sounddevice as sd
                import struct as _struct
                sr = self.synth.sample_rate
                # Write 16-bit PCM WAV
                pcm = (audio * 32767).astype(np.int16)
                with open(path, 'wb') as f:
                    # RIFF header
                    data_size = len(pcm) * 2
                    f.write(b'RIFF')
                    f.write(_struct.pack('<I', 36 + data_size))
                    f.write(b'WAVE')
                    # fmt chunk
                    f.write(b'fmt ')
                    f.write(_struct.pack('<I', 16))
                    f.write(_struct.pack('<H', 1))  # PCM
                    f.write(_struct.pack('<H', 1))  # mono
                    f.write(_struct.pack('<I', sr))
                    f.write(_struct.pack('<I', sr * 2))  # byte rate
                    f.write(_struct.pack('<H', 2))  # block align
                    f.write(_struct.pack('<H', 16))  # bits per sample
                    # data chunk
                    f.write(b'data')
                    f.write(_struct.pack('<I', data_size))
                    f.write(pcm.tobytes())
                QTimer.singleShot(0, lambda: self.statusBar().showMessage(f"Exported WAV: {path}"))
            except Exception as e:
                QTimer.singleShot(0, lambda: QMessageBox.critical(self, "WAV Error", str(e)))

        threading.Thread(target=_render_wav, daemon=True).start()

    # ── Transpose dialog ───────────────────────────────────────
    def _transpose_dialog(self):
        self.piano_roll._transpose_dialog()

    # ── Help dialogs ────────────────────────────────────────────
    def _show_shortcuts(self):
        shortcuts = [
            ("Space", "Play / Stop"),
            ("1 / 2 / 3", "Select / Draw / Erase tool"),
            ("Ctrl+Z", "Undo"),
            ("Ctrl+Shift+Z", "Redo"),
            ("Ctrl+A", "Select all notes"),
            ("Ctrl+C", "Copy selected notes"),
            ("Ctrl+X", "Cut selected notes"),
            ("Ctrl+V", "Paste notes"),
            ("Ctrl+D", "Duplicate selected"),
            ("Ctrl+T", "Transpose dialog"),
            ("Ctrl+Shift+Up/Down", "Transpose +/-1"),
            ("Delete", "Delete selected"),
            ("Ctrl+N", "New project"),
            ("Ctrl+O", "Import MIDI"),
            ("Ctrl+S", "Export MIDI"),
            ("Ctrl+Shift+S", "Export WAV"),
            ("Ctrl+=", "Zoom in"),
            ("Ctrl+-", "Zoom out"),
            ("Shift+Scroll", "Horizontal scroll"),
            ("Middle-click drag", "Pan view"),
            ("Home", "Go to start"),
        ]
        from PySide6.QtWidgets import QDialog, QTableWidget, QTableWidgetItem, QVBoxLayout as _VL
        dlg = QDialog(self)
        dlg.setWindowTitle("Keyboard Shortcuts")
        dlg.resize(420, 480)
        dlg.setStyleSheet("QDialog{background:#14142a;}QTableWidget{background:#1a1a30;"
                          "color:#c8c8d8;gridline-color:#2a2a44;border:1px solid #2a2a44;}"
                          "QHeaderView::section{background:#1e1e32;color:#a0a0c8;border:1px solid #2a2a44;"
                          "padding:4px;}")
        lay = _VL(dlg)
        table = QTableWidget(len(shortcuts), 2)
        table.setHorizontalHeaderLabels(["Key", "Action"])
        table.horizontalHeader().setStretchLastSection(True)
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        table.setSelectionMode(QTableWidget.NoSelection)
        for i, (key, action) in enumerate(shortcuts):
            table.setItem(i, 0, QTableWidgetItem(key))
            table.setItem(i, 1, QTableWidgetItem(action))
        lay.addWidget(table)
        dlg.exec()

    def _show_about(self):
        QMessageBox.about(self, "About MIDI LLM Studio",
            "<h3>MIDI LLM Studio</h3>"
            "<p>A PySide6 piano roll editor with local LLM-powered MIDI generation.</p><p>Features: multi-template prompts, streaming output, AI chat, layer generation, generation history, and genre-aware system prompts.</p>"
            "<p>Generate entire songs, loop packs, or sample MIDI packs using local GGUF models "
            "via llama.cpp, or use the built-in demo mode for instant patterns.</p>"
            "<p>Edit notes with full undo/redo, transpose, quantize, copy/paste, velocity editing, "
            "and export to MIDI or WAV.</p>")

    def keyPressEvent(self, event):
        key = event.key()
        mod = event.modifiers()
        if key == Qt.Key_Space and not (mod & Qt.ControlModifier):
            self._toggle_playback(not self._is_playing)
        elif key == Qt.Key_1: self.toolbar.set_tool("select")
        elif key == Qt.Key_2: self.toolbar.set_tool("draw")
        elif key == Qt.Key_3: self.toolbar.set_tool("erase")
        elif key == Qt.Key_Home: self._seek_to_tick(0)
        elif key == Qt.Key_End: self._seek_to_tick(self.piano_roll.total_ticks)
        else: super().keyPressEvent(event)

    def closeEvent(self, event):
        self._stop_playback()
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    pal = app.palette()
    pal.setColor(pal.ColorRole.Window, QColor(10, 10, 20))
    pal.setColor(pal.ColorRole.WindowText, QColor(200, 200, 220))
    pal.setColor(pal.ColorRole.Base, QColor(18, 18, 30))
    pal.setColor(pal.ColorRole.AlternateBase, QColor(26, 26, 42))
    pal.setColor(pal.ColorRole.Text, QColor(200, 200, 220))
    pal.setColor(pal.ColorRole.Button, QColor(26, 26, 42))
    pal.setColor(pal.ColorRole.ButtonText, QColor(200, 200, 220))
    pal.setColor(pal.ColorRole.Highlight, QColor(124, 92, 252))
    pal.setColor(pal.ColorRole.HighlightedText, QColor(255, 255, 255))
    app.setPalette(pal)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()