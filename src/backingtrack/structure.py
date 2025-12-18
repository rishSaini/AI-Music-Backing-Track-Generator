# src/backingtrack/structure.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional

HatType = Literal["closed", "open", "ride"]


@dataclass(frozen=True)
class BarStyle:
    """
    Per-bar style controls used by arrange.py.
    """
    density: float                # 0..1 (how busy)
    brightness: float             # -1..1 (pad register tilt)
    drums_enabled: bool
    hat: HatType
    fill_every: int               # bars; 4 = more fills, 8 = fewer
    intensity: float              # 0..1 (velocity/energy shaping)


@dataclass(frozen=True)
class Section:
    name: str
    start_bar: int   # inclusive
    end_bar: int     # exclusive
    base_style: BarStyle

    def contains(self, bar: int) -> bool:
        return self.start_bar <= bar < self.end_bar


@dataclass(frozen=True)
class SongStructure:
    sections: List[Section]

    def style_for_bar(self, bar: int) -> BarStyle:
        for s in self.sections:
            if s.contains(bar):
                # Optional: subtle ramp within section (build)
                length = max(1, s.end_bar - s.start_bar)
                pos = (bar - s.start_bar) / float(length)  # 0..1
                # build a tiny bit toward the end of each section
                density = _clamp01(s.base_style.density + 0.08 * pos)
                intensity = _clamp01(s.base_style.intensity + 0.10 * pos)
                return BarStyle(
                    density=density,
                    brightness=s.base_style.brightness,
                    drums_enabled=s.base_style.drums_enabled,
                    hat=s.base_style.hat,
                    fill_every=s.base_style.fill_every,
                    intensity=intensity,
                )
        # fallback
        return BarStyle(density=0.5, brightness=0.0, drums_enabled=True, hat="closed", fill_every=8, intensity=0.6)

    def describe(self) -> str:
        lines = []
        for s in self.sections:
            lines.append(f"{s.name}: bars {s.start_bar}..{s.end_bar}  "
                         f"(dens={s.base_style.density:.2f}, hat={s.base_style.hat}, fills={s.base_style.fill_every})")
        return "\n".join(lines)


def auto_structure(num_bars: int, base_density: float, base_brightness: float) -> SongStructure:
    """
    Create a reasonable structure based on total bars.
    """
    num_bars = max(1, int(num_bars))

    def st(dens, bright, drums, hat, fill, inten) -> BarStyle:
        return BarStyle(
            density=_clamp01(dens),
            brightness=_clamp11(bright),
            drums_enabled=drums,
            hat=hat,
            fill_every=max(1, int(fill)),
            intensity=_clamp01(inten),
        )

    sections: List[Section] = []

    # Small songs: just one evolving section
    if num_bars <= 8:
        sections.append(Section("Main", 0, num_bars, st(base_density, base_brightness, True, "closed", 8, 0.6)))
        return SongStructure(sections)

    # Decide intro/outro sizes
    intro = 4 if num_bars >= 24 else 2
    outro = 4 if num_bars >= 24 else 2

    cursor = 0

    # Intro (often no drums at first)
    if intro > 0:
        # first half: no drums, second half: light drums
        half = intro // 2
        if half > 0:
            sections.append(Section("Intro (no drums)", cursor, cursor + half,
                                    st(base_density * 0.35, base_brightness - 0.2, False, "closed", 16, 0.35)))
            cursor += half
        sections.append(Section("Intro", cursor, cursor + (intro - (intro // 2)),
                                st(base_density * 0.45, base_brightness - 0.15, True, "closed", 16, 0.40)))
        cursor += (intro - (intro // 2))

    remaining = num_bars - cursor - outro
    remaining = max(0, remaining)

    # Build verse/chorus blocks of 8 bars
    block = 8
    names = ["Verse", "Chorus"]
    idx = 0

    while remaining > 0:
        length = block if remaining >= block else remaining
        name = names[idx % 2]

        if name == "Verse":
            sections.append(Section(
                "Verse",
                cursor,
                cursor + length,
                st(base_density * 0.75, base_brightness + 0.0, True, "closed", 8, 0.55),
            ))
        else:
            sections.append(Section(
                "Chorus",
                cursor,
                cursor + length,
                st(base_density * 1.05, base_brightness + 0.25, True, "ride", 4, 0.80),
            ))

        cursor += length
        remaining -= length
        idx += 1

        # Optional: add a Bridge if we have room and already did at least one chorus
        # (keeps it from being verse/chorus forever)
        if remaining >= 8 and idx >= 4:
            sections.append(Section(
                "Bridge",
                cursor,
                cursor + 8,
                st(base_density * 0.90, base_brightness + 0.10, True, "ride", 4, 0.70),
            ))
            cursor += 8
            remaining -= 8

    # Outro
    if outro > 0 and cursor < num_bars:
        end = num_bars
        sections.append(Section(
            "Outro",
            cursor,
            end,
            st(base_density * 0.55, base_brightness - 0.10, True, "closed", 16, 0.45),
        ))

    return SongStructure(sections)


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else float(x)


def _clamp11(x: float) -> float:
    return -1.0 if x < -1.0 else 1.0 if x > 1.0 else float(x)
