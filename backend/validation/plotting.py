"""Small SVG plotting helpers to keep validation scripts dependency-light."""

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Mapping, Sequence


def _scale(value: float, lo: float, hi: float, out_lo: float, out_hi: float) -> float:
    if abs(hi - lo) < 1e-12:
        return 0.5 * (out_lo + out_hi)
    return out_lo + (value - lo) * (out_hi - out_lo) / (hi - lo)


def write_line_chart(
    series: Mapping[str, Sequence[tuple[float, float]]],
    path: Path,
    *,
    title: str,
    x_label: str,
    y_label: str,
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 900, 520
    left, right, top, bottom = 82, 36, 54, 74
    colors = ("#2f6f73", "#b35f2e", "#5954a4", "#7b6b3a")
    points = [point for values in series.values() for point in values]
    x_values = [p[0] for p in points]
    y_values = [p[1] for p in points]
    x_lo, x_hi = min(x_values), max(x_values)
    y_lo, y_hi = min(0.0, min(y_values)), max(y_values)
    y_hi = y_hi * 1.12 if y_hi else 1.0
    lines = []
    legend = []
    for idx, (label, values) in enumerate(series.items()):
        color = colors[idx % len(colors)]
        coords = [
            (
                _scale(x, x_lo, x_hi, left, width - right),
                _scale(y, y_lo, y_hi, height - bottom, top),
            )
            for x, y in values
        ]
        path_data = " ".join(f"{x:.1f},{y:.1f}" for x, y in coords)
        lines.append(f'<polyline fill="none" stroke="{color}" stroke-width="3" points="{path_data}" />')
        for x, y in coords:
            lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="{color}" />')
        legend.append(
            f'<g transform="translate({left + idx * 190},28)"><rect width="18" height="4" y="8" fill="{color}" />'
            f'<text x="26" y="14" font-size="14" fill="#243033">{escape(label)}</text></g>'
        )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#f7f7f2"/>
<text x="{left}" y="24" font-size="20" font-family="Arial" font-weight="700" fill="#1f2a2d">{escape(title)}</text>
{''.join(legend)}
<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#445" stroke-width="1.5"/>
<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#445" stroke-width="1.5"/>
<text x="{width/2}" y="{height-24}" font-size="15" font-family="Arial" text-anchor="middle" fill="#243033">{escape(x_label)}</text>
<text x="24" y="{height/2}" font-size="15" font-family="Arial" text-anchor="middle" transform="rotate(-90 24 {height/2})" fill="#243033">{escape(y_label)}</text>
{''.join(lines)}
</svg>"""
    path.write_text(svg, encoding="utf-8")
    return str(path)


def write_bar_chart(
    rows: Sequence[Mapping[str, float | str]],
    path: Path,
    *,
    title: str,
    category_key: str,
    value_key: str,
    y_label: str,
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 980, 520
    left, right, top, bottom = 88, 42, 60, 98
    max_value = max(float(row[value_key]) for row in rows) if rows else 1.0
    max_value = max(max_value * 1.15, 1e-9)
    bar_width = (width - left - right) / max(1, len(rows))
    palette = ("#2f6f73", "#b35f2e", "#5954a4", "#8a7b46", "#426b9f", "#9b4f5e")
    bars = []
    for idx, row in enumerate(rows):
        value = float(row[value_key])
        x = left + idx * bar_width + bar_width * 0.15
        w = bar_width * 0.70
        y = _scale(value, 0.0, max_value, height - bottom, top)
        h = height - bottom - y
        color = palette[idx % len(palette)]
        label = str(row[category_key])
        bars.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="3" fill="{color}" />')
        bars.append(
            f'<text x="{x + w/2:.1f}" y="{height-bottom+24}" font-size="13" font-family="Arial" text-anchor="middle" fill="#243033">{escape(label)}</text>'
        )
        bars.append(
            f'<text x="{x + w/2:.1f}" y="{y-8:.1f}" font-size="12" font-family="Arial" text-anchor="middle" fill="#243033">{value:.4g}</text>'
        )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#f7f7f2"/>
<text x="{left}" y="32" font-size="20" font-family="Arial" font-weight="700" fill="#1f2a2d">{escape(title)}</text>
<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#445" stroke-width="1.5"/>
<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#445" stroke-width="1.5"/>
<text x="26" y="{height/2}" font-size="15" font-family="Arial" text-anchor="middle" transform="rotate(-90 26 {height/2})" fill="#243033">{escape(y_label)}</text>
{''.join(bars)}
</svg>"""
    path.write_text(svg, encoding="utf-8")
    return str(path)
