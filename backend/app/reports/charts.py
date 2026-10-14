"""Report graphics: spectral plot (matplotlib SVG), score gauge and risk icons (hand-built SVG).

Matplotlib is imported lazily so importing the report package stays cheap for the API.
"""

from __future__ import annotations

import io
import math
from xml.sax.saxutils import escape

from mukha_pipeline.schema.models import (
    AS7341_SPECTRAL_CHANNELS,
    AS7341_WAVELENGTH_NM,
    RiskLevel,
    SpectralReadingResult,
)

from app.reports.labels import RISK_COLORS, num, site_name

LESION_COLOR = "#a21caf"  # magenta: distinct from every risk colour
REFERENCE_COLOR = "#2563eb"  # blue: matches "reference only" on the site map
INK = "#1d2939"
MUTED = "#667085"
GRID = "#e4e7ec"
FONT_STACK = ["Inter", "DejaVu Sans"]


def _wavelength_rgb(nm: float) -> tuple[float, float, float]:
    """Approximate display colour of a wavelength (Bruton), for the decorative spectrum strip."""
    if 380 <= nm < 440:
        r, g, b = -(nm - 440) / 60, 0.0, 1.0
    elif nm < 490:
        r, g, b = 0.0, (nm - 440) / 50, 1.0
    elif nm < 510:
        r, g, b = 0.0, 1.0, -(nm - 510) / 20
    elif nm < 580:
        r, g, b = (nm - 510) / 70, 1.0, 0.0
    elif nm < 645:
        r, g, b = 1.0, -(nm - 645) / 65, 0.0
    else:
        r, g, b = 1.0, 0.0, 0.0
    if nm < 420:
        f = 0.3 + 0.7 * (nm - 380) / 40
    elif nm > 700:
        f = 0.3 + 0.7 * (780 - nm) / 80
    else:
        f = 1.0
    return (r * f) ** 0.8, (g * f) ** 0.8, (b * f) ** 0.8


def _series(reading: SpectralReadingResult) -> tuple[list[float], list[float], list[float]]:
    xs: list[float] = []
    ys: list[float] = []
    ss: list[float] = []
    for ch in AS7341_SPECTRAL_CHANNELS:
        v = reading.values.get(ch)
        if v is None or not math.isfinite(v):
            continue
        s = reading.values_std.get(ch, 0.0)
        xs.append(float(AS7341_WAVELENGTH_NM[ch]))
        ys.append(float(v))
        ss.append(float(s) if s is not None and math.isfinite(s) else 0.0)
    return xs, ys, ss


def spectral_y_label(signal_type: str) -> str:
    return "Reflectance" if signal_type == "reflectance" else "Relative signal"


def spectral_plot_svg(
    lesion: SpectralReadingResult, reference: SpectralReadingResult | None
) -> str:
    """Lesion vs reference spectra (mean ± std band) across the 8 visible AS7341 channels.

    The y label is "Reflectance" for calibrated readings and "Relative signal" otherwise. The
    540–580 nm haemoglobin band is shaded. Returns a standalone SVG document.
    """
    import matplotlib
    from matplotlib.figure import Figure

    rc = {
        "svg.fonttype": "none",
        "svg.hashsalt": "mukhaspectra-report",
        "font.family": FONT_STACK,
        "font.size": 7.5,
        "axes.edgecolor": "#98a2b3",
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
    }
    with matplotlib.rc_context(rc):
        fig = Figure(figsize=(4.3, 2.45), dpi=100)
        ax = fig.add_axes((0.125, 0.25, 0.85, 0.69))
        ax.axvspan(540, 580, color="#f4e4e4", zorder=0, lw=0)
        ax.text(560, 0.985, "Hb band", transform=ax.get_xaxis_transform(), ha="center",
                va="top", fontsize=6.5, color="#9b4f4f")

        all_lo: list[float] = []
        all_hi: list[float] = []
        curves = [(reference, REFERENCE_COLOR, "Reference"), (lesion, LESION_COLOR, "Lesion")]
        for reading, color, role in curves:
            if reading is None:
                continue
            xs, ys, ss = _series(reading)
            if not xs:
                continue
            lo = [y - s for y, s in zip(ys, ss, strict=True)]
            hi = [y + s for y, s in zip(ys, ss, strict=True)]
            all_lo += lo
            all_hi += hi
            ax.fill_between(xs, lo, hi, color=color, alpha=0.16, lw=0, zorder=2)
            label = f"{role} — {site_name(reading.site)} (n={reading.n_readings})"
            ax.plot(xs, ys, color=color, lw=1.7, marker="o", ms=3.2, zorder=3, label=label,
                    linestyle="--" if role == "Reference" else "-")

        if all_lo:
            span = max(all_hi) - min(all_lo)
            pad = max(span * 0.12, 1e-3)
            ax.set_ylim(max(0.0, min(all_lo) - pad) if min(all_lo) >= 0 else min(all_lo) - pad,
                        max(all_hi) + pad * 1.6)
        else:
            ax.text(0.5, 0.5, "No usable channel values", transform=ax.transAxes,
                    ha="center", va="center", color=MUTED)

        ax.set_xlim(402, 693)
        ax.set_xticks([AS7341_WAVELENGTH_NM[c] for c in AS7341_SPECTRAL_CHANNELS])
        ax.tick_params(axis="x", pad=7, length=2.5, labelsize=6.8)
        ax.tick_params(axis="y", length=2.5, labelsize=6.8)
        ax.set_xlabel("Wavelength (nm)", fontsize=7.2, labelpad=2)
        ax.set_ylabel(spectral_y_label(lesion.signal_type), fontsize=7.2)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="upper left", fontsize=6.6, frameon=False, handlelength=2.2,
                      borderaxespad=0.2)

        # Decorative visible-spectrum strip directly under the x axis.
        strip = fig.add_axes((0.125, 0.225, 0.85, 0.022))
        lo_nm, hi_nm = ax.get_xlim()
        steps = 120
        colors = [
            _wavelength_rgb(lo_nm + (hi_nm - lo_nm) * i / (steps - 1)) for i in range(steps)
        ]
        strip.imshow([colors], aspect="auto", extent=(lo_nm, hi_nm, 0, 1),
                     interpolation="bilinear")
        strip.set_axis_off()

        buf = io.StringIO()
        fig.savefig(buf, format="svg", metadata={"Date": None})
    return buf.getvalue()


# --------------------------------------------------------------------------------------------
# Hand-built SVG: risk icons and the score gauge
# --------------------------------------------------------------------------------------------


def risk_icon_svg(level: RiskLevel | str, color: str, size: float = 16) -> str:
    """Icon that encodes the level by shape, not colour: 1/2/3 filled bars, or "?"."""
    level = RiskLevel(level)
    head = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 16 16" class="risk-icon" aria-hidden="true">'
    )
    if level == RiskLevel.INCONCLUSIVE:
        body = (
            f'<circle cx="8" cy="8" r="6.6" fill="none" stroke="{color}" stroke-width="1.6"/>'
            f'<path d="M6 6.2 C6 4.9 7 4.2 8.1 4.2 C9.3 4.2 10.2 5 10.2 6.1 C10.2 7.6 8.2 7.7 '
            f'8.2 9.3" fill="none" stroke="{color}" stroke-width="1.6" stroke-linecap="round"/>'
            f'<circle cx="8.2" cy="11.7" r="1" fill="{color}"/>'
        )
    else:
        filled = {RiskLevel.LOW: 1, RiskLevel.MODERATE: 2, RiskLevel.HIGH: 3}[level]
        bars = []
        for i, h in enumerate((5.0, 9.0, 13.0)):
            x = 1.5 + i * 5
            opacity = "1" if i < filled else "0.32"
            bars.append(
                f'<rect x="{x}" y="{14.5 - h}" width="3.6" height="{h}" rx="0.8" '
                f'fill="{color}" opacity="{opacity}"/>'
            )
        body = "".join(bars)
    return head + body + "</svg>"


def score_gauge_svg(
    score: float | None,
    moderate: float,
    high: float,
    level: RiskLevel | str,
    *,
    width: float = 300,
) -> str:
    """Horizontal gauge: rule score against the MODERATE and HIGH thresholds."""
    level = RiskLevel(level)
    h = 44.0
    left, right = 6.0, width - 6.0
    top, bar_h = 18.0, 9.0
    vmax = max(high * 1.6, high + 3.0, (score or 0.0) * 1.12)

    def x_of(v: float) -> float:
        return left + (right - left) * max(0.0, min(v, vmax)) / vmax

    zones = [
        (0.0, moderate, RISK_COLORS[RiskLevel.LOW]),
        (moderate, high, RISK_COLORS[RiskLevel.MODERATE]),
        (high, vmax, RISK_COLORS[RiskLevel.HIGH]),
    ]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{h}" '
        f'viewBox="0 0 {width} {h}" font-family="Inter, \'DejaVu Sans\', sans-serif" '
        'class="gauge" role="img" aria-label="Rule score against thresholds">'
    ]
    for a, b, c in zones:
        parts.append(
            f'<rect x="{x_of(a):.1f}" y="{top}" width="{x_of(b) - x_of(a):.1f}" '
            f'height="{bar_h}" fill="{c["tint"]}" stroke="{c["strong"]}" stroke-width="0.6"/>'
        )
    for v, label in ((moderate, "Moderate"), (high, "High")):
        x = x_of(v)
        parts.append(
            f'<line x1="{x:.1f}" y1="{top - 2}" x2="{x:.1f}" y2="{top + bar_h + 2}" '
            f'stroke="{INK}" stroke-width="0.9"/>'
            f'<text x="{x:.1f}" y="{top + bar_h + 12}" text-anchor="middle" font-size="7.4" '
            f'fill="{MUTED}">{label} ≥ {num(v, 1)}</text>'
        )
    if score is not None and math.isfinite(score):
        x = x_of(score)
        c = RISK_COLORS[level]["strong"] if level in RISK_COLORS else INK
        anchor = "end" if x > width * 0.8 else ("start" if x < width * 0.2 else "middle")
        parts.append(
            f'<path d="M{x - 4.5:.1f},{top - 9} L{x + 4.5:.1f},{top - 9} L{x:.1f},{top - 2} Z" '
            f'fill="{c}"/>'
            f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + bar_h}" stroke="{c}" '
            'stroke-width="2.2"/>'
            f'<text x="{x:.1f}" y="{top - 11}" text-anchor="{anchor}" font-size="8" '
            f'font-weight="700" fill="{c}">{escape("Score " + num(score, 1))}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)
