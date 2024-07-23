# ruff: noqa: E731

import copy
import re
from collections import defaultdict
from typing import Callable
from pathlib import Path

from PIL import Image, ImageDraw

from exceed_card_generator.types import (
    Chunk,
    FontFamily,
    Keyword,
    KwargsChunk,
    TextChunk,
)

ICON = lambda icon: lambda assets: Image.open(Path(assets) / f"7/icons/{icon}.png")
NUM = r"\d+(?:-\d+)?"
AMOUNT = rf"(?:[+-]?{NUM} )?"
KEYWORDS = [
    Keyword(
        rf"{AMOUNT}Range",
        dict(fill="#3ba7e1", bold=True),
        dict(fill="white", bold=False, icon=ICON("range")),
    ),
    Keyword(
        rf"{AMOUNT}Power",
        dict(fill="#ec2d28", bold=True),
        dict(fill="white", bold=False, icon=ICON("power")),
    ),
    Keyword(
        rf"{AMOUNT}Speed",
        dict(fill="#fbf008", bold=True),
        dict(fill="white", bold=False, icon=ICON("speed")),
    ),
    Keyword(
        rf"{AMOUNT}Armor",
        dict(fill="#bc5da2", bold=True),
        dict(fill="white", bold=False, icon=ICON("armor")),
    ),
    Keyword(
        rf"{AMOUNT}Guard",
        dict(fill="#62b944", bold=True),
        dict(fill="white", bold=False, icon=ICON("guard")),
    ),
    Keyword(
        r"Continuous Boost",
        dict(fill="orange", bold=True),
        dict(fill="white", bold=False, icon=ICON("continuous")),
    ),
    Keyword(rf"{AMOUNT}Force", dict(), dict(icon=ICON("force"))),
    Keyword(r"BEFORE:", dict(bold=True), dict(bold=False)),
    Keyword(rf"HIT(?:, RANGE {NUM})?:", dict(bold=True), dict(bold=False)),
    Keyword(r"AFTER:", dict(bold=True), dict(bold=False)),
    Keyword(r"NOW:", dict(bold=True), dict(bold=False)),
    Keyword(r"Advantage", dict(bold=True), dict(bold=False)),
    Keyword(r"Ignore Armor\.", dict(bold=True), dict(bold=False)),
    Keyword(r"Ignore Guard\.", dict(bold=True), dict(bold=False)),
    Keyword(r"Stun Immunity\.", dict(bold=True), dict(bold=False)),
    Keyword(r"You cannot be Pushed or Pulled\.", dict(bold=True), dict(bold=False)),
    Keyword(r"If the opponent initiated,", dict(bold=True), dict(bold=False)),
    Keyword(r"If you initiated,", dict(bold=True), dict(bold=False)),
    Keyword(r"Your attack is EX", dict(bold=True), dict(bold=False)),
    Keyword(r"“", dict(italic=True), dict()),
    Keyword(r"”", dict(), dict(italic=False)),
    Keyword(r"\(", dict(italic=True), dict()),
    Keyword(r"\)", dict(), dict(italic=False)),
]


def shadow_text(text_func: Callable, kwargs: dict, shadow: dict | None):
    if shadow := copy.copy(shadow):
        xy = tuple(map(sum, zip(kwargs["xy"], shadow.pop("offset"))))
        text_func(**dict(kwargs, xy=xy, **shadow))


# draw lines with horizontal centering and rich text formatting
def rich_text(
    text: str,
    *,
    img: Image,
    family: FontFamily,
    xy: tuple[int, int],
    spacing: int = 4,
    stroke_width: int | None = None,
    **text_kwargs,
):
    text_kwargs["stroke_width"] = stroke_width
    draw = ImageDraw.Draw(img)
    lines = text.split("\n")
    xy = list(xy)

    line_height = (
        draw.textbbox((0, 0), "A", family.regular, stroke_width=stroke_width)[3]
        + stroke_width
        + spacing
    )

    max_lines = 4
    pad_lines = (max_lines - len(lines)) // 2
    lines = pad_lines * [" "] + lines + pad_lines * [" "]
    if len(lines) < max_lines:
        xy[1] += line_height / 2

    bold = False
    italic = False
    line_chunks = []
    line_cum_widths = []
    for line in lines:
        chunks = rich_text_chunks(line)
        chunk_to_width = defaultdict(lambda: 0)

        def text_width(i, text, font, *_):
            kwargs = {"font": font, "stroke_width": stroke_width}
            chunk_to_width[i] = draw.textbbox((0, 0), text, **kwargs)[2]

        def icon_width(i, icon):
            chunk_to_width[i] = icon.size[0]

        fold_chunks(text_width, icon_width, chunks, family, bold, italic, text_kwargs)

        cum_widths = [chunk_to_width[0]]
        for i in range(len(chunks))[1:]:
            cum_widths.append(cum_widths[-1] + chunk_to_width[i])

        line_chunks.append(chunks)
        line_cum_widths.append(cum_widths)

    for line, chunks, cum_widths in zip(lines, line_chunks, line_cum_widths):

        def get_xy(i):
            x = xy[0] - cum_widths[-1] / 2
            if i > 0:
                x += cum_widths[i - 1]
            return x, xy[1]

        def draw_text(i, text, font, kwargs):
            kwargs = kwargs.copy()
            kwargs.pop("bold", None)
            kwargs.pop("italic", None)
            draw.text(get_xy(i), text, **{**kwargs, "font": font})

        def draw_icon(i, icon):
            x, y = get_xy(i)
            img.paste(icon, (int(x), int(y) + 8), mask=icon)

        bold, italic, color = fold_chunks(
            draw_text, draw_icon, chunks, family, bold, italic, text_kwargs
        )
        xy[1] += line_height


def rich_text_chunks(line: str) -> list[Chunk]:
    chunks = []
    i = 0
    for m in re.finditer(r"|".join(rf"({w.pattern})" for w in KEYWORDS), line):
        keyword = next(w for i, w in enumerate(KEYWORDS) if m.group(i + 1))
        if i < m.start():
            chunks.append(TextChunk(line[i : m.start()]))
        chunks.append(KwargsChunk(keyword.begin))
        chunks.append(TextChunk(m.group()))
        if keyword.end.get("icon"):
            chunks.append(TextChunk(" "))
        chunks.append(KwargsChunk(keyword.end))
        i = m.end()
    if i < len(line):
        chunks.append(TextChunk(line[i:]))
    return chunks


def fold_chunks(
    f_text: Callable,
    f_icon: Callable,
    line_chunks: list[Chunk],
    family: FontFamily,
    bold: bool,
    italic: bool,
    kwargs: dict,
) -> tuple[bool, bool, dict]:
    for i, chunk in enumerate(line_chunks):
        match chunk:
            case TextChunk(text):
                font = (
                    (kwargs.get("bold") and family.bold)
                    or (kwargs.get("italic") and family.italic)
                    or family.regular
                )
                f_text(i, text, font, kwargs)
            case KwargsChunk(kwargs_now):
                kwargs_now = kwargs_now.copy()
                if icon := kwargs_now.pop("icon", None):
                    f_icon(i, icon("assets/"))  # TODO: assets path
                kwargs |= kwargs_now
    return bold, italic, kwargs
