#!/usr/bin/env python
# ruff: noqa: E731
# flake8: noqa: E501,E203

# Copyright 2024 https://github.com/ruby0b
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# You can receive a copy of the GNU General Public License at <http://www.gnu.org/licenses/>.

import json
import os
import re
from argparse import ArgumentParser
from collections import defaultdict, namedtuple
from csv import DictReader
from inspect import signature
from pathlib import Path
from string import ascii_letters, digits
from typing import Literal, Callable

from PIL import Image, ImageDraw, ImageFont, ImageOps

# Types
FontFamily = namedtuple("FontFamily", ["regular", "bold", "italic"])
Keyword = namedtuple("Keyword", ["pattern", "begin", "end"])
TextChunk = namedtuple("TextChunk", ["text"])  # text with the current formatting
ColorChunk = namedtuple("ColorChunk", ["color"])  # override current color
ItalicChunk = namedtuple("ItalicChunk", ["italic"])  # begin/end italic
BoldChunk = namedtuple("BoldChunk", ["bold"])  # begin/end bold
IconChunk = namedtuple("IconChunk", ["icon"])  # insert icon
Chunk = TextChunk | ColorChunk | ItalicChunk | BoldChunk | IconChunk

# Images
ASSETS = Path("assets" if Path("assets").is_dir() else os.environ["ASSETS"])
ASSET_IMG = lambda path: Image.open(ASSETS / path)
ASSET_JSON = lambda path: json.loads((ASSETS / path).read_text())
SPECIAL_BASE = ASSET_IMG("special-base.png")
PORTRAIT_MASK = ASSET_IMG("mask-portrait.png")
INSTANT_BOOST = ASSET_IMG("special-instant.png")
INSTANT_BOOST_CANCEL = ASSET_IMG("special-instant-cancel.png")
CONTINUOUS_BOOST = ASSET_IMG("special-continuous.png")
CONTINUOUS_BOOST_CANCEL = ASSET_IMG("special-continuous-cancel.png")
ULTRA_BAR = ASSET_IMG("special-ultra.png")
SPECIAL_BAR = ASSET_IMG("special-special.png")
ARMOR_BAR = ASSET_IMG("special-armor.png")
GUARD_BAR = ASSET_IMG("special-guard.png")
INNATE_BG = ASSET_IMG("innate-bg.png")
INNATE_BASE = ASSET_IMG("innate-base.png")
EXCEED_BG = ASSET_IMG("exceed-bg.png")
EXCEED_BASE = ASSET_IMG("exceed-base.png")

# Data
NORMALS_DATA = ASSET_JSON("normals.json")

# Sizes
TEXT_FONT_SIZE = 30
TEXT_LINE_HEIGHT = 34
STAT_STROKE_WIDTH = 2
DESCRIPTION_WIDTH = 570

# Fonts
GGST_FONT_FILE = ASSETS / "Impact-Strive.otf"
TEXT_FONT_FILE = ASSETS / "Montserrat-VariableFont_wght.ttf"
TEXT_FONT_ITALIC_FILE = ASSETS / "Montserrat-Italic-VariableFont_wght.ttf"
TEXT_FONT = ImageFont.truetype(TEXT_FONT_FILE, size=TEXT_FONT_SIZE)
TEXT_FONT.set_variation_by_name("Regular")
TEXT_FONT_BOLD = ImageFont.truetype(TEXT_FONT_FILE, size=TEXT_FONT_SIZE)
TEXT_FONT_BOLD.set_variation_by_name("Bold")
TEXT_FONT_ITALIC = ImageFont.truetype(TEXT_FONT_ITALIC_FILE, size=TEXT_FONT_SIZE)
TEXT_FONT_ITALIC.set_variation_by_name("Italic")
TEXT_FONT_FAMILY = FontFamily(TEXT_FONT, TEXT_FONT_BOLD, TEXT_FONT_ITALIC)
TITLE_FONT = ImageFont.truetype(GGST_FONT_FILE, size=40)
GAUGE_FONT = ImageFont.truetype(GGST_FONT_FILE, size=66)
CHARACTER_NAME_FONT = ImageFont.truetype(GGST_FONT_FILE, size=56)

# Colors
INSTANT_BOOST_NAME_COLOR = "#d989b5"
CONTINUOUS_BOOST_NAME_COLOR = "#ffa864"
GAUGE_COST_COLOR = "#00adef"
RANGE_STROKE = "#374fa2"
POWER_STROKE = "#6a1f1f"
SPEED_STROKE = "#7c531f"
ARMOR_STROKE = "#581f59"
GUARD_STROKE = "#2d5529"


# Pixel Coordinates
class SPECIAL_XY:
    PORTRAIT_SIZE = (150, 150)
    GAUGE_TOPLEFT = (62, 30)
    NAME_TOPLEFT = (198, 43)
    DESCRIPTION_TOPLEFT = (96, 680)
    BOOST_NAME_TOPLEFT = (67, 840)
    BOOST_DESCRIPTION_TOPLEFT = (96, 875)
    RANGE_TOPLEFT = (49, 222)
    POWER_TOPLEFT = (49, 312)
    SPEED_TOPLEFT = (49, 402)
    ARMOR_TOPLEFT = (49, 492)
    GUARD_TOPLEFT = (49, 584)


class INNATE_XY:
    NAME_TOPLEFT = (65, 725)
    DESCRIPTION_TOPLEFT = (96, 865)
    GAUGE_TOPLEFT = (598, 732)


# Keyword eDSL
BOLD = [BoldChunk(True)]
NO_BOLD = [BoldChunk(False)]
ITALIC = [ItalicChunk(True)]
NO_ITALIC = [ItalicChunk(False)]
COLOR = lambda c: [ColorChunk(c)]
NO_COLOR = [ColorChunk("white")]
ICON = lambda icon: [TextChunk(" "), IconChunk(ASSET_IMG(f"icon-{icon}.png"))]
NUM = r"\d+(?:-\d+)?"
AMOUNT = rf"(?:[+-]?{NUM} )?"
KEYWORDS = [
    Keyword(
        rf"{AMOUNT}Range", BOLD + COLOR("#3ba7e1"), NO_BOLD + NO_COLOR + ICON("range")
    ),
    Keyword(
        rf"{AMOUNT}Power", BOLD + COLOR("#ec2d28"), NO_BOLD + NO_COLOR + ICON("power")
    ),
    Keyword(
        rf"{AMOUNT}Speed", BOLD + COLOR("#fbf008"), NO_BOLD + NO_COLOR + ICON("speed")
    ),
    Keyword(
        rf"{AMOUNT}Armor", BOLD + COLOR("#bc5da2"), NO_BOLD + NO_COLOR + ICON("armor")
    ),
    Keyword(
        rf"{AMOUNT}Guard", BOLD + COLOR("#62b944"), NO_BOLD + NO_COLOR + ICON("guard")
    ),
    Keyword(
        r"Continuous Boost",
        BOLD + COLOR("orange"),
        NO_BOLD + NO_COLOR + ICON("continuous"),
    ),
    Keyword(r"{AMOUNT}Force", [], [ICON("force")]),
    Keyword(r"BEFORE:", BOLD, NO_BOLD),
    Keyword(rf"HIT(?:, RANGE {NUM})?:", BOLD, NO_BOLD),
    Keyword(r"AFTER:", BOLD, NO_BOLD),
    Keyword(r"NOW:", BOLD, NO_BOLD),
    Keyword(r"Advantage", BOLD, NO_BOLD),
    Keyword(r"Ignore Armor\.", BOLD, NO_BOLD),
    Keyword(r"Ignore Guard\.", BOLD, NO_BOLD),
    Keyword(r"Stun Immunity\.", BOLD, NO_BOLD),
    Keyword(r"You cannot be Pushed or Pulled\.", BOLD, NO_BOLD),
    Keyword(r"If the opponent initiated,", BOLD, NO_BOLD),
    Keyword(r"If you initiated,", BOLD, NO_BOLD),
    Keyword(r"Your attack is EX", BOLD, NO_BOLD),
    Keyword(r"“", ITALIC, []),
    Keyword(r"”", [], NO_ITALIC),
    Keyword(r"\(", ITALIC, []),
    Keyword(r"\)", [], NO_ITALIC),
]


def make_character_card(
    *,
    image: Image.Image | None,
    name: str,
    description: str,
    kind: Literal["innate", "exceed"],
    cost: str | None,
) -> Image.Image:
    output = Image.new("RGB", INNATE_BG.size)
    match kind:
        case "innate":
            bg, base = INNATE_BG, INNATE_BASE
        case "exceed":
            bg, base = EXCEED_BG, EXCEED_BASE
        case _:
            raise ValueError(f"Unknown kind: {kind}")
    output.paste(bg)
    if image:
        image = ImageOps.fit(image, bg.size)
        output.paste(image, mask=image)
    output.paste(base, mask=base)

    if cost:
        text(
            cost,
            img=output,
            font=GAUGE_FONT,
            topleft=INNATE_XY.GAUGE_TOPLEFT,
            fill=GAUGE_COST_COLOR,
        )

    text(
        name,
        img=output,
        font=CHARACTER_NAME_FONT,
        topleft=INNATE_XY.NAME_TOPLEFT,
        fill="white",
    )

    rich_text(
        description,
        img=output,
        family=TEXT_FONT_FAMILY,
        topleft=INNATE_XY.DESCRIPTION_TOPLEFT,
        max_width=DESCRIPTION_WIDTH,
    )

    return output


def make_move_card(
    *,
    image: Image.Image | None,
    portrait: Image.Image | None = None,
    name: str,
    description: str,
    boost: str,
    boost_name: str,
    boost_type: Literal["instant", "continuous"],
    cancel: bool,
    kind: Literal["special", "ultra"],
    cost: str | None,
    range: str,
    power: str,
    speed: str,
    armor: str | None,
    guard: str | None,
) -> Image.Image:
    output = Image.new("RGB", SPECIAL_BASE.size)

    if image:
        output.paste(ImageOps.fit(image, (665, 570)), box=(85, 100))

    output.paste(SPECIAL_BASE, mask=SPECIAL_BASE)

    # Gauge
    if kind == "ultra":
        assert cost is not None
        output.paste(ULTRA_BAR, mask=ULTRA_BAR)
        text(
            cost,
            img=output,
            font=GAUGE_FONT,
            topleft=SPECIAL_XY.GAUGE_TOPLEFT,
            fill=GAUGE_COST_COLOR,
        )
    elif kind == "normal":
        normal_bar = ASSET_IMG(f"normal-{name.lower()}.png")
        output.paste(normal_bar, mask=normal_bar)
    elif kind == "special":
        output.paste(SPECIAL_BAR, mask=SPECIAL_BAR)
        if portrait:
            portrait = ImageOps.fit(portrait, SPECIAL_XY.PORTRAIT_SIZE)
            output.paste(portrait, box=(0, 0), mask=PORTRAIT_MASK)

    if kind != "normal":
        # Name shadow
        text(
            name,
            img=output,
            font=TITLE_FONT,
            topleft=(SPECIAL_XY.NAME_TOPLEFT[0] - 2, SPECIAL_XY.NAME_TOPLEFT[1] + 2),
            fill="black",
        )
        # Name
        text(
            name,
            img=output,
            font=TITLE_FONT,
            topleft=SPECIAL_XY.NAME_TOPLEFT,
            fill="white",
        )

    # Description
    rich_text(
        description,
        align=True,
        img=output,
        family=TEXT_FONT_FAMILY,
        topleft=SPECIAL_XY.DESCRIPTION_TOPLEFT,
        max_width=DESCRIPTION_WIDTH,
    )

    # Range
    centered_text(
        range,
        img=output,
        font=TITLE_FONT,
        topleft=SPECIAL_XY.RANGE_TOPLEFT,
        max_width=60,
        fill="white",
        stroke_width=STAT_STROKE_WIDTH,
        stroke_fill=RANGE_STROKE,
    )
    # Power
    text(
        power,
        img=output,
        font=TITLE_FONT,
        topleft=SPECIAL_XY.POWER_TOPLEFT,
        fill="white",
        stroke_width=STAT_STROKE_WIDTH,
        stroke_fill=POWER_STROKE,
    )
    # Speed
    text(
        speed,
        img=output,
        font=TITLE_FONT,
        topleft=SPECIAL_XY.SPEED_TOPLEFT,
        fill="white",
        stroke_width=STAT_STROKE_WIDTH,
        stroke_fill=SPEED_STROKE,
    )

    # Armor
    if armor:
        output.paste(ARMOR_BAR, mask=ARMOR_BAR)
        text(
            armor,
            img=output,
            font=TITLE_FONT,
            topleft=SPECIAL_XY.ARMOR_TOPLEFT,
            fill="white",
            stroke_width=STAT_STROKE_WIDTH,
            stroke_fill=ARMOR_STROKE,
        )
    # Guard
    if guard:
        output.paste(GUARD_BAR, mask=GUARD_BAR)
        text(
            guard,
            img=output,
            font=TITLE_FONT,
            topleft=SPECIAL_XY.GUARD_TOPLEFT,
            fill="white",
            stroke_width=STAT_STROKE_WIDTH,
            stroke_fill=GUARD_STROKE,
        )

    if boost_type == "instant":
        boost_name_color = INSTANT_BOOST_NAME_COLOR
        boost_image = INSTANT_BOOST
        boost_cancel_image = INSTANT_BOOST_CANCEL
    else:
        boost_name_color = CONTINUOUS_BOOST_NAME_COLOR
        boost_image = CONTINUOUS_BOOST
        boost_cancel_image = CONTINUOUS_BOOST_CANCEL

    output.paste(boost_image, mask=boost_image)
    if cancel:
        output.paste(boost_cancel_image, mask=boost_cancel_image)

    # Boost name
    text(
        boost_name,
        img=output,
        font=TEXT_FONT_BOLD,
        topleft=SPECIAL_XY.BOOST_NAME_TOPLEFT,
        fill=boost_name_color,
    )
    # Boost description
    rich_text(
        boost,
        img=output,
        family=TEXT_FONT_FAMILY,
        topleft=SPECIAL_XY.BOOST_DESCRIPTION_TOPLEFT,
        max_width=DESCRIPTION_WIDTH,
    )

    return output


def text(
    text: str,
    *,
    img: Image,
    font: ImageFont.ImageFont,
    topleft: tuple[int, int],
    **text_kwargs,
):
    ImageDraw.Draw(img).text(topleft, text, font=font, **text_kwargs)


# horizontally centered text
def centered_text(
    text: str,
    *,
    img: Image,
    font: ImageFont.ImageFont,
    topleft: tuple[int, int],
    max_width: int,
    **text_kwargs,
):
    draw = ImageDraw.Draw(img)
    msg_size = draw.textbbox((0, 0), text, font=font)[2:]
    draw.text(
        (topleft[0] + (max_width - msg_size[0]) / 2, topleft[1]),
        text,
        font=font,
        **text_kwargs,
    )


# draw lines with horizontal centering and rich text formatting
def rich_text(
    text: str,
    *,
    img: Image,
    family: FontFamily,
    topleft: tuple[int, int],
    max_width: int,
    **text_kwargs,
):
    draw = ImageDraw.Draw(img)
    lines = text.split("\n")
    topleft = list(topleft)

    if 0 < len(lines) < 3:
        lines = [" "] + lines + [" "]
    if len(lines) == 3:
        topleft[1] += TEXT_LINE_HEIGHT / 2

    bold = False
    italic = False
    color = "white"
    for line in lines:
        chunks = rich_text_chunks(line)

        chunk_to_width = defaultdict(lambda: 0)

        def text_width(i, text, font, *_):
            chunk_to_width[i] = draw.textbbox((0, 0), text, font=font)[2]

        def icon_width(i, icon):
            chunk_to_width[i] = icon.size[0]

        fold_chunks(text_width, icon_width, chunks, family, bold, italic, color)

        cum_widths = [chunk_to_width[0]]
        for i in range(len(chunks))[1:]:
            cum_widths.append(cum_widths[-1] + chunk_to_width[i])

        def xy(i):
            x = topleft[0] + (max_width - cum_widths[-1]) / 2
            if i > 0:
                x += cum_widths[i - 1]
            return x, topleft[1]

        def draw_text(i, text, font, bold, italic, color):
            draw.text(xy(i), text, font=font, fill=color, **text_kwargs)

        def draw_icon(i, icon):
            x, y = xy(i)
            img.paste(icon, (int(x), int(y) + 8), mask=icon)

        bold, italic, color = fold_chunks(
            draw_text, draw_icon, chunks, family, bold, italic, color
        )

        topleft[1] += TEXT_LINE_HEIGHT


def rich_text_chunks(line: str) -> list[Chunk]:
    chunks = []
    i = 0
    for m in re.finditer(r"|".join(rf"({w.pattern})" for w in KEYWORDS), line):
        keyword = next(w for i, w in enumerate(KEYWORDS) if m.group(i + 1))
        if i < m.start():
            chunks.append(TextChunk(line[i : m.start()]))
        chunks.extend(keyword.begin)
        chunks.append(TextChunk(m.group()))
        chunks.extend(keyword.end)
        i = m.end()
    if i < len(line):
        chunks.append(TextChunk(line[i:]))
    return chunks


def fold_chunks(
    f: Callable,
    f_icon: Callable,
    line_chunks: list[Chunk],
    family: FontFamily,
    bold: bool,
    italic: bool,
    color: str,
) -> tuple[bool, bool, str]:
    for i, chunk in enumerate(line_chunks):
        match chunk:
            case TextChunk(text):
                font = (
                    (bold and family.bold)
                    or (italic and family.italic)
                    or family.regular
                )
                f(i, text, font, bold, italic, color)
            case BoldChunk(bold_now):
                bold = bold_now
            case ItalicChunk(italic_now):
                italic = italic_now
            case ColorChunk(color_now):
                color = color_now
            case IconChunk(icon):
                f_icon(i, icon)
            case _:
                pass
    return bold, italic, color


def main():
    p = ArgumentParser()
    p.add_argument(
        "folder", help="folder containing a CSV file and character images", type=Path
    )
    p.add_argument(
        "--no-duplicates",
        help="only put one copy of each card in the grid",
        action="store_true",
    )
    p.add_argument(
        "--no-normals",
        help="do not include normal moves",
        action="store_true",
    )
    p.add_argument(
        "--back",
        help="the image to use for the card backs",
        type=Image.open,
        default=ASSET_IMG("card-back.jpeg"),
    )
    p.add_argument(
        "--grid",
        help="tiling grid format (format: <rows>x<columns>)",
        type=size_2d,
        default=None,
    )
    p.add_argument(
        "--output",
        help="output folder (default = output/<input folder name>)",
        type=Path,
    )
    csv_to_cards(**vars(p.parse_args()))


def size_2d(s: str) -> tuple[int, int]:
    if len(xy := tuple(map(int, s.split("x")))) != 2:
        raise ValueError("Expected 2 integers separated by 'x'")
    return xy


def csv_to_cards(
    folder: Path,
    no_duplicates: bool,
    no_normals: bool,
    back: Image.Image,
    grid: tuple[int, int] | None = None,
    output: Path | None = None,
):
    if not folder.exists():
        print(f"Folder not found: {folder}")
        return

    if not (csv := next(folder.glob("*.csv"), None)):
        print(f"No CSV file found in {folder}")
        return

    print(f"Using CSV file: {csv}")

    character = folder.name
    output = output or Path("output") / character
    output.mkdir(exist_ok=True, parents=True)

    portrait = open_image(folder / "portrait.jpg")

    with open(csv, newline="") as f:
        rows = list(DictReader(f))

    data = [{k.lower().replace(" ", "_"): v for k, v in row.items()} for row in rows]

    cards = {
        "innate": [],
        "exceed": [],
        "special": [],
        "ultra": [],
        "normal": [],
    }

    data.extend(NORMALS_DATA)

    for row in data:
        for key, cell in row.items():
            if isinstance(cell, str):
                # remove surrounding whitespace
                cell = "\n".join(line.strip() for line in cell.split("\n"))
                # make leading quotes fancy
                cell = re.sub(
                    r"^\"(.*)\"",
                    lambda m: f"“{m.group(1)}”",
                    cell,
                    count=1,
                    flags=re.DOTALL,
                )
            if key in ["kind", "boost_type"]:
                cell = cell.lower()
            if isinstance(cell, str) and cell.lower() == "false":
                cell = False
            row[key] = cell

        if not row["name"]:
            continue

        if no_normals and row["kind"] == "normal":
            continue

        match row["kind"]:
            case "special" | "ultra" | "normal":
                func = make_move_card
                row["portrait"] = portrait
                fname = valid_filename(row["name"])
                default_copies = 2
            case "innate" | "exceed" as kind:
                func = make_character_card
                fname = f"{kind}.png"
                default_copies = 1
            case "" | None:
                continue
            case _:
                print(f"Ignoring card with unknown kind: {row['kind']}")
                continue

        sig = signature(func)
        for key, cell in row.copy().items():
            if key not in sig.parameters:
                if cell:
                    print(f"Ignoring unknown column: {key}")
                del row[key]
        for key in sig.parameters:
            if key not in row:
                row[key] = None

        row["image"] = open_image(folder / fname)
        card = func(**row)
        card.save(output / fname)
        copies = 1 if no_duplicates else int(row.get("copies") or default_copies)
        cards[row["kind"]].extend(copies * [card])

    foregrounds = cards["normal"] + cards["special"] + cards["ultra"] + cards["innate"]
    grid = grid or calculate_grid(len(foregrounds), 7, 7)
    g_fg = image_grid(foregrounds, *grid)
    g_fg.save(output / f"{character}-fg.jpg", quality=75, subsampling=0)

    backgrounds = cards["exceed"]
    if (missing_bgs := len(foregrounds) - len(backgrounds)) > 0:
        backgrounds = [back] * missing_bgs + backgrounds
    g_bg = image_grid(backgrounds, *grid)
    g_bg.save(output / f"{character}-bg.jpg", quality=75, subsampling=0)

    references = []
    for card in cards["special"] + cards["ultra"]:
        if card not in references:
            references.append(card)
    g_ref = image_grid(references, *calculate_grid(len(references), 10, 2))
    g_ref.save(output / f"{character}-ref.jpg", quality=75, subsampling=0)


def valid_filename(name: str) -> Path:
    valid_fname = ascii_letters + digits + "_-"
    return Path(
        "".join(filter(lambda c: c in valid_fname, name.replace(" ", "_"))) + ".png"
    )


def calculate_grid(n: int, w: int, min_h: int) -> tuple[int, int]:
    return (w, max(min_h, n // w + 1))


def open_image(path: Path) -> Image.Image | None:
    if path.exists():
        return Image.open(path)
    print(f"Image not found: {path}")
    return None


def image_grid(images: list[Image.Image], w: int, h: int) -> Image.Image:
    width, height = images[0].size
    grid = Image.new("RGB", (w * width, h * height))
    for i, img in enumerate(images):
        grid.paste(img, ((i % w) * width, (i // w) * height))
    return grid


if __name__ == "__main__":
    main()
