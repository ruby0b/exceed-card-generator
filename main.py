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

import copy
import json
import os
import re
import string
import sys
from argparse import ArgumentParser
from collections import defaultdict, namedtuple
from csv import DictReader
from pathlib import Path
from string import ascii_letters, digits
from typing import Callable

from strictyaml import (
    Any,
    Bool,
    dirty_load,
    FixedSeq,
    Int,
    Map,
    MapCombined,
    MapPattern,
    Optional,
    Seq,
    Str,
    Validator,
    YAML,
    YAMLError,
)
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
    Keyword(rf"{AMOUNT}Force", [], [ICON("force")]),
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


def monkeypatch_method(cls):
    def decorator(func):
        setattr(cls, func.__name__, func)
        return func

    return decorator


@monkeypatch_method(Validator)
def and_then(self, other):
    new = copy.deepcopy(self)
    setattr(new, "on_success", other)
    return new


@monkeypatch_method(Validator)
def or_else(self, other):
    new = copy.deepcopy(self)
    setattr(new, "on_failure", other)
    return new


class Schema:
    def expr(t):
        t = copy.deepcopy(t)
        t.on_failure = MapCombined({"match": Str()}, Str(), t)
        return Any().and_then(t)

    color = Str()
    font = Str()
    xy = FixedSeq([Int(), Int()])
    text = MapCombined({"text": Str()}, Str(), Any()).and_then(
        Map(
            {
                "text": expr(Str()),
                "xy": expr(xy),
                "font": expr(font),
                "fill": expr(color),
                Optional("width"): expr(Int()),
                Optional("shadow"): expr(
                    Map({"offset": expr(xy), "fill": expr(color)})
                ),
                Optional("rich"): expr(Bool()),
                Optional("align"): expr(Bool()),
                Optional("centered"): expr(Bool()),
                Optional("stroke_width"): expr(Int()),
                Optional("stroke_fill"): expr(color),
            }
        )
    )
    image = MapCombined({"image": Str()}, Str(), Any()).and_then(
        Map(
            {
                "image": expr(Str()),
                Optional("xy"): expr(xy),
                Optional("fit"): expr(xy),
                Optional("mask", default=True): expr(Bool() | Str()),
            }
        )
    )
    match = MapCombined({"match": Str()}, Str(), Any())
    layers = Seq(MapPattern(Str(), Any()).and_then(text.or_else(image.or_else(match))))
    match.on_success = MapCombined({"match": Str()}, Str(), layers)
    schema = Map(
        {
            "fonts": MapPattern(
                Str(),
                Map(
                    {
                        "file": Str(),
                        "size": Int(),
                        Optional("variations"): Map(
                            {"regular": Str(), "bold": Str(), "italic": Str()}
                        ),
                    }
                ),
            ),
            "layers": layers,
        }
    )


def load_season(season_file: Path) -> dict:
    yaml_str = season_file.read_text()
    try:
        yaml = dirty_load(
            yaml_str, Schema.schema, label=season_file, allow_flow_style=True
        )
        bfs_revalidate(yaml)
    except YAMLError as error:
        print(error)
        sys.exit(1)
    return yaml.data


def bfs_revalidate(yaml: YAML):
    chain_revalidate(yaml)
    if yaml.is_sequence():
        for value in yaml:
            bfs_revalidate(value)
    elif yaml.is_mapping():
        for key, value in yaml.items():
            bfs_revalidate(value)


def chain_revalidate(yaml: YAML, revalidator=None, errs: tuple[YAMLError, ...] = ()):
    if not (revalidator := revalidator or getattr(yaml.validator, "on_success", None)):
        return
    try:
        yaml.revalidate(revalidator)
    except YAMLError as e:
        if on_failure := getattr(revalidator, "on_failure", None):
            chain_revalidate(yaml, revalidator=on_failure, errs=(*errs, e))
        else:
            if not errs:
                raise e
            err_msgs = ("Fix one of the following errors:", *errs, e)
            raise YAMLError("\n\n".join(str(err) for err in err_msgs))
    chain_revalidate(yaml, errs=errs)


def interpret(data: dict, row: dict[str, str]) -> Image.Image:
    img = Image.new("RGB", (750, 1024))
    fonts = {
        name: ImageFont.truetype(font["file"], size=font["size"])
        for name, font in data["fonts"].items()
    }
    for layer in data["layers"]:
        interpret_layer(layer, img, row, fonts)
    return img


def interpret_layer(
    layer: dict, img: Image.Image, row: dict[str, str], fonts: dict[str, ImageFont]
):
    if "text" in layer:
        data = interpret_all_expressions(layer, row)
        text(
            row[data["text"]],
            img=img,
            font=fonts[data["font"]],
            topleft=data["xy"],
            fill=data["fill"],
            # width=data.get("width"),
            # shadow=data.get("shadow"),
            # rich=data.get("rich"),
            # align=data.get("align"),
            # centered=data.get("centered"),
            stroke_width=data.get("stroke_width", 0),
            stroke_fill=data.get("stroke_fill"),
        )

    elif "image" in layer:
        data = interpret_all_expressions(layer, row)
        image_path = data["image"]
        if not (image := open_image(image_path)):
            return

        if fit := data.get("fit"):
            image = ImageOps.fit(image, fit)

        mask = None
        if isinstance(mask_val := data.get("mask"), str):
            mask = open_image(mask_val)
        elif mask_val is True:
            mask = image

        img.paste(image, box=data.get("xy", (0, 0)), mask=mask)

    elif "match" in layer:
        value = row[layer["match"]]
        for pattern, sub_layers in layer.items():
            if pattern == "match":
                continue
            if re.match(pattern, value):
                print(f"Match: {pattern}")
                for sub_layer in sub_layers:
                    interpret_layer(sub_layer, img, row, fonts)


def interpret_all_expressions(d: dict, row: dict[str, str]):
    return {k: interpret_expression(v, row) for k, v in d.items()}


class FilenameFormatter(string.Formatter):
    def convert_field(self, field, conversion):
        return valid_filename(super().convert_field(field, conversion))


def interpret_expression(expr, row: dict[str, str]):
    if isinstance(expr, dict) and "match" in expr:
        value = row[expr["match"]]
        for pattern, sub_expr in expr.items():
            if pattern == "match":
                continue
            if re.match(pattern, value):
                return interpret_expression(sub_expr, row)
    elif isinstance(expr, str):
        return FilenameFormatter().format(expr, **row)
    return expr


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
        "--season",
        help="the season yaml to use",
        type=Path,
        required=True,
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
    season: Path,
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

    season_data = load_season(season)

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
                row["portrait"] = portrait
                fname = Path(valid_filename(row["name"]) + ".png")
                default_copies = 2
            case "innate" | "exceed" as kind:
                fname = f"{kind}.png"
                default_copies = 1
            case "" | None:
                continue
            case _:
                print(f"Ignoring card with unknown kind: {row['kind']}")
                continue

        row.setdefault("owner", character)
        row["image"] = open_image(folder / fname)
        card = interpret(season_data, row)
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


def valid_filename(name: str) -> str:
    valid_fname = ascii_letters + digits + "_-"
    return "".join(filter(lambda c: c in valid_fname, name.replace(" ", "_")))


def calculate_grid(n: int, w: int, min_h: int) -> tuple[int, int]:
    return (w, max(min_h, n // w + 1))


def open_image(path: Path | str) -> Image.Image | None:
    path = Path(path)
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
