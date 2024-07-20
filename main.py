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

ASSETS = Path("assets" if Path("assets").is_dir() else os.environ["ASSETS"])
ASSET_IMG = lambda path: Image.open(ASSETS / path)
ASSET_JSON = lambda path: json.loads((ASSETS / path).read_text())

# Keyword eDSL
BOLD = [BoldChunk(True)]
NO_BOLD = [BoldChunk(False)]
ITALIC = [ItalicChunk(True)]
NO_ITALIC = [ItalicChunk(False)]
COLOR = lambda c: [ColorChunk(c)]
NO_COLOR = [ColorChunk("white")]
ICON = lambda icon: [TextChunk(" "), IconChunk(ASSET_IMG(f"s7/icons/{icon}.png"))]
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
    rich_text = MapCombined({"rich text": Str()}, Str(), Any()).and_then(
        Map(
            {
                "rich text": expr(Str()),
                "xy": expr(xy),
                "font": expr(font),
                "fill": expr(color),
                Optional("shadow"): expr(
                    Map({"offset": expr(xy), "fill": expr(color)})
                ),
                Optional("stroke_width"): expr(Int()),
                Optional("stroke_fill"): expr(color),
            }
        )
    )
    text = MapCombined({"text": Str()}, Str(), Any()).and_then(
        Map(
            {
                "text": expr(Str()),
                "xy": expr(xy),
                "font": expr(font),
                "fill": expr(color),
                Optional("shadow"): expr(
                    Map({"offset": expr(xy), "fill": expr(color)})
                ),
                Optional("anchor"): expr(Str()),
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
    layers = Seq(
        MapPattern(Str(), Any()).and_then(
            rich_text.or_else(text.or_else(image.or_else(match)))
        )
    )
    match.on_success = MapCombined({"match": Str()}, Str(), layers)
    font = Map(
        {
            "file": expr(Str()),
            "size": expr(Int()),
            Optional("variation"): expr(Str()),
        }
    )
    schema = Map(
        {
            Optional("normals"): Seq(MapPattern(Str(), Any())),
            "fonts": MapPattern(
                Str(),
                Map(
                    {
                        "regular": font,
                        Optional("bold"): font,
                        Optional("italic"): font,
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
        name: FontFamily(
            regular := load_font(**fonts["regular"]),
            load_font(**fonts["bold"]) if fonts.get("bold") else regular,
            load_font(**fonts["italic"]) if fonts.get("italic") else regular,
        )
        for name, fonts in data["fonts"].items()
    }
    for layer in data["layers"]:
        interpret_layer(layer, img, row, fonts)
    return img


def interpret_layer(
    layer: dict,
    img: Image.Image,
    row: dict[str, str],
    fonts: dict[str, FontFamily],
):
    if "rich text" in layer:
        data = interpret_all_expressions(layer, row)
        kwargs = dict(
            text=row[data["rich text"]],
            img=img,
            family=fonts[data["font"]],
            xy=data["xy"],
            fill=data["fill"],
            stroke_width=data.get("stroke_width", 0),
            stroke_fill=data.get("stroke_fill"),
        )
        if data.get("shadow"):
            shadow_text(rich_text, **(kwargs | data["shadow"]))
        rich_text(**kwargs)

    elif "text" in layer:
        data = interpret_all_expressions(layer, row)
        kwargs = dict(
            text=row[data["text"]],
            img=img,
            font=fonts[data["font"]].regular,
            xy=data["xy"],
            fill=data["fill"],
            anchor=data.get("anchor", "la"),
            stroke_width=data.get("stroke_width", 0),
            stroke_fill=data.get("stroke_fill"),
        )
        if data.get("shadow"):
            shadow_text(text, **(kwargs | data["shadow"]))
        text(**kwargs)

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
                for sub_layer in sub_layers:
                    interpret_layer(sub_layer, img, row, fonts)
                break


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
        return None
    elif isinstance(expr, str):
        return FilenameFormatter().format(expr, **row)
    return expr


def shadow_text(
    text_func: Callable,
    offset: tuple[int, int],
    **text_kwargs,
):
    text_func(
        **{
            **text_kwargs,
            "xy": (
                text_kwargs["xy"][0] + offset[0],
                text_kwargs["xy"][1] + offset[1],
            ),
        }
    )


def text(
    text: str,
    *,
    img: Image,
    font: ImageFont.ImageFont,
    xy: tuple[int, int],
    **text_kwargs,
):
    ImageDraw.Draw(img).multiline_text(xy, text, font=font, **text_kwargs)


# draw lines with horizontal centering and rich text formatting
def rich_text(
    text: str,
    *,
    img: Image,
    family: FontFamily,
    xy: tuple[int, int],
    spacing: int = 4,
    fill: str,
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
            kwargs = dict(font=font, stroke_width=stroke_width)
            chunk_to_width[i] = draw.textbbox((0, 0), text, **kwargs)[2]

        def icon_width(i, icon):
            chunk_to_width[i] = icon.size[0]

        fold_chunks(text_width, icon_width, chunks, family, bold, italic, fill)

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

        def draw_text(i, text, font, bold, italic, color):
            draw.text(get_xy(i), text, **{**text_kwargs, "font": font, "fill": color})

        def draw_icon(i, icon):
            x, y = get_xy(i)
            img.paste(icon, (int(x), int(y) + 8), mask=icon)

        bold, italic, color = fold_chunks(
            draw_text, draw_icon, chunks, family, bold, italic, fill
        )

        xy[1] += line_height


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
        default=ASSET_IMG("s7/card-back.jpeg"),
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

    with open(csv, newline="") as f:
        rows = list(DictReader(f))

    data = [{k.lower().replace(" ", "_"): v for k, v in row.items()} for row in rows]

    cards = {
        "Innate": [],
        "Exceed": [],
        "Special": [],
        "Ultra": [],
        "Normal": [],
    }

    data.extend(season_data.get("normals", []))

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
            if isinstance(cell, str) and cell.lower() == "false":
                cell = False
            row[key] = cell

        if not row["name"]:
            continue

        if no_normals and row["kind"] == "Normal":
            continue

        match row["kind"]:
            case "Special" | "Ultra" | "Normal":
                fname = Path(valid_filename(row["name"]) + ".png")
                default_copies = 2
            case "Innate" | "Exceed" as kind:
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

    foregrounds = cards["Normal"] + cards["Special"] + cards["Ultra"] + cards["Innate"]
    grid = grid or calculate_grid(len(foregrounds), 7, 7)
    g_fg = image_grid(foregrounds, *grid)
    g_fg.save(output / f"{character}-fg.jpg", quality=75, subsampling=0)

    backgrounds = cards["Exceed"]
    if (missing_bgs := len(foregrounds) - len(backgrounds)) > 0:
        backgrounds = [back] * missing_bgs + backgrounds
    g_bg = image_grid(backgrounds, *grid)
    g_bg.save(output / f"{character}-bg.jpg", quality=75, subsampling=0)

    references = []
    for card in cards["Special"] + cards["Ultra"]:
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


def load_font(
    file: str, size: int, variation: str | None = None
) -> ImageFont.ImageFont:
    font = ImageFont.truetype(file, size=size)
    if variation:
        font.set_variation_by_name(variation)
    return font


if __name__ == "__main__":
    main()
