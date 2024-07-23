#!/usr/bin/env python
# ruff: noqa: E731

import os
import re
import string
from argparse import ArgumentParser
from csv import DictReader
from pathlib import Path
from string import ascii_letters, digits

from PIL import Image, ImageDraw, ImageFont, ImageOps

from exceed_card_generator import schema, text, types


ASSETS = Path("assets" if Path("assets").is_dir() else os.environ["ASSETS"])


def interpret(data: dict, row: dict[str, str]) -> Image.Image:
    img = Image.new("RGB", (750, 1024))
    fonts = {
        name: types.FontFamily(
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
    fonts: dict[str, types.FontFamily],
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
        text.shadow_text(text.rich_text, kwargs, data.get("shadow"))
        text.rich_text(**kwargs)

    elif "text" in layer:
        data = interpret_all_expressions(layer, row)
        kwargs = dict(
            text=row[data["text"]],
            font=fonts[data["font"]].regular,
            xy=data["xy"],
            fill=data["fill"],
            anchor=data.get("anchor", "la"),
            stroke_width=data.get("stroke_width", 0),
            stroke_fill=data.get("stroke_fill"),
        )
        text_func = ImageDraw.Draw(img).multiline_text
        text.shadow_text(text_func, kwargs, data.get("shadow"))
        text_func(**kwargs)

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


def main():
    p = ArgumentParser()
    p.add_argument(
        "folder", help="folder containing a CSV file and character images", type=Path
    )
    p.add_argument(
        "--season",
        help=f"the season to use (available: {', '.join(d.name for d in ASSETS.glob('*'))})",
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
        default=Image.open(ASSETS / "7/card-back.jpeg"),
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

    season_data = schema.load_season(ASSETS / season / "season.yaml")

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
