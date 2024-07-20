import copy
import sys
from pathlib import Path

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
    @staticmethod
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
            return chain_revalidate(yaml, revalidator=on_failure, errs=(*errs, e))
        elif not errs:
            raise e
        err_msgs = ("Fix one of the following errors:", *errs, e)
        raise YAMLError("\n\n".join(str(err) for err in err_msgs))
    chain_revalidate(yaml, errs=errs)
