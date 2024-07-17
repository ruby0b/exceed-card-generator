#!/usr/bin/env sh
# Script to easily paste screenshots from clipboard

[ -d "$1" ] || {
    echo "usage: $0 IMAGE_DIR" >&2
    exit 1
}

while read -r line; do
    xclip -selection clipboard -o >"$1/$line.png"
    ./main.py "$1"
done
