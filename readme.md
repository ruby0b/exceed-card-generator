## eXceed Card Generator

A python script for generating custom GG:Strive [eXceed](https://boardgamegeek.com/boardgame/224483/exceed-fighting-system)-cards.

### Example

![Slayer](output/Slayer/Slayer-fg.jpg)

Arguments used: `input/Slayer --no-duplicates --grid 6x3`

### Usage

- Using [nix](https://nixos.org/): `nix run github:ruby0b/exceed-card-generator -- --help`
- Otherwise, make sure python and pillow are installed: `pip install -r requirements.txt`
  - Run the script: `python main.py --help`
  - Make sure to have the assets subdir in the current working directory (or set the ASSETS environment variable)

### Help

```
usage: exceed-card-generator [-h] [--no-duplicates] [--no-normals]
                             [--back BACK] [--grid GRID] [--output OUTPUT]
                             folder

positional arguments:
  folder           folder containing a CSV file and character images

options:
  -h, --help       show this help message and exit
  --no-duplicates  only put one copy of each card in the grid
  --no-normals     do not include normal moves
  --back BACK      the image to use for the card backs
  --grid GRID      tiling grid format (format: <rows>x<columns>)
  --output OUTPUT  output folder (default = output/<input folder name>)
```

### Notes

- Card art will be fitted to 665x570 pixels

### License

This software project is licensed under the GPL-3.0 or later. See the [LICENSE](LICENSE) file for details.
The license does not apply to images and fonts used in the assets directory which are owned by their respective creators.
