#!/usr/bin/env python3
import os

figures = [
    "3.3.3.svg",
]


if __name__ == "__main__":
    for figure_file in figures:
        os.system(f'inkscape --export-type="pdf" {figure_file}')
