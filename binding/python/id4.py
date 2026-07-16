"""Console entry point for Ideogram 4 image generation."""

from typing import NoReturn

from ._launcher import exec_native


def main() -> NoReturn:
    exec_native("hrx-id4")
