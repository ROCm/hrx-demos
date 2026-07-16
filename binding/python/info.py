"""Console entry point for HRX device diagnostics."""

from typing import NoReturn

from ._launcher import exec_native


def main() -> NoReturn:
    exec_native("hrx-info")
