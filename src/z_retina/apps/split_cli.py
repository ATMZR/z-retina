"""CLI: generate split JSON (same backends as training config)."""

from __future__ import annotations


def main() -> None:
    from z_retina.pseudo_patient import main as pseudo_main

    pseudo_main()


if __name__ == "__main__":
    main()
