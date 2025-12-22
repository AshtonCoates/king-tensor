from __future__ import annotations

import argparse
import importlib.metadata
from typing import Sequence


def _run_default(_args: argparse.Namespace) -> None:
    from king_tensor.main import main as run_main

    run_main()


def _run_boxer(args: argparse.Namespace) -> None:
    from king_tensor.utils import boxer

    boxer.main([args.image_path])


def _run_dataclean(extra_args: Sequence[str]) -> None:
    from king_tensor.utils import dataclean

    dataclean.main(list(extra_args))


def _run_calibrate(args: argparse.Namespace) -> None:
    from king_tensor.utils import calibrate

    calibrate.run(
        output_filename=args.output,
        template_filename=args.template,
        full=args.full,
        timer=args.timer,
    )


def _run_help(_args: argparse.Namespace) -> None:
    from king_tensor.utils import help as help_module

    help_module.main()


def _build_parser() -> argparse.ArgumentParser:
    version = "unknown"
    try:
        version = importlib.metadata.version("king-tensor")
    except importlib.metadata.PackageNotFoundError:
        pass

    parser = argparse.ArgumentParser(
        prog="kt",
        description="Command line utilities for the King Tensor toolkit.",
    )
    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version=f"%(prog)s {version}",
        help="Show the installed version and exit.",
    )

    subparsers = parser.add_subparsers(dest="command")
    subparsers.required = False
    parser.set_defaults(func=_run_default)

    run_parser = subparsers.add_parser(
        "run",
        help="Run the main King Tensor workflow (default).",
    )
    run_parser.set_defaults(func=_run_default)

    boxer_parser = subparsers.add_parser(
        "boxer",
        help="Interactively draw a bounding box on an image.",
    )
    boxer_parser.add_argument(
        "image_path",
        help="Path to the image to annotate.",
    )
    boxer_parser.set_defaults(func=_run_boxer)

    dataclean_parser = subparsers.add_parser(
        "dataclean",
        help="Run dataset-cleanup utilities (clean-names, sort, etc.).",
        add_help=False,
    )

    calibrate_parser = subparsers.add_parser(
        "calibrate",
        help="Capture a screenshot and annotate UI bounding boxes.",
    )
    calibrate_parser.add_argument(
        "-o",
        "--output",
        default="ui.yaml",
        help="Output filename written inside the config directory.",
    )
    calibrate_parser.add_argument(
        "--template",
        default="ui.yaml",
        help="Template file within the config directory that lists calibration targets.",
    )
    calibrate_parser.add_argument(
        "-f",
        "--full",
        action="store_true",
        help="Prompt for every UI element instead of only screen.",
    )
    calibrate_parser.add_argument(
        "-t",
        "--timer",
        type=float,
        default=0.0,
        help="Delay before capturing the screenshot (seconds).",
    )
    calibrate_parser.set_defaults(func=_run_calibrate)

    help_parser = subparsers.add_parser(
        "help",
        help="Display the King Tensor helper text.",
    )
    help_parser.set_defaults(func=_run_help)

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = _build_parser()
    args, remainder = parser.parse_known_args(argv)
    if args.command == "dataclean":
        _run_dataclean(remainder)
        return
    if remainder:
        parser.error(f"unrecognized arguments: {' '.join(remainder)}")
    args.func(args)


if __name__ == "__main__":
    main()
