#!/usr/bin/env python
"""Run Notebook B's immutable MRI projected-reference validation workflow.

This driver never accesses the original MRI NetCDF and has no dependency on
Skyborn or SPHEREPACK.  It consumes exactly the three hash-pinned files under
``--reference-dir`` and delegates all Aeolus numerics to the canonical W5
runner/model/backend APIs.
"""
from __future__ import annotations

import argparse
import pathlib
import sys


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--reference-dir", required=True,
        help="directory containing manifest.json, t42_64x128.npz, and "
             "t63_96x192.npz")
    parser.add_argument(
        "--output-root", required=True,
        help="validation root (typically on mounted Google Drive)")
    parser.add_argument(
        "--repository-root", default=".",
        help="checked-out Aeolus repository used for provenance (default: .)")
    parser.add_argument(
        "--repository-url",
        default="https://github.com/AlexandreEros/Aeolus.git")
    parser.add_argument(
        "--git-ref", default="feat/w5-mri-validation",
        help="requested ref recorded in validation provenance")
    parser.add_argument(
        "--skip-t42", action="store_true",
        help="do not execute/reuse the T42 stage")
    parser.add_argument(
        "--skip-t63", action="store_true",
        help="do not execute/reuse the T63 stage")
    parser.add_argument(
        "--force-rerun", action="store_true",
        help="preserve the existing stage under a superseded name and run a "
             "fresh stage; never delete prior results")
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    from planetary_sandbox.validation.w5_mri_workflow import (
        WorkflowConfig, run_validation_workflow)

    config = WorkflowConfig(
        reference_dir=pathlib.Path(args.reference_dir).expanduser(),
        output_root=pathlib.Path(args.output_root).expanduser(),
        repository_root=pathlib.Path(args.repository_root).expanduser(),
        repository_url=args.repository_url,
        git_ref=args.git_ref,
        run_t42=not args.skip_t42,
        run_t63=not args.skip_t63,
        force_rerun=args.force_rerun,
    )
    run_validation_workflow(config)
    return 0


if __name__ == "__main__":
    sys.exit(main())

