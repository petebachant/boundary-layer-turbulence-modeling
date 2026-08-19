"""Script for running the simulation."""

import argparse
import json
import os
import shutil
import subprocess
import sys

import foampy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ny",
        type=int,
        help="How many cells to create in the y-direction.",
        default=40,
    )
    parser.add_argument(
        "--turbulence-model",
        choices=["laminar", "k-epsilon", "new"],
        default="k-epsilon",
    )
    parser.add_argument(
        "--coeffs-json",
        help=(
            "Path to a JSON file with ransFromDns coefficients to write into "
            "turbulenceProperties."
        ),
    )
    parser.add_argument(
        "--case-name",
        help="Override the default case directory name.",
    )
    parser.add_argument(
        "--overwrite", "-f", action="store_true", default=False
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        default=False,
        help="Do not create a new directory for this case.",
    )
    args = parser.parse_args()
    case_name = args.case_name or f"{args.turbulence_model}-ny-{args.ny}"
    if not args.in_place:
        case_dir = os.path.join("cases", case_name)
    else:
        case_dir = "."
    # Copy case files into a case directory, deleting anything that might
    # exist
    # If the case has already been run, we should see it in the results, or
    # maybe we should use DVC to sort this out
    if (
        not args.overwrite
        and not args.in_place
        and os.path.isdir(case_dir)
        and os.listdir(case_dir)
    ):
        print("Case directory is not empty; exiting")
        sys.exit(0)
    if args.overwrite and not args.in_place and os.path.isdir(case_dir):
        # Delete the case and recreate from scratch
        shutil.rmtree(case_dir)
    if not os.path.isdir(case_dir):
        print(f"Creating case directory {case_dir}")
    nx_base = [6, 350, 20, 20, 8]
    ny_base = 40
    nx = [int(x * args.ny / ny_base) for x in nx_base]
    os.makedirs(case_dir, exist_ok=True)
    system_dir = os.path.join(case_dir, "system")
    os.makedirs(system_dir, exist_ok=True)
    blockmeshdict_fpath = os.path.join(system_dir, "blockMeshDict")
    foampy.fill_template(
        "system/blockMeshDict.template", blockmeshdict_fpath, nx=nx, ny=args.ny
    )
    model_names = {
        "k-epsilon": "kEpsilon",
        "laminar": "kEpsilon",
        "new": "ransFromDns",
    }
    coeffs = {
        "Cmu": 0.09,
        "C1": 1.44,
        "C2": 1.92,
        "C3": 0.0,
        "sigmak": 1.0,
        "sigmaEps": 1.3,
        "f1ProductionK": 1.0,
        "f2DissipationK": 1.0,
        "f1ProductionEps": 1.0,
        "f2DissipationEps": 1.0,
    }
    if args.coeffs_json:
        with open(args.coeffs_json, "r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        for key in coeffs:
            if key in loaded:
                coeffs[key] = float(loaded[key])
    constant_dir = os.path.join(case_dir, "constant")
    os.makedirs(constant_dir, exist_ok=True)
    is_ras = args.turbulence_model in {"k-epsilon", "new"}
    foampy.fill_template(
        "constant/turbulenceProperties.template",
        os.path.join(constant_dir, "turbulenceProperties"),
        turbulence_model=model_names[args.turbulence_model],
        turbulence_on="on" if is_ras else "off",
        simulation_type="RAS" if is_ras else "laminar",
        **coeffs,
    )
    if not args.in_place:
        shutil.copytree("0", os.path.join(case_dir, "0"))
        # All other non template files to copy over
        paths = [
            "constant/transportProperties",
            "system/controlDict",
            "system/fvSchemes",
            "system/fvSolution",
            "system/sample",
        ]
        for path in paths:
            shutil.copy(path, os.path.join(case_dir, path))
    # Move into the case directory
    print(f"Changing working directory to {case_dir}")
    os.chdir(case_dir)
    # Create the mesh
    foampy.run("blockMesh", overwrite=args.overwrite)
    # Run simpleFoam
    foampy.run(
        (
            "ransFromDnsSimpleFoam"
            if args.turbulence_model == "new"
            else "simpleFoam"
        ),
        overwrite=args.overwrite,
    )
    # Post-process
    foampy.run(
        "postProcess",
        args=["-latestTime", "-func", "sample"],
        overwrite=args.overwrite,
    )
    # Touch case.foam file so we can easily open with ParaView
    subprocess.call(["touch", "case.foam"])
