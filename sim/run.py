"""Script for running the simulation."""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys

import foampy

def grading_cell_centres(height, ncells, ratio):
    """Cell-centre y positions for a blockMesh simpleGrading column.

    ratio is last-cell/first-cell, so the common ratio is ratio^(1/(n-1)).
    """
    import numpy as np
    if abs(ratio - 1.0) < 1e-9:
        edges = np.linspace(0.0, height, ncells + 1)
    else:
        q = ratio ** (1.0 / (ncells - 1))
        h0 = height * (q - 1.0) / (q ** ncells - 1.0)
        sizes = h0 * q ** np.arange(ncells)
        edges = np.concatenate(([0.0], np.cumsum(sizes)))
    return 0.5 * (edges[:-1] + edges[1:])


def seed_laminar_kinetic_energy(case_dir):
    """Split the inlet's measured fluctuation energy between kkLOmega's kt
    and kl.

    The DNS inlet is already streaky: inside the layer its fluctuation
    energy is far above the free stream's, and that is energy in
    pre-transitional streaks, which kkLOmega carries as kl. Handing all of
    it to kt, as the plain run does, puts it in the variable the model
    damps near the wall and leaves kl to grow from nothing. Here kt is the
    free-stream value, capped there, and kl is the excess above it.
    """
    import re

    import numpy as np

    kt_path = os.path.join(case_dir, "0", "kt")
    kl_path = os.path.join(case_dir, "0", "kl")
    pattern = re.compile(
        r"(inlet\s*\{[^}]*?value\s+)nonuniform List<scalar>\s*(\d+)\s*"
        r"\((.*?)\)", re.S)
    with open(kt_path, "r", encoding="utf-8") as handle:
        kt_text = handle.read()
    m = pattern.search(kt_text)
    if m is None:
        raise ValueError("kt has no nonuniform inlet profile to split")
    k = np.array(m.group(3).split(), dtype=float)
    k_fs = float(k[-1])
    kt = np.minimum(k, k_fs)
    kl = np.maximum(k - k_fs, 0.0)
    print(f"  seeding kl at the inlet: peak {kl.max():.3e}, "
          f"kt capped at the free-stream {k_fs:.3e}")

    def as_list(vals):
        body = "\n".join(f"{v:.8g}" for v in vals)
        return f"nonuniform List<scalar> {len(vals)}\n(\n{body}\n)"

    kt_text = kt_text[: m.start()] + m.group(1) + as_list(kt) + \
        kt_text[m.end():]
    with open(kt_path, "w", encoding="utf-8") as handle:
        handle.write(kt_text)
    with open(kl_path, "r", encoding="utf-8") as handle:
        kl_text = handle.read()
    i = kl_text.index("inlet")
    j = kl_text.index("}", i)
    block = "\n".join(
        (line[: len(line) - len(line.lstrip())]
         + f"value           {as_list(kl)};")
        if line.strip().startswith("value") else line
        for line in kl_text[i:j].splitlines())
    with open(kl_path, "w", encoding="utf-8") as handle:
        handle.write(kl_text[:i] + block + kl_text[j:])


def write_dns_inlet(case_dir, prof, ny, ygrad, beta=None,
                    init_internal=False):
    """Overwrite the inlet patch of 0/U, 0/k and 0/omega with DNS profiles.

    With init_internal, the initial k and omega inside the domain are set to
    the inlet's free-stream values too, rather than left at the template's.

    blockMesh orders the faces of a single-block inlet patch by increasing y,
    which is what the nonuniform lists below assume. The assumption is checked
    after the run by sampling the inlet station and comparing with the DNS, so
    a wrong ordering would show up as a large error at the first station rather
    than passing silently.
    """
    import numpy as np
    import numpy as np
    omega_inlet = prof["omega_inlet"]
    table = prof.get("omega_inlet_by_beta") or {}
    if beta is not None and table:
        # Nearest fitted beta; the table is dense enough that interpolation
        # would add precision the fit does not have.
        key = min(table, key=lambda b: abs(float(b) - beta))
        omega_inlet = table[key]["omega_inlet"]
        print(f"  inlet omega {omega_inlet:.4f} for beta={beta:.4f} "
              f"(table entry {key})")
    yc = grading_cell_centres(prof["y_max"], ny, ygrad)
    yp = np.array(prof["y"])
    Up = np.interp(yc, yp, np.array(prof["U"]))
    kp = np.interp(yc, yp, np.array(prof["k"]))

    def as_list(vals, fmt="{:.8g}"):
        body = "\n".join(fmt.format(v) for v in vals)
        return f"nonuniform List<scalar> {len(vals)}\n(\n{body}\n)"

    def as_vlist(vals):
        body = "\n".join(f"({v:.8g} 0 0)" for v in vals)
        return f"nonuniform List<vector> {len(vals)}\n(\n{body}\n)"

    replacements = {
        "U": as_vlist(Up),
        "k": as_list(kp),
        "omega": f"uniform {omega_inlet:.8g}",
    }
    if "ReThetat_inlet" in prof:
        replacements["ReThetat"] = (
            f"uniform {prof['ReThetat_inlet']:.8g}")
    for fname, value in replacements.items():
        path = os.path.join(case_dir, "0", fname)
        if not os.path.isfile(path):
            continue
        with open(path, "r", encoding="utf-8") as handle:
            text = handle.read()
        i = text.index("inlet")
        j = text.index("}", i)
        block = text[i:j]
        # Replace the value entry inside the inlet block only
        lines = []
        for line in block.splitlines():
            if line.strip().startswith("value"):
                indent = line[: len(line) - len(line.lstrip())]
                lines.append(f"{indent}value           {value};")
            else:
                lines.append(line)
        text = text[:i] + "\n".join(lines) + text[j:]
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(text)
    if init_internal:
        # A steady run that stops on its residuals can stop before an
        # initial field far from the inlet's is washed out: on Wu et al.'s
        # flows, started from the JHTDB run-up's omega, 400 times the inlet
        # value, the free stream stayed at 25 times the inlet omega and k was
        # destroyed within the first tenth of the plate
        for fname, value in (("k", float(np.max(kp))),
                             ("omega", float(omega_inlet))):
            path = os.path.join(case_dir, "0", fname)
            with open(path, "r", encoding="utf-8") as handle:
                text = handle.read()
            text, n = re.subn(r"internalField\s+uniform\s+[-\d.eE+]+;",
                              f"internalField   uniform {value:.8g};", text)
            assert n == 1, f"0/{fname} has no single uniform internalField"
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(text)
    if "ReThetat_inlet" in prof:
        path = os.path.join(case_dir, "0", "ReThetat")
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as handle:
                text = handle.read()
            text = re.sub(r"internalField\s+uniform\s+[-\d.eE+]+;",
                          f"internalField   uniform "
                          f"{prof['ReThetat_inlet']:.8g};", text)
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(text)
    print(f"Wrote DNS inlet profiles onto {ny} faces "
          f"(Ue={max(Up):.4f}, Tu={100*np.sqrt(2*max(kp)/3)/max(Up):.2f}%)")


def write_top_velocity(case_dir, path):
    """Impose a DNS's velocity along the top of the domain.

    With a zero-gradient top the displaced boundary layer accelerates the
    free stream, where the JHTDB DNS's decelerates, so the model sees a mildly
    favorable pressure gradient ahead of onset where the DNS sees an adverse
    one, and transitions late. Imposing the DNS's own top velocity, both
    components so the displacement can leave through the top, gives it the
    DNS's pressure gradient. The table is (x, (U V 0)) for a fixedProfile
    condition along x.
    """
    with open(path, "r", encoding="utf-8") as handle:
        top = json.load(handle)
    fpath = os.path.join(case_dir, "0", "U")
    with open(fpath, "r", encoding="utf-8") as handle:
        text = handle.read()
    i = text.index("upperWall")
    j = text.index("{", i)
    depth, k = 0, j
    while True:
        if text[k] == "{":
            depth += 1
        elif text[k] == "}":
            depth -= 1
            if depth == 0:
                break
        k += 1
    rows = "\n".join(f"            ({x:.8g} ({u:.8g} {v:.8g} 0))"
                     for x, u, v in zip(top["x"], top["U"], top["V"]))
    block = ("{\n        type            fixedProfile;\n"
             "        profile         table\n        (\n" + rows +
             "\n        );\n        direction       (1 0 0);\n"
             "        origin          0;\n    }")
    with open(fpath, "w", encoding="utf-8") as handle:
        handle.write(text[:j] + block + text[k + 1:])
    print(f"Top velocity imposed from {path} at {len(top['x'])} points")


def write_sample_dict(case_dir, prof, n):
    """Sample n stations evenly over the domain, each a line from the wall
    to the top of the domain, in the plate's sample format."""
    import numpy as np
    x0, x1, ym = prof["x_inlet"], prof["x_outlet"], prof["y_max"]
    xs = x0 + (x1 - x0) * (np.arange(n) + 0.5) / n
    sets = "\n".join(
        f"    x{x:.0f}\n    {{\n        type uniform;\n        axis y;\n"
        f"        start ({x:.6g} 0 0);\n        end ({x:.6g} {ym:.6g} 0);\n"
        f"        nPoints 2000;\n    }}"
        for x in xs)
    text = (
        "FoamFile\n{\n    version 2.0;\n    format ascii;\n"
        "    class dictionary;\n    object sample;\n}\n\n"
        "type sets;\nlibs (sampling);\nwriteControl onEnd;\n"
        "setFormat csv;\nfields (U p k omega gamma nut);\n"
        "interpolationScheme cellPoint;\n\nsets\n{\n" + sets + "\n}\n")
    with open(os.path.join(case_dir, "system", "sample"), "w",
              encoding="utf-8") as handle:
        handle.write(text)


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
        choices=["laminar", "k-epsilon", "new", "clip-k-gamma",
                 "k-omega-sst", "k-omega-sst-lm", "kkl-omega",
                 "kkl-omega-seeded"],
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
        "--y-grading",
        type=float,
        default=8.0,
        help=(
            "Ratio of the largest to smallest cell in y. The default of 8 "
            "leaves the first cell at y+ ~ 30, which is fine for a "
            "wall-function k-epsilon run but far too coarse for a "
            "wall-resolved transition model; use ~2000 for y+ < 1."
        ),
    )
    parser.add_argument(
        "--dns-domain",
        action="store_true",
        default=False,
        help=(
            "Use the domain matched to the DNS: inlet at the DNS inlet "
            "station with the measured inlet profile, rather than 120 units "
            "of run-up ahead of an elliptical leading edge. The run-up made "
            "the measured free-stream decay impossible to impose and forced "
            "a 20 percent inlet turbulence intensity, which is why the "
            "transition baselines tripped at the leading edge."
        ),
    )
    parser.add_argument(
        "--inlet-json",
        default=os.path.join("..", "results", "inlet-profiles.json"),
        help="Inlet and domain for --dns-domain (the JHTDB plate by default).",
    )
    parser.add_argument(
        "--inlet-key",
        help="Key of the flow to use when --inlet-json holds several.",
    )
    parser.add_argument(
        "--nx", type=int,
        help="Cells in x for --dns-domain; the plate's count by default.",
    )
    parser.add_argument(
        "--x-grading", type=float,
        help=(
            "Ratio of the last to the first cell in x for --dns-domain. Long "
            "plates need it, since the layer thickens tenfold or more."
        ),
    )
    parser.add_argument(
        "--init-from-inlet", action="store_true", default=False,
        help=(
            "Start k and omega inside the domain at the inlet's free-stream "
            "values rather than the template's."
        ),
    )
    parser.add_argument(
        "--top-velocity",
        help=(
            "JSON table of a DNS's velocity along the top of the domain to "
            "impose there, instead of a zero-gradient top."
        ),
    )
    parser.add_argument(
        "--end-time", type=int,
        help="Solver iterations, overriding the controlDict.",
    )
    parser.add_argument(
        "--sample-stations", type=int, default=0,
        help=(
            "Write a sample dict with this many evenly spaced stations over "
            "the domain instead of copying the plate's."
        ),
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
    inlet_profiles = None
    if args.dns_domain:
        with open(args.inlet_json, "r", encoding="utf-8") as handle:
            inlet_profiles = json.load(handle)
        if args.inlet_key:
            inlet_profiles = inlet_profiles[args.inlet_key]
        # One block spanning exactly the DNS domain
        nx_dns = args.nx or int(round(700 * args.ny / 80))
        extra = {}
        template = "system/blockMeshDict-dns.template"
        if args.x_grading:
            template = "system/blockMeshDict-graded.template"
            extra["xgrad"] = args.x_grading
        foampy.fill_template(
            template,
            blockmeshdict_fpath,
            x_min=inlet_profiles["x_inlet"],
            x_max=inlet_profiles["x_outlet"],
            y_max=inlet_profiles["y_max"],
            nx=nx_dns,
            ny=args.ny,
            ygrad=args.y_grading,
            **extra,
        )
    else:
        foampy.fill_template(
            "system/blockMeshDict.template",
            blockmeshdict_fpath,
            nx=nx,
            ny=args.ny,
            ygrad=args.y_grading,
            ygrad_inv=1.0 / args.y_grading,
        )
    model_names = {
        "k-epsilon": "kEpsilon",
        "laminar": "kEpsilon",
        "new": "ransFromDns",
        "clip-k-gamma": "clipKGamma",
        # State-of-the-art baselines. kOmegaSST is the standard workhorse for
        # attached boundary layers; kOmegaSSTLM is the Langtry-Menter
        # gamma-Re_theta transition model, which is what this closure actually
        # has to beat; kkLOmega is Walters-Cokljat, whose laminar kinetic
        # energy is the closest published relative of our streak reservoir.
        "k-omega-sst": "kOmegaSST",
        "k-omega-sst-lm": "kOmegaSSTLM",
        "kkl-omega": "kkLOmega",
        # The same model with the inlet's fluctuation energy above the
        # free-stream level handed to the laminar kinetic energy, which is
        # what the DNS's pre-transitional streaks are, rather than to kt
        "kkl-omega-seeded": "kkLOmega",
    }
    # Models resolved to the wall, which need the low-Re field set rather than
    # the wall-function one
    WALL_RESOLVED = {"clip-k-gamma", "k-omega-sst", "k-omega-sst-lm",
                     "kkl-omega", "kkl-omega-seeded"}
    # Extra 0/ fields each model needs beyond k, omega, nut, U, p
    EXTRA_FIELDS = {
        "k-omega-sst-lm": ["gammaInt", "ReThetat"],
        "kkl-omega": ["kl"],
        "kkl-omega-seeded": ["kl"],
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
        # clipKGamma
        "alphaOmega": 0.52,
        "beta": 0.072,
        "betaStar": 0.09,
        "CL": 0.03,
        "Cgam": 0.6,
        "LambdaC": 440.0,
        "pExp": 1.0,
        "Cnu": 2.0,
        "Cs": 0.30,
        "a1": 0.0,
        "c1": 10.0,
        "Cd": 0.0,
        # How the strain-based omega production is gated: none, gamma or
        # exact. See clipKGamma.H. This is a word, not a number.
        "omegaGating": "none",
        "gseedOmega": 0.02,
        # Lift-up viscosity: form (mixing|komega) and whether it is
        # restricted to the un-activated flow. Both are words to OpenFOAM.
        "liftupForm": "mixing",
        "liftupGate": "false",
        "gammaFs": 0.02,
        "gseed": 0.01,
        "sigmak_ko": 2.0,
        "sigmaOmega": 2.0,
        "sigmaGamma": 1.0,
    }
    if args.coeffs_json:
        with open(args.coeffs_json, "r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        # Fitted coefficient files may nest the OpenFOAM-named values
        if "openfoam_coeffs" in loaded:
            loaded = loaded["openfoam_coeffs"]
        for key in coeffs:
            if key in loaded:
                # Most coefficients are numbers, but omegaGating is a word.
                # Coercing it to float would raise, so keep strings as they
                # are rather than silently dropping the structural choice.
                value = loaded[key]
                coeffs[key] = (value if isinstance(value, str)
                               else float(value))
    constant_dir = os.path.join(case_dir, "constant")
    os.makedirs(constant_dir, exist_ok=True)
    is_ras = args.turbulence_model in (
        {"k-epsilon", "new", "clip-k-gamma"} | WALL_RESOLVED)
    foampy.fill_template(
        "constant/turbulenceProperties.template",
        os.path.join(constant_dir, "turbulenceProperties"),
        turbulence_model=model_names[args.turbulence_model],
        turbulence_on="on" if is_ras else "off",
        simulation_type="RAS" if is_ras else "laminar",
        **coeffs,
    )
    if not args.in_place:
        # Wall-resolved models need low-Reynolds-number wall treatment
        # rather than the wall functions the k-epsilon cases use.
        zero_dir = ("fields-low-re" if args.turbulence_model in WALL_RESOLVED
                    else "0")
        shutil.copytree(zero_dir, os.path.join(case_dir, "0"),
                        ignore=shutil.ignore_patterns("*.template"))
        # Free-stream turbulence is an inflow property of the case, not a
        # model coefficient. Its level AND decay rate drive bypass transition,
        # so the inlet k and omega are set from the same decay law the model
        # was fitted under rather than left at arbitrary defaults.
        # Every wall-resolved model gets inlet k and omega fitted to the SAME
        # measured DNS free-stream decay. Leaving the baselines at arbitrary
        # defaults while ours are fitted would be a rigged comparison: bypass
        # transition is driven by the free-stream turbulence, so a transition
        # model handed the wrong intensity fails for a reason that has nothing
        # to do with its closure.
        fs = None
        if args.turbulence_model == "clip-k-gamma":
            fs_path = os.path.join("..", "results",
                                   "clip-k-gamma-coeffs.json")
            if os.path.isfile(fs_path):
                with open(fs_path, "r", encoding="utf-8") as handle:
                    fs = json.load(handle).get("freestream_inlet")
        elif args.turbulence_model in WALL_RESOLVED:
            fs_path = os.path.join("..", "results",
                                   "baseline-freestream-bcs.json")
            if os.path.isfile(fs_path):
                with open(fs_path, "r", encoding="utf-8") as handle:
                    fs = json.load(handle)["models"].get(
                        model_names[args.turbulence_model])
        if fs:
            print(f"Inlet free-stream: k={fs['k_inlet']:.4e} "
                  f"omega={fs['omega_inlet']:.4f}")
            for fname, tmpl_key in (("k", "k_inlet"),
                                    ("omega", "omega_inlet")):
                tmpl = os.path.join(zero_dir, f"{fname}.template")
                if os.path.isfile(tmpl):
                    foampy.fill_template(
                        tmpl, os.path.join(case_dir, "0", fname),
                        **{tmpl_key: fs[tmpl_key]})
        # Inlet profiles taken straight from the DNS station, so every model
        # is handed the same measured inflow instead of developing its own.
        if inlet_profiles is not None:
            # Each model gets the inlet omega consistent with its OWN
            # free-stream destruction coefficient, so all of them start from
            # the same measured decay rather than the same number.
            model_beta = {"k-omega-sst": 0.0828, "k-omega-sst-lm": 0.0828,
                          # kkLOmega's own free stream: Cw2 = 0.92 with
                          # betaStar = 1, not a k-omega model's constants
                          "kkl-omega": 0.92,
                          "kkl-omega-seeded": 0.92}.get(args.turbulence_model)
            if args.turbulence_model == "clip-k-gamma":
                model_beta = float(coeffs.get("beta", 0.0828))
            write_dns_inlet(case_dir, inlet_profiles, args.ny,
                            args.y_grading, beta=model_beta,
                            init_internal=args.init_from_inlet)

        # kkLOmega calls its turbulent energy kt, not k, and carries a
        # separate laminar kinetic energy kl.
        if args.turbulence_model in ("kkl-omega", "kkl-omega-seeded"):
            k_path = os.path.join(case_dir, "0", "k")
            kt_path = os.path.join(case_dir, "0", "kt")
            with open(k_path, "r", encoding="utf-8") as handle:
                kt_text = handle.read().replace("object      k;",
                                                "object      kt;")
            with open(kt_path, "w", encoding="utf-8") as handle:
                handle.write(kt_text)
            if (args.turbulence_model == "kkl-omega-seeded"
                    and inlet_profiles is not None):
                seed_laminar_kinetic_energy(case_dir)
        # Drop 0/ fields a model does not use, so an unused field cannot be
        # mistaken for part of the solution
        own = set(EXTRA_FIELDS.get(args.turbulence_model, []))
        for other, extras in EXTRA_FIELDS.items():
            if other == args.turbulence_model:
                continue
            for fname in extras:
                if fname in own:
                    continue
                stale = os.path.join(case_dir, "0", fname)
                if os.path.isfile(stale):
                    os.remove(stale)
        # All other non template files to copy over
        paths = [
            "constant/transportProperties",
            "system/controlDict",
            "system/fvSchemes",
            "system/fvSolution",
            "system/sample",
        ]
        for path in paths:
            if path == "system/sample" and args.sample_stations:
                continue
            shutil.copy(path, os.path.join(case_dir, path))
        if args.top_velocity:
            write_top_velocity(case_dir, args.top_velocity)
        if args.sample_stations and inlet_profiles is not None:
            write_sample_dict(case_dir, inlet_profiles, args.sample_stations)
        # A flow in other units carries its own viscosity
        if inlet_profiles is not None and "nu" in inlet_profiles:
            tp = os.path.join(case_dir, "constant", "transportProperties")
            with open(tp, "r", encoding="utf-8") as handle:
                text = handle.read()
            text, n = re.subn(r"\bnu\s+[^;]+;",
                              f"nu              {inlet_profiles['nu']:.10g};",
                              text)
            assert n == 1, "transportProperties has no single nu entry"
            with open(tp, "w", encoding="utf-8") as handle:
                handle.write(text)
        if args.end_time:
            cd = os.path.join(case_dir, "system", "controlDict")
            with open(cd, "r", encoding="utf-8") as handle:
                text = handle.read()
            text, n = re.subn(r"\bendTime\s+\d+;",
                              f"endTime         {args.end_time};", text)
            assert n == 1, "controlDict has no single endTime entry"
            with open(cd, "w", encoding="utf-8") as handle:
                handle.write(text)
    # Move into the case directory
    print(f"Changing working directory to {case_dir}")
    os.chdir(case_dir)
    # Create the mesh
    foampy.run("blockMesh", overwrite=args.overwrite)
    # Run simpleFoam
    foampy.run(
        # clipKGamma is a plain library RAS model, so it runs under
        # simpleFoam. ransFromDnsSimpleFoam carries extra hard-coded
        # momentum source terms and must not be used for model comparison.
        ("ransFromDnsSimpleFoam" if args.turbulence_model == "new"
         else "simpleFoam"),
        overwrite=args.overwrite,
    )
    # Post-process
    foampy.run(
        "postProcess",
        args=["-latestTime", "-func", "sample"],
        overwrite=args.overwrite,
    )

    # Fail loudly if the solver did not actually run.
    #
    # foampy.run does not propagate the solver's exit status, so a case that
    # dies on the first iteration -- a missing fvSchemes entry, say -- still
    # leaves the stage looking successful, and DVC records an empty
    # postProcessing directory as a valid output. A baseline that failed this
    # way would enter the model comparison and lose for a reason that has
    # nothing to do with its physics, which is precisely the outcome the
    # comparison exists to rule out.
    solver_log = "log.ransFromDnsSimpleFoam" if args.turbulence_model == "new" \
        else "log.simpleFoam"
    if os.path.isfile(solver_log):
        with open(solver_log, "r", encoding="utf-8", errors="replace") as fh:
            tail = fh.read()
        if "FOAM FATAL" in tail:
            first = tail[tail.index("FOAM FATAL"):].splitlines()
            raise SystemExit(
                "Solver failed:\n  " + "\n  ".join(first[:6]))
        if not tail.rstrip().endswith("End"):
            raise SystemExit(
                f"Solver did not reach 'End'; see {solver_log}")
    sample_root = os.path.join("postProcessing", "sample")
    csvs = []
    for root, _dirs, files in os.walk(sample_root):
        csvs.extend(f for f in files if f.endswith(".csv"))
    if not csvs:
        raise SystemExit(
            f"No sampled profiles written under {sample_root}; the case "
            "produced no usable output.")
    print(f"Solver finished; {len(csvs)} sampled profiles written.")
    # Touch case.foam file so we can easily open with ParaView
    subprocess.call(["touch", "case.foam"])
