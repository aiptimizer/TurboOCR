"""Packaging invariants across the four mutually-exclusive engine wheels.

turboocr-engine-{cpu,cuda,openvino,rocm} are FOUR distributions built from ONE
source tree: same `turboocr_engine` package, same `_turboocr` extension,
different name + CMake args (python/wheels/README.md). Everything user-visible
that is not the accelerator therefore has to stay identical between them, and
nothing enforces that — the per-backend pyproject.toml files are hand-maintained
copies. The pure-Python `turboocr` umbrella (python-sdk/) is checked at the
end: its extras are what normally install these wheels, so its pins must track
the engine version.

These are pure-Python (tomllib + the pyproject files); no build, no network.
"""

from __future__ import annotations

import os
import tomllib

import pytest

PY_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: dist name -> its pyproject. The BASE wheel first: it is the reference the
#: others are compared against.
PYPROJECTS = {
    "turboocr-engine-cpu": os.path.join(PY_DIR, "pyproject.toml"),
    # NVIDIA is two distributions, one per CUDA major (see wheels/README.md).
    "turboocr-engine-cuda12": os.path.join(PY_DIR, "wheels", "cuda12", "pyproject.toml"),
    "turboocr-engine-cuda13": os.path.join(PY_DIR, "wheels", "cuda13", "pyproject.toml"),
    "turboocr-engine-openvino": os.path.join(PY_DIR, "wheels", "openvino", "pyproject.toml"),
    "turboocr-engine-rocm": os.path.join(PY_DIR, "wheels", "rocm", "pyproject.toml"),
}

BASE = "turboocr-engine-cpu"
UMBRELLA_PYPROJECT = os.path.join(os.path.dirname(PY_DIR), "python-sdk", "pyproject.toml")


def _load(path: str) -> dict:
    with open(path, "rb") as fh:
        return tomllib.load(fh)


@pytest.fixture(scope="module")
def projects() -> dict:
    missing = [p for p in PYPROJECTS.values() if not os.path.exists(p)]
    if missing:
        pytest.skip(f"pyproject not present: {missing}")
    return {name: _load(path)["project"] for name, path in PYPROJECTS.items()}


def test_the_four_pyprojects_declare_the_four_names(projects):
    """A copied file that kept the name it was copied from would build two
    wheels with the same distribution name and one would silently overwrite the
    other on the index."""
    assert {p["name"] for p in projects.values()} == set(PYPROJECTS)
    for name, proj in projects.items():
        assert proj["name"] == name, (name, proj["name"])


def _extras(proj: dict) -> dict:
    # Normalized so ordering inside a table can't fail the comparison — the
    # requirement SET is what pip resolves, the list order is cosmetic.
    return {k: sorted(v) for k, v in proj.get("optional-dependencies", {}).items()}


def test_optional_dependencies_are_identical_across_wheels(projects):
    """`pip install "turboocr-engine-cuda12[pdf]"` must pull exactly what
    `pip install "turboocr-engine-cpu[pdf]"` does.

    The extras tables are four hand-copied duplicates: adding a dependency to
    `pdf` in the base pyproject and forgetting the other three gives an NVIDIA
    user a `[pdf]` install that silently lacks it, and the failure surfaces as
    an ImportError deep inside read_pdf(). Compare against the base wheel."""
    base = _extras(projects[BASE])
    assert base, "the base wheel declares no extras — the test would be vacuous"
    for name, proj in projects.items():
        if name == BASE:
            continue
        assert _extras(proj) == base, (
            f"{name}'s [project.optional-dependencies] drifted from {BASE}'s:\n"
            f"  {name}: {_extras(proj)}\n  {BASE}: {base}"
        )


def test_all_extra_is_the_union_of_the_others(projects):
    """`all` is a hand-written concatenation of the other extras, so it is the
    one most likely to be left behind when a real extra changes."""
    for name, proj in projects.items():
        extras = _extras(proj)
        union = sorted(
            {req for key, reqs in extras.items() if key != "all" for req in reqs}
        )
        assert extras.get("all") == union, (name, extras.get("all"), union)


def test_runtime_dependencies_and_entry_point_match(projects):
    """Same reasoning as the extras: the accelerator is the ONLY difference
    between these wheels, so a diverging install_requires or console script is a
    copy-paste slip, not a design choice."""
    base = projects[BASE]
    for name, proj in projects.items():
        assert sorted(proj["dependencies"]) == sorted(base["dependencies"]), name
        assert proj["scripts"] == base["scripts"], name
        assert proj["requires-python"] == base["requires-python"], name


def test_no_wheel_declares_a_turboocr_sibling_as_a_dependency(projects):
    """They are MUTUALLY EXCLUSIVE — all four own the `turboocr_engine` import
    package. A dependency from one onto another (or on the `turboocr` umbrella,
    which would recurse) would make pip install both and leave whichever landed
    last in control of the import name."""
    for name, proj in projects.items():
        reqs = list(proj["dependencies"]) + [
            r for reqs in proj.get("optional-dependencies", {}).values() for r in reqs
        ]
        for req in reqs:
            assert not req.lower().startswith("turboocr"), (name, req)


def test_umbrella_extras_pin_the_engine_wheels_to_its_own_version():
    """python-sdk's extras are the front door to these engine wheels. Each must
    pin `turboocr-engine-<variant>==<umbrella version>` — an unpinned or drifted
    pin installs an engine whose API the umbrella facade was not written
    against.

    NVIDIA is two distributions (one per CUDA major), so `cuda12` and `cuda13`
    are the real extras and bare `cuda` is an ALIAS for whichever is the
    low-friction default. The alias is checked separately: it must still pin at
    the umbrella's own version, and must point at a real NVIDIA extra rather
    than a name nobody publishes."""
    if not os.path.exists(UMBRELLA_PYPROJECT):
        pytest.skip("python-sdk not present (engine-only checkout)")
    proj = _load(UMBRELLA_PYPROJECT)["project"]
    version = proj["version"]
    extras = proj["optional-dependencies"]
    for variant in ("cpu", "cuda12", "cuda13", "openvino", "rocm"):
        assert extras.get(variant) == [f"turboocr-engine-{variant}=={version}"], (
            variant,
            extras.get(variant),
        )
    # The alias must resolve to one of the two NVIDIA wheels, at the same
    # version — not to the retired `turboocr-engine-cuda`, which is not built
    # or published any more.
    assert extras.get("cuda") in (
        [f"turboocr-engine-cuda12=={version}"],
        [f"turboocr-engine-cuda13=={version}"],
    ), extras.get("cuda")
