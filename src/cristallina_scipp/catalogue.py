"""Run catalogue: an experiment's run groups, ROIs and geometry in one TOML file.

One file, kept next to the data and shared by everyone, replaces run lists
retyped in each notebook, per-notebook ROI dictionaries and geometry modules.
Only what cannot be read from the data files goes in it: what each group of
runs was for, the ROIs, and the detector calibration. Energy, field and
transmission are read from the runs themselves
([`run_metadata`][cristallina_scipp.series.run_metadata]).

```toml
raw = "/sf/cristallina/data/p2xxxx/raw"   # relative paths are relative to this file

[rois]
"peak A" = { left = 370, right = 500, bottom = 700, top = 760 }

[[geometry]]                              # detector calibration, by run range
runs = "1-780"
x0 = 416.81
y0 = 775.27
d0 = 0.568
z0 = 0.2
stt0 = -0.09935076

[groups."energy 0T"]
runs = "467-486, 525-533"                 # inclusive ranges
roi = "peak A"
sample = "S1"
reflection = "A"
note = "energy scan at 0 T"
```

Any key in a group other than `runs` and `roi` is a label: it becomes a
coordinate of the [`run_series`][cristallina_scipp.series.run_series] result.
`roi` can also map run ranges to ROI names, for peaks that move with field:
`roi = { "781-786" = "600 0T", "787-789" = "600 0.6T" }`.

A catalogue can also be a folder of TOML files, read as one: people working
on different series then edit different files. For keeping it in git during
an experiment, see `cristallina-catalogue setup` and the how-to.
"""

SECTIONS = {"raw", "rois", "geometry", "groups", "samples"}


def _files(path):
    if path.is_dir():
        files = sorted(path.glob("*.toml"))
        if not files:
            raise ValueError(f"{path} holds no .toml files")
        return files
    return [path]


def _merge(files):
    """One document from several catalogue files; a name defined twice, or an unknown section, is an error."""
    doc, origin = {"rois": {}, "groups": {}, "samples": {}, "geometry": []}, {}
    for f in files:
        try:
            part = tomllib.loads(f.read_text())
        except tomllib.TOMLDecodeError as err:
            raise ValueError(f"{f.name}: {err}") from None
        unknown = set(part) - SECTIONS
        if unknown:
            raise ValueError(f"{f.name}: unknown section(s) {sorted(unknown)}; expected {sorted(SECTIONS)}")
        if "raw" in part:
            if "raw" in doc and doc["raw"] != part["raw"]:
                raise ValueError(f"{f.name}: raw = {part['raw']!r}, but {origin['raw']} says {doc['raw']!r}")
            doc["raw"], origin["raw"] = part["raw"], f.name
        for section in ("rois", "groups", "samples"):
            for name, value in part.get(section, {}).items():
                if name in doc[section]:
                    raise ValueError(f"{f.name}: {section[:-1]} {name!r} is also defined in {origin[section, name]}")
                doc[section][name], origin[section, name] = value, f.name
        doc["geometry"].extend(part.get("geometry", []))
    return doc


def _digest(files, single):
    if single:
        return hashlib.sha256(files[0].read_bytes()).hexdigest()
    h = hashlib.sha256()
    for f in files:
        h.update(f.name.encode() + b"\0" + f.read_bytes() + b"\0")
    return h.hexdigest()


def _git_commit(folder):
    """The commit of the git checkout holding `folder`, with "+modified" for uncommitted changes; "" outside git."""
    git = ["git", "-c", "safe.directory=*", "-C", str(folder)]  # a checkout shared by the group is owned by one person
    try:
        commit = subprocess.run([*git, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run([*git, "status", "--porcelain", "--", "."], capture_output=True, text=True,
                               check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""
    return commit + ("+modified" if dirty else "")

import hashlib
import subprocess
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from .load import ROI


def parse_runs(spec) -> list[int]:
    """Run numbers from `"467-486, 525-533, 540"` (ranges inclusive), an int, or a list of either."""
    if isinstance(spec, int):
        return [spec]
    if isinstance(spec, list):
        return [r for item in spec for r in parse_runs(item)]
    runs = []
    for part in str(spec).split(","):
        lo, _, hi = part.strip().partition("-")
        runs.extend(range(int(lo), int(hi or lo) + 1))
    return runs


@dataclass
class Catalogue:
    """The run groups, ROIs and geometry of one experiment.

    Load with [`Catalogue.load`][cristallina_scipp.catalogue.Catalogue.load].
    """

    raw: Path
    rois: dict[str, ROI] = field(default_factory=dict)
    groups: dict[str, dict] = field(default_factory=dict)
    samples: dict[str, dict] = field(default_factory=dict)
    geometry_ranges: list[tuple[set[int], dict]] = field(default_factory=list)
    path: Path | None = None
    sha256: str | None = None  # of the file(s) as loaded, recorded in saved series
    commit: str = ""  # git commit of the catalogue's checkout, "+modified" if it has uncommitted changes

    @classmethod
    def load(cls, path) -> "Catalogue":
        """Read a catalogue: one TOML file, or a folder whose `*.toml` files are read as one.

        Fails, naming the file, on invalid TOML, unknown sections, a name
        defined in two files, run ranges covered by two `[[geometry]]`
        entries, and runs of a group without an ROI.
        """
        path = Path(path).resolve()
        files = _files(path)
        doc = _merge(files)
        folder = path if path.is_dir() else path.parent
        rois = {name: ROI(**limits, name=name) for name, limits in doc["rois"].items()}
        geometry = []
        for entry in doc["geometry"]:
            entry = dict(entry)
            runs = set(parse_runs(entry.pop("runs")))
            for other, _ in geometry:
                if runs & other:
                    raise ValueError(f"runs {sorted(runs & other)[:5]} are in two [[geometry]] entries")
            geometry.append((runs, entry))
        groups = doc["groups"]
        cat = cls(raw=folder / doc.get("raw", "."), rois=rois, groups=groups, samples=doc["samples"],
                  geometry_ranges=geometry,
                  path=path, sha256=_digest(files, not path.is_dir()), commit=_git_commit(folder))
        for name in groups:  # fail on load, not halfway through a series
            for run in cat.runs(name):
                if "roi" in groups[name]:
                    cat.roi(name, run)
        return cat

    def runs(self, group) -> list[int]:
        """Run numbers of a group, in catalogue order."""
        return parse_runs(self.groups[group]["runs"])

    def labels(self, group) -> dict:
        """A group's labels: every key except `runs` and `roi`."""
        return {k: v for k, v in self.groups[group].items() if k not in ("runs", "roi")}

    def roi(self, group, run) -> ROI | None:
        """The ROI for `run` in `group`, or None if the group names none."""
        spec = self.groups[group].get("roi")
        if spec is None:
            return None
        if isinstance(spec, dict):
            names = [name for runs, name in spec.items() if run in parse_runs(runs)]
            if not names:
                raise KeyError(f"group {group!r} has no ROI for run {run}")
            spec = names[0]
        if spec not in self.rois:
            raise KeyError(f"group {group!r} uses ROI {spec!r}, which [rois] does not define")
        return self.rois[spec]

    def geometry(self, run) -> dict:
        """Calibration for `run`: keyword arguments for
        [`geometry_from_step`][cristallina_scipp.angles.geometry_from_step]."""
        for runs, params in self.geometry_ranges:
            if run in runs:
                return dict(params)
        raise KeyError(f"no [[geometry]] entry covers run {run}")

    def scan_path(self, run) -> Path:
        """Path of a run's `meta/scan.json`."""
        return self.raw / f"run{run:04}" / "meta" / "scan.json"

    def scan(self, run):
        """A run's `SFScanInfo`."""
        from sfdata import SFScanInfo

        return SFScanInfo(str(self.scan_path(run)))

    def overview(self, runs=None):
        """[`run_overview`][cristallina_scipp.series.run_overview] of `raw`, with each run's group(s).

        `group` is empty for runs no group lists, which is what to look at
        when bringing the catalogue up to date; a run in two groups shows both,
        separated by "; ". Runs a group lists but `raw` lacks appear with status
        "not in raw".
        """
        from .series import run_overview

        table = run_overview(self.raw, runs)
        groups = {}
        for name in self.groups:
            for run in self.runs(name):
                groups.setdefault(run, []).append(name)
        listed = set(groups) if runs is None else set(groups) & set(runs)
        for run in sorted(listed - set(table.index)):
            table.loc[run, "status"] = "not in raw"
        table["group"] = [("; ".join(groups.get(run, []))) for run in table.index]
        return table.sort_index()

    def series(self, groups, detector=None, i0=None, **kwargs):
        """[`run_series`][cristallina_scipp.series.run_series] over one group or several.

        Runs are taken in catalogue order, with each group's ROI and labels and
        the group name as a `group` label. A run listed in two of the groups is
        an error: it usually means two ranges overlap by one. The result's
        `settings` also record the catalogue file, the SHA-256 of its contents
        when loaded, and the group names, so a saved series says exactly which
        definitions produced it.

        Args:
            groups: A group name or a list of names, e.g. the groups of one
                field-angle series.
            detector, i0: Default to `channels.JF` and `channels.JF_I0`.
            **kwargs: Passed to `run_series` (cutoffs, `photon_energy`, `n_edge`, ...).
        """
        from . import channels
        from .series import run_series

        groups, scans, rois, labels = self._collect(groups)
        out = run_series(scans, detector or channels.JF, channels.JF_I0 if i0 is None else i0,
                         roi=rois, labels=labels, **kwargs)
        return self._field_angles(self._provenance(out, groups))

    def _collect(self, groups):
        groups = [groups] if isinstance(groups, str) else list(groups)
        scans, rois, labels = {}, {}, {}
        for group in groups:
            for run in self.runs(group):
                if run in scans:
                    raise ValueError(f"run {run} is in group {group!r} and in {labels[run]['group']!r}")
                scans[run] = self.scan(run)
                rois[run] = self.roi(group, run)
                labels[run] = {"group": group, **self.labels(group)}
        return groups, scans, rois, labels

    def _provenance(self, out, groups):
        out["settings"]["catalogue"] = str(self.path or "")
        out["settings"]["catalogue_sha256"] = self.sha256 or ""
        out["settings"]["catalogue_commit"] = self.commit
        out["settings"]["groups"] = ", ".join(groups)
        return out

    def sample_of(self, run):
        """The `sample` label of the group(s) listing `run`; an error unless there is exactly one."""
        names = {self.groups[g].get("sample") for g in self.groups if run in self.runs(g)} - {None}
        if len(names) != 1:
            raise ValueError(f"run {run}: its groups name {len(names)} samples; need exactly one `sample` label")
        return names.pop()

    def _field_angles(self, out):
        """Add field_angle and field_tilt when the series' samples give their field axes."""
        from .series import add_field_angles

        runs = out["summary"].coords["run"].values
        names = [{self.groups[g].get("sample") for g in self.groups if run in self.runs(g)} - {None} for run in runs]
        names = [n.pop() if len(n) == 1 else None for n in names]
        specs = [self.samples.get(n, {}) for n in names]
        given = ["field_angle_zero" in s for s in specs]
        if not any(given):
            return out
        if not all(given):
            missing = sorted({str(n) for n, g in zip(names, given) if not g})
            raise ValueError(f"samples {missing} give no field_angle_zero, but others in this series do")
        missing = sorted({str(n) for n, s in zip(names, specs) if "field_angle_90" not in s})
        if missing:
            raise ValueError(f"samples {missing} give field_angle_zero but no field_angle_90")
        return add_field_angles(out, [s["field_angle_zero"] for s in specs], [s["field_angle_90"] for s in specs])

    def sample(self, name):
        """A `cristallina.rsm.Sample` from `[samples."name"]` (needs the `[rsm]` extra).

        The section holds the `Sample` arguments: `lattice` (`{a, b, c, alp,
        bet, gam}`, nm and degrees), `surface_normal` (default `[0, 0, 1]`) and
        the mounting offsets `alpha0`, `ov0`, `chi0` (degrees), plus `offset`,
        added to the rocking readback to give `alpha`. A remounted sample is a
        new entry. Optional `field_angle_zero` and `field_angle_90` are the
        field directions, in the magnet's own x, y, z frame, at which the
        field angle is 0° and +90°; series of the sample's runs then get
        `field_angle` and `field_tilt` coordinates (see
        [`add_field_angles`][cristallina_scipp.series.add_field_angles]).
        """
        import numpy as np
        from cristallina import rsm as crsm

        if name not in self.samples:
            raise KeyError(f"sample {name!r} is not in [samples]")
        spec = self.samples[name]
        return crsm.Sample(dict(spec["lattice"]), name, surface_normal=np.array(spec.get("surface_normal", [0, 0, 1])),
                           alpha0=spec.get("alpha0", 0), ov0=spec.get("ov0", 0), chi0=spec.get("chi0", 0))

    def detector(self, run, size=(1030, 1614)):
        """A `cristallina.rsm.Detector` for `run`, at its arm angle, from `[[geometry]]` and the run's first step.

        Distance, direct-beam pixel and 2θ come from
        [`geometry_from_step`][cristallina_scipp.angles.geometry_from_step]
        with the catalogue's calibration for that run; `size` is the full
        detector in pixels (columns, rows), the Jungfrau 1.5M by default.
        """
        from cristallina import rsm as crsm

        from .angles import geometry_from_step
        from .load import _open

        with _open(self.scan(run), 0) as step:
            g = geometry_from_step(step, **self.geometry(run))
        det = crsm.Detector(R_m=g["distance"], px_size_um=g["pixel_size"] * 1e6, center_pixel=(g["x0"], g["y0"]),
                            detector_size_px=tuple(size))
        det.move_angles(gamma=g["two_theta"])
        return det

    def to_q(self, run, stack, reflection, toward=(0, 0, 1)):
        """Add `h`, `k`, `l` and the local `strain`, `tilt_1`, `tilt_2` around `reflection` to a run's stack.

        The sample is the one named by the `sample` label of the run's group
        (see [`sample`][cristallina_scipp.catalogue.Catalogue.sample]); the
        detector comes from [`detector`][cristallina_scipp.catalogue.Catalogue.detector];
        the energy is the run's monochromator readback.
        """
        from .rsm import local_coords, pixel_hkl
        from .series import run_metadata

        name = self.sample_of(run)
        energy = run_metadata(self.scan(run))["mono_energy"].to(unit="eV").value
        sample = self.sample(name)
        hkl = pixel_hkl(stack, sample, self.detector(run), energy, offset=self.samples[name].get("offset", 0.0))
        return local_coords(hkl, reflection, sample.sample_dict, toward=toward)

    def split_series(self, groups, reflection=None, *, toward=(0, 0, 1), axis="x", bins=None, shape="voigt",
                     detector=None, i0=None, **kwargs):
        """[`split_series`][cristallina_scipp.series.split_series] over catalogue groups.

        Without `reflection`, each run's detector projection along `axis` is
        fitted. With `reflection` (e.g. `(2, 0, 0)`), each run is mapped to
        strain and tilts around it with the catalogue's `[samples]` and
        `[[geometry]]` (see [`to_q`][cristallina_scipp.catalogue.Catalogue.to_q])
        and fitted in reciprocal space. `settings` record the catalogue, as for
        [`series`][cristallina_scipp.catalogue.Catalogue.series], and the reflection.
        """
        from . import channels
        from .series import split_series

        groups, scans, rois, labels = self._collect(groups)
        to_q = None if reflection is None else (lambda run, stack: self.to_q(run, stack, reflection, toward))
        out = split_series(scans, detector or channels.JF, channels.JF_I0 if i0 is None else i0, roi=rois,
                           labels=labels, axis=axis, to_q=to_q, bins=bins, shape=shape, **kwargs)
        if reflection is not None:
            out["settings"]["reflection"] = " ".join(str(i) for i in reflection)
            out["settings"]["toward"] = " ".join(str(i) for i in toward)
        return self._field_angles(self._provenance(out, groups))


def check(path, raw=True) -> tuple[list[str], list[str]]:
    """Errors (the catalogue does not load) and warnings (it loads, but something looks wrong).

    Warnings: runs listed in two groups, and, with `raw`, runs a group lists
    that the raw folder does not have (yet).
    """
    try:
        cat = Catalogue.load(path)
    except (ValueError, KeyError, TypeError) as err:
        return [f"{type(err).__name__}: {err}"], []
    warnings, seen = [], {}
    for group in cat.groups:
        for run in cat.runs(group):
            if run in seen:
                warnings.append(f"run {run} is in groups {seen[run]!r} and {group!r}")
            seen.setdefault(run, group)
    if raw and cat.raw.is_dir():
        missing = sorted(r for r in seen if not (cat.raw / f"run{r:04}").exists())
        if missing:
            warnings.append(f"{len(missing)} listed run(s) not in {cat.raw}: {missing[:10]}{' ...' if len(missing) > 10 else ''}")
    return [], warnings


PRE_RECEIVE = """#!/bin/sh
# Refuse a push whose catalogue does not load (installed by cristallina-catalogue setup).
zero=0000000000000000000000000000000000000000
while read old new ref; do
  [ "$new" = "$zero" ] && continue
  tmp=$(mktemp -d) || exit 1
  git archive "$new" | tar -x -C "$tmp"
  "{python}" -m cristallina_scipp.catalogue check --no-raw "$tmp"
  status=$?
  rm -rf "$tmp"
  [ $status -eq 0 ] || {{ echo "push refused: the catalogue does not load (see above)"; exit 1; }}
done
"""

POST_RECEIVE = """#!/bin/sh
# Update the read-only checkout that notebooks load (installed by cristallina-catalogue setup).
unset GIT_DIR GIT_WORK_TREE
cd "{deploy}" || exit 1
git -c safe.directory='*' pull -q --ff-only origin main
chmod a-w -- *.toml 2>/dev/null
exit 0
"""

STARTER = """# Run catalogue. Edit in your own clone, then commit and push; see the cristallina-scipp how-to.
raw = "../../raw"   # relative to this folder: <pgroup>/work/catalogue -> <pgroup>/raw

[rois]

[groups]
"""


def _git(*args, cwd=None):
    import os

    env = dict(os.environ)
    ident = subprocess.run(["git", "config", "user.email"], capture_output=True, text=True, cwd=cwd).stdout.strip()
    extra = [] if ident else ["-c", f"user.name={env.get('USER', 'catalogue')}", "-c",
                              f"user.email={env.get('USER', 'catalogue')}@localhost"]
    return subprocess.run(["git", "-c", "safe.directory=*", *extra, *args], cwd=cwd, check=True,
                          capture_output=True, text=True).stdout


def setup(work, source=None, python=None) -> Path:
    """Create a shared catalogue repository in `work` and the checkout notebooks load.

    Makes `work/catalogue.git`, a bare repository writable by the group, with
    two hooks: `pre-receive` refuses a push whose catalogue does not load,
    and `post-receive` updates `work/catalogue`, a read-only checkout of
    `main`. Everyone clones the repository, edits, commits and pushes;
    notebooks load `work/catalogue`, and saved series record its commit.

    Args:
        work: The experiment's shared folder, e.g. `/sf/cristallina/data/p2xxxx/work`.
        source: An existing catalogue file or folder to start from; a starter file otherwise.
        python: The interpreter the hook runs the check with; this one by default.

    Returns:
        The path of the checkout to load.
    """
    import os
    import shutil
    import sys
    import tempfile

    work = Path(work).resolve()
    bare, deploy = work / "catalogue.git", work / "catalogue"
    for p in (bare, deploy):
        if p.exists():
            raise FileExistsError(f"{p} exists; refusing to overwrite it")
    _git("init", "-q", "--bare", "--shared=group", "-b", "main", str(bare))
    hooks = {"pre-receive": PRE_RECEIVE.format(python=python or sys.executable),
             "post-receive": POST_RECEIVE.format(deploy=deploy)}
    for name, text in hooks.items():
        (bare / "hooks" / name).write_text(text)
        (bare / "hooks" / name).chmod(0o775)
    with tempfile.TemporaryDirectory() as tmp:
        clone = Path(tmp) / "clone"
        _git("clone", "-q", str(bare), str(clone))
        if source is None:
            (clone / "catalogue.toml").write_text(STARTER)
        else:
            for f in _files(Path(source).resolve()):
                shutil.copy(f, clone / f.name)
        _git("add", ".", cwd=clone)
        _git("commit", "-q", "-m", "Start the run catalogue", cwd=clone)
        _git("push", "-q", "origin", "HEAD:main", cwd=clone)  # the pre-receive hook checks it
    _git("clone", "-q", str(bare), str(deploy))
    _git("config", "core.sharedRepository", "group", cwd=deploy)
    for root, dirs, files in os.walk(deploy):
        for d in dirs:
            os.chmod(Path(root) / d, 0o2775)  # group-writable, new files inherit the group
        for f in files:
            mode = 0o444 if f.endswith(".toml") and Path(root) == deploy else 0o664
            os.chmod(Path(root) / f, mode)
    os.chmod(deploy, 0o2775)
    return deploy


def main(argv=None):
    """`cristallina-catalogue check PATH` or `cristallina-catalogue setup WORK [--from FILE]`."""
    import argparse

    parser = argparse.ArgumentParser(prog="cristallina-catalogue", description="Check or share a run catalogue.")
    sub = parser.add_subparsers(dest="command", required=True)
    p_check = sub.add_parser("check", help="load a catalogue file or folder and report problems")
    p_check.add_argument("path")
    p_check.add_argument("--no-raw", action="store_true", help="do not look for the runs in the raw folder")
    p_setup = sub.add_parser("setup", help="create a shared git repository and read-only checkout in WORK")
    p_setup.add_argument("work")
    p_setup.add_argument("--from", dest="source", help="catalogue file or folder to start from")
    args = parser.parse_args(argv)
    if args.command == "check":
        errors, warnings = check(args.path, raw=not args.no_raw)
        for line in errors:
            print("error:", line)
        for line in warnings:
            print("warning:", line)
        if not errors:
            print(f"ok: {args.path}")
        return 1 if errors else 0
    deploy = setup(args.work, args.source)
    print(f"clone with:  git clone {deploy.parent / 'catalogue.git'}\nload with:   Catalogue.load({str(deploy)!r})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
