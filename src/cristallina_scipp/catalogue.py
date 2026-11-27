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
"""

import hashlib
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
    geometry_ranges: list[tuple[set[int], dict]] = field(default_factory=list)
    path: Path | None = None
    sha256: str | None = None  # of the file as loaded, recorded in saved series

    @classmethod
    def load(cls, path) -> "Catalogue":
        """Read a catalogue TOML file."""
        path = Path(path).resolve()
        text = path.read_text()
        doc = tomllib.loads(text)
        rois = {name: ROI(**limits, name=name) for name, limits in doc.get("rois", {}).items()}
        geometry = []
        for entry in doc.get("geometry", []):
            entry = dict(entry)
            geometry.append((set(parse_runs(entry.pop("runs"))), entry))
        groups = doc.get("groups", {})
        cat = cls(raw=path.parent / doc.get("raw", "."), rois=rois, groups=groups, geometry_ranges=geometry,
                  path=path, sha256=hashlib.sha256(text.encode()).hexdigest())
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

        groups = [groups] if isinstance(groups, str) else list(groups)
        scans, rois, labels = {}, {}, {}
        for group in groups:
            for run in self.runs(group):
                if run in scans:
                    raise ValueError(f"run {run} is in group {group!r} and in {labels[run]['group']!r}")
                scans[run] = self.scan(run)
                rois[run] = self.roi(group, run)
                labels[run] = {"group": group, **self.labels(group)}
        out = run_series(scans, detector or channels.JF, channels.JF_I0 if i0 is None else i0,
                         roi=rois, labels=labels, **kwargs)
        out["settings"]["catalogue"] = str(self.path or "")
        out["settings"]["catalogue_sha256"] = self.sha256 or ""
        out["settings"]["groups"] = ", ".join(groups)
        return out
