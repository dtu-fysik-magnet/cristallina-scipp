"""Catalogue folders, checks, and the shared git repository with its hooks."""

import shutil
import subprocess

import pytest

import cristallina_scipp as cs
from cristallina_scipp import catalogue, testing

BASE = """raw = "raw"
[rois]
"peak A" = { left = 0, right = 64, bottom = 22, top = 38 }
[[geometry]]
runs = "1-100"
x0 = 416.81
y0 = 775.27
d0 = 0.568
z0 = 0.2
stt0 = -0.09935076
"""
GROUP = """[groups."energy 0T"]
runs = "60-68"
roi = "peak A"
"""


@pytest.fixture
def folder(tmp_path):
    d = tmp_path / "catalogue"
    d.mkdir()
    (d / "00-setup.toml").write_text(BASE)
    (d / "energy.toml").write_text(GROUP)
    return d


def test_a_folder_reads_as_one_catalogue(folder):
    cat = cs.Catalogue.load(folder)
    assert cat.runs("energy 0T") == list(range(60, 69)) and cat.roi("energy 0T", 60).name == "peak A"
    assert cat.raw == folder / "raw" and len(cat.sha256) == 64
    before = cat.sha256
    (folder / "energy.toml").write_text(GROUP.replace("60-68", "60-69"))
    assert cs.Catalogue.load(folder).sha256 != before


@pytest.mark.parametrize("text, message", [
    ('[groups."energy 0T"]\nruns = "1"\n', "also defined in energy.toml"),
    ('[group."typo"]\nruns = "1"\n', "unknown section"),
    ('raw = "elsewhere"\n', "but 00-setup.toml says"),
    ('[[geometry]]\nruns = "50-60"\nx0 = 1\ny0 = 1\nd0 = 1\nz0 = 0\nstt0 = 0\n', "two [[geometry]]"),
    ('[groups."x"\nruns = "1"\n', "zz-new.toml"),
])
def test_load_names_the_problem(folder, text, message):
    (folder / "zz-new.toml").write_text(text)
    with pytest.raises(ValueError, match=message.replace("[", r"\[").replace("]", r"\]")):
        cs.Catalogue.load(folder)


def test_check_reports_errors_and_warnings(folder, capsys):
    (folder / "more.toml").write_text('[groups."again"]\nruns = "68-70"\nroi = "peak A"\n')
    (folder.parent / "catalogue" / "raw" / "run0060").mkdir(parents=True)
    errors, warnings = catalogue.check(folder)
    assert not errors
    assert any("run 68 is in groups" in w for w in warnings)
    assert any("not in" in w and "61" in w for w in warnings)
    (folder / "broken.toml").write_text("[rois\n")
    assert catalogue.main(["check", str(folder)]) == 1
    assert "broken.toml" in capsys.readouterr().out


def git(*args, cwd):
    return subprocess.run(["git", "-c", "safe.directory=*", *args], cwd=cwd, check=True, capture_output=True,
                          text=True).stdout.strip()


@pytest.fixture
def identity(monkeypatch):
    for key, value in {"GIT_AUTHOR_NAME": "Test", "GIT_AUTHOR_EMAIL": "test@example.org",
                       "GIT_COMMITTER_NAME": "Test", "GIT_COMMITTER_EMAIL": "test@example.org"}.items():
        monkeypatch.setenv(key, value)


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_shared_repository_validates_and_deploys(tmp_path, identity):
    work = tmp_path / "work"
    work.mkdir()
    source = tmp_path / "start.toml"
    source.write_text(testing.EXAMPLE_CATALOGUE)
    deploy = catalogue.setup(work, source)
    cat = cs.Catalogue.load(deploy)
    assert "energy 0T" in cat.groups
    assert cat.commit == git("rev-parse", "main", cwd=work / "catalogue.git")

    mine = tmp_path / "mine"
    git("clone", "-q", str(work / "catalogue.git"), str(mine), cwd=tmp_path)
    f = mine / "start.toml"
    f.write_text(f.read_text() + '\n[groups."field 30deg"]\nruns = "80-83"\nroi = "peak B"\n')
    git("commit", "-qam", "Add the 30 degree series", cwd=mine)
    git("push", "-q", "origin", "HEAD:main", cwd=mine)
    cat = cs.Catalogue.load(deploy)
    assert "field 30deg" in cat.groups and cat.commit == git("rev-parse", "HEAD", cwd=mine)

    f.write_text(f.read_text() + '\n[groups."broken"]\nruns = "90"\nroi = "no such roi"\n')
    git("commit", "-qam", "Break it", cwd=mine)
    push = subprocess.run(["git", "push", "origin", "HEAD:main"], cwd=mine, capture_output=True, text=True)
    assert push.returncode != 0 and "push refused" in push.stderr
    assert "broken" not in cs.Catalogue.load(deploy).groups  # the checkout notebooks load is unchanged

    with pytest.raises(PermissionError):
        (deploy / "start.toml").write_text("edited in place")  # read-only: edits go through git
    with pytest.raises(FileExistsError):
        catalogue.setup(work)


def test_series_record_the_commit(tmp_path):
    cat = cs.Catalogue.load(testing.example_experiment(tmp_path))
    assert cat.commit == ""  # not in git
    out = cat.series("field 0deg", cutoff=7.0, i0_cutoff=2.5, photon_energy=7.511)
    assert out["settings"]["catalogue_commit"] == ""
