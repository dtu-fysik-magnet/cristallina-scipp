"""Run overview of a raw folder, with the catalogue's groups."""

import pytest

import cristallina_scipp as cs
from cristallina_scipp import testing


@pytest.fixture(scope="module")
def cat(tmp_path_factory):
    return cs.Catalogue.load(testing.example_experiment(tmp_path_factory.mktemp("exp")))


def test_overview_lists_every_run_with_its_scan_and_metadata(cat):
    table = cs.run_overview(cat.raw)
    assert list(table.index) == [45, *range(60, 69), *range(70, 80)]
    assert table.loc[79, "status"] == "no scan.json"
    row = table.loc[72]
    assert (row["status"], row["scanned"], row["steps"], row["field"]) == ("ok", "SARES31-GPS:ROTTHETA", 41, 2.0)
    assert table.loc[45, "unit"] == "eV" and table.loc[64, "photon_energy"] == pytest.approx(7.511)


def test_catalogue_overview_shows_groups_and_gaps(cat):
    table = cat.overview()
    assert table.loc[60, "group"] == "energy 0T" and table.loc[45, "group"] == "fluorescence"
    assert table.loc[78, "group"] == ""  # a run no group lists
    cat.groups["field 0deg"]["runs"] = "70-73, 81"
    assert cat.overview().loc[81, "status"] == "not in raw"
    assert list(cat.overview(runs=[70, 71]).index) == [70, 71]
