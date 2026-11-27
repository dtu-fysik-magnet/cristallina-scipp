"""Pixel-energy histograms for choosing the fluorescence window."""

import scipp as sc
from sfdata import SFScanInfo

import cristallina_scipp as cs
from cristallina_scipp import testing


def test_pixel_energies_separate_fluorescence_from_elastic(tmp_path, capfd):
    scan = SFScanInfo(str(testing.fluorescence_energy_scan(tmp_path, run=45)))
    below, above = cs.pixel_energies(scan, range(5)), cs.pixel_energies(scan, range(36, 41))
    window = slice(sc.scalar(5.5, unit="keV"), sc.scalar(6.8, unit="keV"))
    assert above["energy", window].sum().value > 2 * below["energy", window].sum().value  # fluorescence switches on
    assert below.coords["energy"].unit == sc.Unit("keV") and below.sizes["energy"] == 150
    one = sum(cs.pixel_energies(scan, i).sum().value for i in range(5))
    assert below.sum().value == one  # several steps add up
    assert "already converted" not in capfd.readouterr().out  # opened quietly
