"""Checks against cristallina's own regression numbers (needs the [rsm] extra and its checkout's test data)."""

from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import scipp as sc
from sfdata import SFScanInfo

import cristallina_scipp as cs
from cristallina_scipp import ROI

# import before patching: cristallina replaces jungfrau_utils' file locators on import
perform_image_stack_sum = pytest.importorskip("cristallina.analysis").perform_image_stack_sum
crROI = pytest.importorskip("cristallina.utils").ROI

DATA = Path(__file__).parents[2] / "cristallina_analysis_package/tests/data"
RUN = DATA / "p20841/raw/run0205"
JF, I0 = "JF16T03V01", "SARFE10-PBPS053:INTENSITY"

pytestmark = pytest.mark.skipif(not RUN.exists(), reason="cristallina test data not checked out")


@pytest.fixture(autouse=True)
def local_jf_calibration():
    with mock.patch("jungfrau_utils.file_adapter.locate_gain_file", lambda p, **k: str(DATA / "gains.h5")), \
         mock.patch("jungfrau_utils.file_adapter.locate_pedestal_file", lambda p, **k: str(DATA / "JF16T03V01.res.h5")):
        yield


def local_scan():
    scan = SFScanInfo(str(RUN / "meta/scan.json"))
    scan.files = [[str(RUN / "data" / Path(f).name) for f in fs] for fs in scan.files]
    return scan


def test_reduce_scan_matches_cristallina():
    roi = ROI(575, 750, 600, 750, "peak")
    expected = perform_image_stack_sum(
        local_scan().files[0], channel=JF, alignment_channels=[I0], lower_cutoff_threshold=4.0,
        roi=crROI(left=575, right=750, bottom=600, top=750, name="peak"),
    )
    full = cs.reduce_scan(local_scan(), JF, I0)
    cropped = cs.reduce_scan(local_scan(), JF, I0, roi=roi, cutoff=4.0)
    by_label = cs.roi_intensities(full, [roi])
    assert full.sizes == {"step": 1, "y": 1614, "x": 1030}
    assert np.allclose(cropped["step", 0].values, expected)
    assert np.isclose(by_label["roi", 0]["step", 0].value, full["step", 0]["y", 600:750]["x", 575:750].sum().value)
    assert full.coords["n_pulses"].values[0] == 21


def test_photon_variances():
    da = cs.reduce_scan(local_scan(), JF, I0, roi=ROI(575, 750, 600, 750, "peak"), cutoff=4.0, photon_energy=8.0)
    assert da.unit == "counts"
    assert np.array_equal(da.variances, da.values)
