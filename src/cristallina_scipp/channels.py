"""Cristallina channel names used here, copied from `cristallina.channels`."""

GASMONITOR = "SARFE10-PBIG050-EVR0:CALCI"
JF = "JF16T03V02"  # Jungfrau 1.5M
JF_I0 = "JF20T01V01"  # Jungfrau I0 monitor
PBPS149_INTENSITY = "SAROP31-PBPS149:INTENSITY_UJ"

# Run metadata read by cristallina_scipp.series.run_metadata (PVs, one value per step)
FEL_PHOTON_ENERGY = "SARUN:FELPHOTENE.VAL"  # keV, undulator setting
MONO_ENERGY = "SAROP31-ODCC110:MOT_ENY.RBV"  # eV
MAGNET = tuple(f"SARES31-MAG-IPS1-{axis}:SIG_FLD" for axis in "XYZ")  # T, vector magnet
ATTENUATORS = ("SARFE10-OATT053:UsrRec.TC1", "SAROP31-OATA150:UsrRec.TC1")  # front end, experiment
