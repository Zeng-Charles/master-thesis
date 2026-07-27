"""Checks on the phantom env config that don't need Isaac Sim.

Importing ``ultrasound_project`` pulls in Isaac Lab, so these tests read the config file
directly instead. They guard the couple of implicit contracts between the yaml, the task
module and the on-disk asset layout that are easy to break silently.
"""

from pathlib import Path

import pytest
from ruamel.yaml import YAML

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "configs" / "env" / "robotic_US_guidance_phantom.yaml"


@pytest.fixture(scope="module")
def cfg():
    return YAML().load(CONFIG_PATH.read_text())


def test_config_exists():
    assert CONFIG_PATH.is_file(), f"{CONFIG_PATH} is where the phantom task reads its config from"


@pytest.mark.parametrize(
    "section", ["robot", "patient", "bed", "motion_planning", "observation", "action", "reward", "sim"]
)
def test_required_sections(cfg, section):
    assert section in cfg


def test_robot_joint_pos_matches_arm_dof(cfg):
    assert cfg["robot"]["type"] in {"kuka", "fr3"}
    assert len(cfg["robot"]["joint_pos"]) == 7


def test_patient_ids_are_directories_under_assets(cfg):
    """The task builds <patients>/<id>/{raw,stl,usd} paths from patient.id_list."""
    for patient_id in cfg["patient"]["id_list"]:
        patient_dir = REPO_ROOT / "assets" / "patients" / patient_id
        if not patient_dir.is_dir():
            pytest.skip(f"{patient_id} assets not present (they are git-ignored)")
        for sub in ("raw", "stl", "usd"):
            assert (patient_dir / sub).is_dir(), f"missing {patient_id}/{sub}"


def test_target_meshes_follow_the_naming_convention(cfg):
    """VertebraViewer resolves <target_anatomy>.stl and standard_right_traj_<last 2 chars>.stl."""
    target = str(cfg["patient"]["target_anatomy"])
    for patient_id in cfg["patient"]["id_list"]:
        stl_dir = REPO_ROOT / "assets" / "patients" / patient_id / "stl"
        if not stl_dir.is_dir():
            pytest.skip(f"{patient_id} assets not present (they are git-ignored)")
        assert (stl_dir / f"{target}.stl").is_file()
        assert (stl_dir / f"standard_right_traj_{target[-2:]}.stl").is_file()


def test_init_range_sits_inside_the_scan_range(cfg):
    """Probe start poses are sampled from patient_xz_init_range; drifting outside
    patient_xz_range means sampling surface cells USSlicer never built T-maps for."""
    (x_lo, z_lo, _), (x_hi, z_hi, _) = cfg["sim"]["patient_xz_range"]
    (ix_lo, iz_lo, _), (ix_hi, iz_hi, _) = cfg["sim"]["patient_xz_init_range"]
    assert x_lo <= ix_lo <= ix_hi <= x_hi
    assert z_lo <= iz_lo <= iz_hi <= z_hi


def test_net_us_mode_needs_ct(cfg):
    """The 'net' renderer reads ct.nii.gz; the label map here is only a body contour."""
    if cfg["sim"]["us"] == "net":
        assert cfg["sim"]["if_use_ct"] is True
