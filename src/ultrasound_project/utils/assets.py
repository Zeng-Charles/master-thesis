"""Resolving patient body directories across this repo and the SonoGym submodule."""

import os

from ultrasound_project import PATIENTS_DIR


def resolve_patient_dirs(patient_id: str) -> tuple[str, str, str]:
    """Return the ``(raw, stl, usd)`` directories for one patient id.

    Bodies added by this repo are laid out per patient as
    ``assets/patients/<id>/{raw,stl,usd}``. Patients that came with SonoGym's asset
    download keep their original layout inside the submodule, where those same three
    roles are three parallel dataset roots with the patient id one level below. Callers
    get the same triple either way, so a config can mix both kinds of patient.

    The file names inside each directory are fixed by SonoGym: ``USSlicer`` reads
    ``ct.nii.gz`` from raw, ``combined_label_map.nii.gz`` plus the surface caches from
    stl, and the task spawns ``combined_wrapwrap/combined_wrapwrap.usd`` from usd.
    """
    local = os.path.join(PATIENTS_DIR, patient_id)
    if os.path.isdir(local):
        return (
            os.path.join(local, "raw"),
            os.path.join(local, "stl"),
            os.path.join(local, "usd"),
        )

    from spinal_surgery import ASSETS_DATA_DIR

    root = os.path.join(ASSETS_DATA_DIR, "HumanModels")
    return (
        os.path.join(root, "selected_dataset", patient_id),
        os.path.join(root, "selected_dataset_stl", patient_id),
        os.path.join(root, "selected_dataset_body_from_urdf", patient_id),
    )


def resolve_patient_lists(patient_ids) -> tuple[list[str], list[str], list[str]]:
    """:func:`resolve_patient_dirs` over an id list, transposed into three lists."""
    triples = [resolve_patient_dirs(p) for p in patient_ids]
    return (
        [t[0] for t in triples],
        [t[1] for t in triples],
        [t[2] for t in triples],
    )
