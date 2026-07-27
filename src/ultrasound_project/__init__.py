"""Master-thesis code built on top of the SonoGym fork (``sonogym-custom/``).

Importing this package registers every Gym environment defined under
:mod:`ultrasound_project.tasks`, in the same way ``import spinal_surgery``
registers SonoGym's own tasks.
"""

import os

PACKAGE_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(PACKAGE_DIR, "..", ".."))

ASSETS_DIR = os.path.join(PROJECT_DIR, "assets")
PATIENTS_DIR = os.path.join(ASSETS_DIR, "patients")
ROBOTS_DIR = os.path.join(ASSETS_DIR, "robots")
PROBES_DIR = os.path.join(ASSETS_DIR, "probes")
SCENES_DIR = os.path.join(ASSETS_DIR, "scenes")

CONFIGS_DIR = os.path.join(PROJECT_DIR, "configs")
DATA_DIR = os.path.join(PROJECT_DIR, "data")
OUTPUTS_DIR = os.path.join(PROJECT_DIR, "outputs")

from .tasks import *  # noqa: E402,F401,F403
