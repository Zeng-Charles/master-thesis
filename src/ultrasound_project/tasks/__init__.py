"""Gym task registrations that extend SonoGym's own task set.

SonoGym auto-discovers its tasks via ``isaaclab_tasks.utils.import_packages``.
The sub-packages here are listed explicitly instead: there are only a handful of
them, and an explicit list makes it obvious which environment ids this repo adds
on top of the ones ``spinal_surgery`` already registers.
"""

from . import robot_US_guidance_cam  # noqa: F401
from . import robot_US_guidance_phantom  # noqa: F401
