"""Shared fixtures for trajectory tracking tests."""

from pathlib import Path

import pytest
import torch

from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking.trajectory_manager import TrajectoryManager
from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking.library_manager import LibraryManager

TEST_DATA = Path(__file__).parent / "test_data"
STANDING_YAML = str(TEST_DATA / "standing.yaml")
WALKING_YAML = str(TEST_DATA / "walking_100.yaml")
RUNNING_YAML = str(TEST_DATA / "running_200.yaml")
MERGED_LIBRARY_DIR = str(TEST_DATA / "merged_library")

DEVICE = "cpu"


@pytest.fixture(scope="module")
def standing_manager() -> TrajectoryManager:
    """Standing trajectory: periodic, 1 domain (double_support), T=0.4."""
    return TrajectoryManager(STANDING_YAML, None, DEVICE)


@pytest.fixture(scope="module")
def walking_manager() -> TrajectoryManager:
    """Walking trajectory: half_periodic, 1 domain (single_support), T=0.46."""
    return TrajectoryManager(WALKING_YAML, None, DEVICE)


@pytest.fixture(scope="module")
def running_manager() -> TrajectoryManager:
    """Running trajectory: half_periodic, 2 domains (single_support + flight_phase), T=0.299."""
    return TrajectoryManager(RUNNING_YAML, None, DEVICE)


@pytest.fixture(scope="module")
def merged_library() -> LibraryManager:
    """Merged walk/run library: 19 YAML files, mixed speeds."""
    return LibraryManager(MERGED_LIBRARY_DIR, None, DEVICE)
