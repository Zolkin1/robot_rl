# Copyright 2026 Zachary Olkin. All rights reserved.

from trajectory_manager import TrajectoryManager


class TestTrajectoryManager:
    """Test the trajectory manager."""

    def test_load_from_yaml(self):
        # Get path to test yaml file
        manager = TrajectoryManager("robot_rl/robot_rl/assets/robots/test_walking_trajectories")

        assert manager.traj_data.name == "1_walk"

