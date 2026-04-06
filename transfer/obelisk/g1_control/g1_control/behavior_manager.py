import numpy as np
import torch

from sensor_msgs.msg import Joy

from .policy import RLPolicy


class BehaviorManager:
    """
    Manages which behavior is in use.
    """
    def __init__(self,
                 behavior_names: list[str],
                 behavior_buttons: list[int],
                 init_behavior: str,
                 vel_thresholds: list[float],
                 hf_repo_ids: list[str],
                 hf_policy_folders: list[str],):
        """
        Initialize the behavior manager.

        TODO: Need to take in a list of valid behavior transitions.
        TODO: Add support for one policy with multiple behaviors.
        """

        self.behavior_names = behavior_names
        self.behavior_buttons = behavior_buttons
        self.vel_thresholds = vel_thresholds

        self.policies = []
        for repo_id, policy_folder in zip(hf_repo_ids, hf_policy_folders):
            self.policies.append(RLPolicy(repo_id, policy_folder))

        # TODO: Put in a better check
        # if len(self.vel_thresholds) != 0 and len(self.vel_thresholds) != 2*len(self.policies):
        #     raise ValueError(f"Expected no velocity thresholds or length of {2*len(self.policies)}, got: {len(self.vel_thresholds)}")

        self.active_behavior = init_behavior

        self.last_behavior_switch = 0.0
        self.pending_behavior: str | None = None

    def check_behavior_switch(self, joy_msg: Joy, cmd_vel: np.ndarray, time: float) -> str:
        """Check for behavior switch request and queue if valid.

        The actual switch is deferred until phi is near 0 or 0.5.

        Args:
            joy_msg: Joystick message containing button states.
            time: Current time.

        Returns:
            The currently active behavior name.
        """
        if (time - self.last_behavior_switch) > 0.1:
            # if len(self.vel_thresholds) > 1:
            #     for i in range(len(self.behavior_names)):
            #         low = self.vel_thresholds[2 * i + 1]
            #         high = self.vel_thresholds[2 * i]
            #         if low <= cmd_vel[0] < high:
            #             requested_behavior = self.behavior_names[i]
            #             if requested_behavior != self.active_behavior and requested_behavior != self.pending_behavior:
            #                 self.pending_behavior = requested_behavior

            for i, button in enumerate(self.behavior_buttons):
                if joy_msg.buttons[button] == 1:
                    requested_behavior = self.behavior_names[i]
                    # Queue if different from active and not already pending
                    if requested_behavior != self.active_behavior and requested_behavior != self.pending_behavior:
                        self.pending_behavior = requested_behavior

        return self.active_behavior

    def try_execute_pending_switch(self, time: float) -> bool:
        """Execute pending switch if phi is near 0 or 0.5.

        Should be called after create_obs() where phi is computed.

        Args:
            time: Current time for updating last_behavior_switch.

        Returns:
            True if a switch was executed, False otherwise.
        """
        if self.pending_behavior is None:
            return False

        if len(self.vel_thresholds) > 1:
            current_policy = self.get_active_policy()
            phi = current_policy.get_phi()

            # Check if phi is near 0 (including wrap-around) or 0.5
            PHI_TOLERANCE = 0.03

            # For now just hard-code the phasing variables for each transition
            current_policy_idx = self.get_active_policy_idx()
            if self.active_behavior == "walking":
                val = 0.164
            elif self.active_behavior == "running":
                val = 0.03 #0.9 #0.06 #0.5
            else:
                raise ValueError(f"Invalid active behavior: {self.active_behavior}")

            at_val = abs(phi - val) < PHI_TOLERANCE

            if at_val: #at_zero or at_half:
                self.active_behavior = self.pending_behavior
                self.last_behavior_switch = time
                self.pending_behavior = None
                self.policies[self.get_active_policy_idx()].reset_last_action()
                self.policies[self.get_active_policy_idx()].set_last_zero_time(time)
                return True
        else:
            self.active_behavior = self.pending_behavior
            self.last_behavior_switch = time
            self.pending_behavior = None
            self.policies[self.get_active_policy_idx()].reset_last_action()
            self.policies[self.get_active_policy_idx()].set_last_zero_time(time)
            return True

        return False

        ## For debugging:
        # current_policy = self.get_active_policy()
        # phi = current_policy.get_phi()
        # print(f"PHI: {phi}")
        #
        # self.active_behavior = self.pending_behavior
        # self.last_behavior_switch = time
        # self.pending_behavior = None
        # self.policies[self.get_active_policy_idx()].reset_last_action()
        #
        # return True

    def is_switch_pending(self) -> bool:
        """Check if a behavior switch is pending."""
        return self.pending_behavior is not None

    def get_pending_behavior(self) -> str | None:
        """Get the name of the pending behavior, if any."""
        return self.pending_behavior

    def get_active_behavior(self):
        return self.active_behavior

    def get_active_policy(self):
        return self.policies[self.get_active_policy_idx()]

    def get_active_policy_idx(self):
        return self.behavior_names.index(self.active_behavior)

    def create_obs(self,
                   qfb: np.ndarray,
                   vfb_ang: np.ndarray,
                   qjoints: np.ndarray,
                   vjoints: np.ndarray,
                   time: float,
                   cmd_vel: np.ndarray,
                   joint_names: list[str],
                   ):
        policy_idx = self.get_active_policy_idx()
        return self.policies[policy_idx].create_obs(qfb, vfb_ang, qjoints, vjoints, time - self.last_behavior_switch, cmd_vel, joint_names)

    def get_action(self, obs: torch.Tensor, joint_names_out: list[str]) -> np.ndarray:
        policy_idx = self.get_active_policy_idx()
        return self.policies[policy_idx].get_action(obs, joint_names_out)
