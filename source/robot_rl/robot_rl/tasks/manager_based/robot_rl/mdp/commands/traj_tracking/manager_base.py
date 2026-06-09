# Copyright 2026 Zachary Olkin. All rights reserved.

from abc import ABC, abstractmethod

import torch


class ManagerBase(ABC):
    """Abstract base class for trajectory and skill managers.

    Provides a common interface for managing trajectories, including domain timing,
    output computation, reference frames, and contact states.
    """

    @abstractmethod
    def get_domain_times(self, t: torch.Tensor) -> torch.Tensor:
        """Get the duration of the current domain for each environment.

        Args:
            t: Shape [N] where N is the number of environments.

        Returns:
            Domain durations of shape [N] for each environment's current domain.
        """
        pass

    @abstractmethod
    def get_output(self, t: torch.Tensor) -> torch.Tensor:
        """Compute the trajectory output (bezier values) at a given time.

        Args:
            t: Shape [N] where N is the number of environments.

        Returns:
            Outputs of shape [N, 2, num_outputs] where the second dimension contains
            position (index 0) and velocity (index 1).
        """
        pass

    @abstractmethod
    def get_num_outputs(self) -> int:
        """Get the total number of outputs in the trajectory.

        Returns:
            The number of outputs (e.g., frame axes + joints).
        """
        pass

    @abstractmethod
    def get_current_domains(self, t: torch.Tensor) -> torch.Tensor:
        """Determine which domain each environment is in given the time.

        Args:
            t: Shape [N] where N is the number of environments.

        Returns:
            Domain indices of shape [N] for each environment.
        """
        pass

    @abstractmethod
    def get_num_domains(self) -> int:
        """Get the total number of domains in the trajectory.

        Returns:
            The number of domains. For half-periodic trajectories, this is
            twice the number of explicitly defined domains.
        """
        pass

    @abstractmethod
    def get_ref_frames_in_use(self, t: torch.Tensor, ref_frames: list[str]) -> torch.Tensor:
        """Determine the reference frame in use for each environment.

        Args:
            t: Shape [N] where N is the number of environments.
            ref_frames: A list of reference frame names.

        Returns:
            Frame indices of shape [N] where each value is the index into
            ref_frames for the active frame of that environment.
        """
        pass

    @abstractmethod
    def get_contact_state(self, t: torch.Tensor, contact_frames: list[str]) -> torch.Tensor:
        """Return the contact state of each contact point at the given time.

        Args:
            t: Shape [N] where N is the number of environments.
            contact_frames: List of contact frame names to check the state for.

        Returns:
            Contact states of shape [N, num_contacts] where num_contacts is
            len(contact_frames). A value of 1 indicates in contact, 0 otherwise.
        """
        pass
