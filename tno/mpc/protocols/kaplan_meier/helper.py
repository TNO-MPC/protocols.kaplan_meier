"""
Module implements the helper party. This party is introduced for efficiency
purposes as well as enabling Shamir secret sharing. Shamir secret sharing
requires at least three parties.
"""

from typing import Any

from .player import Player


class Helper(Player):
    """
    Helper player that is added to the MPC protocol for efficiency purposes
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        Initializes the helper party

        :param args: arguments to pass on to base class
        :param kwargs: keyword arguments to pass on to base class
        """
        super().__init__(*args, **kwargs)

    async def start_protocol(self) -> None:
        """
        Starts and runs the protocol
        """
        await self.run_mpyc()
