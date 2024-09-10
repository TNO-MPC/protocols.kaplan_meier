"""
Module implements the data owners Alice and Bob.
"""

from __future__ import annotations

import asyncio
from typing import Any, SupportsInt, cast

import numpy as np
import numpy.typing as npt
import pandas as pd

from tno.mpc.communication import Pool
from tno.mpc.encryption_schemes.paillier import Paillier

from .player import Player


class DataOwner(Player):
    """
    Data owner in the MPC protocol
    """

    def __init__(
        self,
        data: pd.DataFrame,  # type: ignore[name-defined]
        pool: Pool,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        r"""
        Initializes data owner

        :param data: the data to use for this data owner
        :param pool: a communication pool
        :param \*args: arguments to pass on to base class
        :param \**kwargs: keyword arguments to pass on to base class
        """
        super().__init__(*args, **kwargs)
        self._paillier_scheme: Paillier | None = None
        self._data = data
        self.pool = pool

    async def receive_message(self, party: str, msg_id: str | None = None) -> Any:
        """
        Receives a message from a party (belonging to an optional message
        identifier)

        :param party: the party to receive a message from
        :param msg_id: the message id
        :return: the received message
        """
        return await self.pool.recv(party, msg_id=msg_id)

    async def send_message(
        self, receiver: str, message: Any, msg_id: str | None = None
    ) -> None:
        """
        Sends a message to a party (with an optional message identifier)

        :param receiver: the party to send a message to
        :param message: the message to send
        :param msg_id: the message id
        """
        await self.pool.send(receiver, message, msg_id=msg_id)

    @property
    def records(self) -> int:
        """
        Number of records in the loaded dataset

        :return: number of records
        """
        return cast(int, self.data.shape[0])

    @property
    def nr_groups(self) -> int:
        """
        Number of groups in the loaded datasets

        :return: number of groups
        :raise NotImplementedError: raised when not implemented
        """
        raise NotImplementedError()

    @property
    def data(self) -> pd.DataFrame:  # type: ignore[name-defined]
        """
        The loaded dataset

        :return: dataset
        :raise ValueError: raised when there is no data available
        """
        if self._data is None:
            raise ValueError("No event data available yet.")
        return self._data

    @property
    def paillier_scheme(self) -> Paillier:
        """
        The Paillier scheme

        :return: Paillier scheme
        :raise ValueError: raised when Paillier scheme is not available yet.
        """

        if self._paillier_scheme is None:
            raise ValueError("There is no Paillier scheme available yet.")
        return self._paillier_scheme

    def start_randomness_generation(self, amount: int) -> None:
        """
        Kicks off the randomness generation. This boosts performance.
        In particular will this decrease the total runtime (as data owners can
        already generate randomness before they need it).

        :param amount: amount of randomness to precompute.
        """
        self.paillier_scheme.boot_randomness_generation(
            amount,
        )

    def encrypt(self, data: npt.NDArray[np.int_]) -> npt.NDArray[np.object_]:
        """
        Method to encrypt a dataset using the initialized Paillier scheme

        :param data: the dataset to encrypt
        :return: an encrypted dataset
        """
        encrypted_data: npt.NDArray[np.object_] = np.vectorize(
            self.paillier_scheme.unsafe_encrypt
        )(data)
        return encrypted_data

    def shut_down_paillier(self) -> None:
        """
        Shut down Paillier scheme.
        """
        self.paillier_scheme.shut_down()


class Alice(DataOwner):
    """
    Alice player in the MPC protocol
    """

    def __init__(
        self,
        *args: Any,
        time_label: str = "time",
        event_label: str = "event",
        security_level: int | None = 40,
        **kwargs: Any,
    ) -> None:
        """
        Initializes player Alice

        :param time_label: the label used to represent the 'time' column in the data set
        :param event_label: the label used to represent the 'event' column in the data set
        :param security_level: the statistical security level to use for the additive masking (in bits)
        :param args: arguments to pass on to base class
        :param kwargs: keyword arguments to pass on to base class
        """
        super().__init__(*args, **kwargs)
        self.time_label = time_label
        self.event_label = event_label
        self.security_level = security_level
        self._encrypted_group_data_: npt.NDArray[np.object_] | None = None
        self._kaplan_meier_statistics_group: npt.NDArray[np.object_] | None = None
        self._kaplan_meier_statistics_total: npt.NDArray[np.int_] | None = None
        self._mask_ht = None
        self._nr_groups = None

    @property
    def nr_groups(self) -> int:
        """
        Number of groups in the datasets.

        :return: number of groups
        :raise ValueError: raised when number of groups is not available (yet)
        """
        if self._nr_groups is None:
            raise ValueError("Number of groups is not available yet")
        return self._nr_groups

    @property
    def unique_event_times(self) -> npt.NDArray[np.int_]:
        """
        List of unique event times.

        :return: unique event times
        """
        return cast(
            npt.NDArray[np.int_],
            self.data[self.time_label]
            .loc[self.data[self.event_label].astype(bool)]
            .unique(),
        )

    @property
    def nr_unique_event_times(self) -> int:
        """
        Number of unique event times.

        :return: number of unique event times
        """
        return len(self.unique_event_times)

    @property
    def _encrypted_group_data(self) -> npt.NDArray[np.object_]:
        """
        Encrypted group data.

        :return: the encrypted group data
        :raise ValueError: raised when the encrypted group data is not yet available
        """
        if self._encrypted_group_data_ is None:
            raise ValueError("Alice is missing some important data.")
        return self._encrypted_group_data_

    @_encrypted_group_data.setter
    def _encrypted_group_data(self, data: npt.NDArray[np.object_]) -> None:
        self._encrypted_group_data_ = data

    @property
    def kaplan_meier_statistics_group(self) -> npt.NDArray[np.object_]:
        """
        Encrypted table with Kaplan-Meier statistics per group.

        Row i contains data for the i-th unique event times. The first set of
        (groups-1) columns contains the number of observed events for group 1,
        2, ..., groups-1. The second set of (groups-1) columns contains the
        number of individuals at risk for those groups.

        :return: the constructed table
        :raise ValueError: raised when table is not yet available
        """
        if self._kaplan_meier_statistics_group is None:
            raise ValueError(
                "Table with Kaplan-Meier statistics per group is not yet set."
            )
        return self._kaplan_meier_statistics_group

    @property
    def kaplan_meier_statistics_total(
        self,
    ) -> npt.NDArray[np.int_]:
        """
        Table with Kaplan-Meier statistics of the entire population.

        Row i contains data for the i-th unique event times. The first column
        contains the number of observed events for the entire population. The
        second column contains the number of individuals at risk for the
        entire population.

        :return: the constructed table
        :raise ValueError: raised when table is not yet available
        """
        if self._kaplan_meier_statistics_total is None:
            raise ValueError("Plain table is not set yet.")
        return self._kaplan_meier_statistics_total

    async def run_protocol(self) -> None:
        """
        Starts and runs the protocol
        """
        await asyncio.gather(
            *[
                self.receive_paillier_scheme(),
                self.receive_number_of_groups(),
            ]
        )
        # for rerandomizing self.kaplan_meier_statistics_group
        self.start_randomness_generation(
            amount=self.nr_unique_event_times * 2 * (self.nr_groups - 1)
        )
        await self.receive_encrypted_group_data()
        self.compute_kaplan_meier_minimal_statistics()
        self.compute_logrank_factors()
        self.generate_additive_share()
        await self.send_additive_share()
        self.shut_down_paillier()
        await self.run_mpyc_logrank_test()

    async def receive_paillier_scheme(self) -> None:
        """
        Method to receive the Paillier scheme that is used by party Bob.
        """
        self._paillier_scheme = await self.receive_message(
            self.party_B, msg_id="paillier_scheme"
        )

    async def receive_number_of_groups(self) -> None:
        """
        Method to receive the number of groups identified by party Bob.
        """
        self._nr_groups = await self.receive_message(
            self.party_B, msg_id="number_of_groups"
        )

    async def receive_encrypted_group_data(self) -> None:
        """
        Method to receive the encrypted group data from party Bob.
        """
        self._encrypted_group_data = await self.receive_message(
            self.party_B, msg_id="encrypted_group_data"
        )

    def compute_kaplan_meier_minimal_statistics(self) -> None:
        """
        Method to compute the minimal statistics of the protocol.
        """
        self._logger.info("Computing Kaplan-Meier features from encrypted data...")
        self._sort_data()
        self._compute_kaplan_meier_minimal_statistics()
        self._logger.info("Done computing Kaplan-Meier features from encrypted data")

    def _sort_data(self) -> None:
        """
        Sort data by time (ascending).

        :raise AttributeError: raised when data is not a pandas dataframe
        """
        self._data = self.data.sort_values(by=self.time_label, ascending=True)
        self._encrypted_group_data = self._encrypted_group_data[
            cast(slice, self.data.index)
        ]

    def _compute_kaplan_meier_minimal_statistics(self) -> None:
        """
        Obtain number of observed events and number of individuals at risk for
        every group.

        :raise ValueError: raised when event indices are not determined
        """
        i_event_time, t_event_time = (
            self.nr_unique_event_times - 1,
            self.unique_event_times[-1],
        )
        at_risk_total = np.zeros((self.nr_unique_event_times, 1), dtype=np.int_)
        at_risk_group = np.zeros(
            (self.nr_unique_event_times, self.nr_groups - 1), dtype=object
        )
        observed_total = np.zeros((self.nr_unique_event_times, 1), dtype=np.int_)
        observed_group = np.zeros(
            (self.nr_unique_event_times, self.nr_groups - 1), dtype=object
        )

        for (_, total_data), group_data in zip(
            self.data.iloc[::-1].iterrows(),
            self._encrypted_group_data[::-1],
        ):
            if total_data[self.time_label] >= t_event_time:
                at_risk_total[i_event_time] += 1
                at_risk_group[i_event_time] += group_data
            else:
                i_event_time -= 1
                t_event_time = self.unique_event_times[i_event_time]
                at_risk_total[i_event_time] = at_risk_total[i_event_time + 1] + 1
                at_risk_group[i_event_time] = (
                    at_risk_group[i_event_time + 1] + group_data
                )
            if total_data[self.event_label] == 1:
                observed_total[i_event_time] += 1
                observed_group[i_event_time] += group_data

        assert i_event_time == 0

        self._kaplan_meier_statistics_group = np.c_[observed_group, at_risk_group]
        self._kaplan_meier_statistics_total = np.c_[observed_total, at_risk_total]

    def compute_logrank_factors(self) -> None:
        """Pre-computes several factors for in the computation of the log-
        rank statistic, leveraging information known by Alice only.

        Computes the following factors: expectation_factors, variance_factors,
        variance_factors_2. These factors satisfy the following relations:

          Expected number of observations in group i =
            expectation_factors[i] * at_risk_group[i]

          Variance of observations in group i =
            (variance_factors_2[i] - variance_factors[i] * at_risk_group[i])
            * at_risk_group[i]
        """
        at_risk_total = self.kaplan_meier_statistics_total[:, 1]
        observed_total = self.kaplan_meier_statistics_total[:, 0]

        # Expected number of observations(E)  =
        #   (observations_total / at_risk_total) *
        #     [at_risk_group]
        expectation_factors = observed_total / at_risk_total

        # Variance =
        #   observations_total * (at_risk_total - observations_total) /
        #     (at_risk_total**2 * (at_risk_total - 1)) *
        #       [at_risk_group] * (at_risk_total - [at_risk_group])
        # Note here that the denominator equals zero if at_risk_total
        # equals one, which is only possible in the last event time.
        # The variance should then also equal zero. Since
        # observations_total is always strictly positive, we find that
        # necessarily at_risk_total - observations_total = 0 if
        # at_risk_total = 0. Therefore, the following produces the
        # correct variance for every event time without dividing by
        # zero.
        variance_factors = (
            observed_total * (at_risk_total - observed_total) / at_risk_total**2
        )
        variance_non_zero_ind = at_risk_total != 1
        variance_factors[variance_non_zero_ind] /= (
            at_risk_total[variance_non_zero_ind] - 1
        )
        variance_factors_2 = variance_factors * at_risk_total
        self._logrank_computation_factors = np.c_[
            expectation_factors, variance_factors, variance_factors_2
        ]

    def generate_additive_share(self) -> None:
        """
        Generates additive secret shares.
        """
        self._additive_share: npt.NDArray[np.int_] = np.vectorize(
            lambda _: self.signed_randomness()
        )(np.ndarray(self.kaplan_meier_statistics_group.shape))
        self._logger.info("Generated share")

    def mask_kaplan_meier_statistics_group(
        self,
    ) -> npt.NDArray[np.object_]:
        """
        Masks the table with Kaplan-meier group statistics.

        :return: the masked table
        """
        return cast(
            npt.NDArray[np.object_],
            self.kaplan_meier_statistics_group - self.additive_share,
        )

    async def send_additive_share(self) -> None:
        """
        Sends additive secret share to party Bob.
        """
        loop = asyncio.get_event_loop()
        masked_group_statistics = await loop.run_in_executor(
            None, self.mask_kaplan_meier_statistics_group
        )
        self._logger.info("Encrypting additive shares...")
        randomize_ndarray(masked_group_statistics)
        self._logger.info("Done encrypting additive shares")
        await self.send_message(self.party_B, masked_group_statistics, msg_id="share")
        self._logger.info("Sent share")

    def signed_randomness(self) -> SupportsInt:
        """
        Returns a signed random plaintext value.

        :return: signed random plaintext value
        """
        return self.paillier_scheme.sample_mask(
            lower_bound=0,
            upper_bound=self.kaplan_meier_statistics_total[0, 1] + 1,
            security_level=self.security_level,
        )


class Bob(DataOwner):
    """
    Bob player in the MPC protocol
    """

    def __init__(
        self,
        *args: Any,
        paillier_scheme: Paillier = Paillier.from_security_parameter(
            key_length=2048, precision=0
        ),
        **kwargs: Any,
    ) -> None:
        """
        Initializes player Bob

        :param paillier_scheme: the Paillier scheme to use for encryption
        :param args: arguments to pass on to base class
        :param kwargs: keyword arguments to pass on to base class
        """
        super().__init__(*args, **kwargs)
        self._paillier_scheme = paillier_scheme
        self.encrypted_data: npt.NDArray[np.object_] | None = None

    @property
    def nr_groups(self) -> int:
        """
        Number of groups in the loaded dataset

        :return: number of groups
        """
        return cast(int, self.data.shape[1])

    async def run_protocol(self) -> None:
        """
        Starts and runs the protocol
        """
        # for encrypting self.data
        self.start_randomness_generation(
            amount=self.data.shape[0] * (self.nr_groups - 1)
        )
        loop = asyncio.get_event_loop()
        _, _, self.encrypted_data = await asyncio.gather(
            self.send_paillier_scheme(),
            self.send_number_of_groups(),
            loop.run_in_executor(None, self.encrypt, self.data.iloc[:, :-1]),
        )
        await self.send_encrypted_group_data()
        self.shut_down_paillier()
        await self.receive_additive_share()
        await self.run_mpyc_logrank_test()

    async def send_paillier_scheme(self) -> None:
        """
        Sends the used Paillier scheme to party Alice.
        """
        await self.send_message(
            self.party_A, self.paillier_scheme, msg_id="paillier_scheme"
        )

    async def send_number_of_groups(self) -> None:
        """
        Sends the number of groups to party Alice.
        """
        await self.send_message(self.party_A, self.nr_groups, msg_id="number_of_groups")

    async def send_encrypted_group_data(self) -> None:
        """
        Sends the encrypted dataset to party Alice.
        """
        self._logger.info("Encrypting group data...")
        randomize_ndarray(cast(npt.NDArray[np.object_], self.encrypted_data))
        self._logger.info("Done encrypting group data")
        await self.send_message(
            self.party_A, self.encrypted_data, msg_id="encrypted_group_data"
        )

    async def receive_additive_share(self) -> None:
        """
        Receive additive secret share produced by party Alice.
        """
        encrypted_additive_share = await self.receive_message(
            self.party_A, msg_id="share"
        )
        self._additive_share = await self.decrypt_additive_share(
            encrypted_additive_share
        )
        self._logrank_computation_factors = np.zeros(
            (len(self._additive_share), 3), dtype=np.int_
        )

    async def decrypt_additive_share(
        self, data: npt.NDArray[np.object_]
    ) -> npt.NDArray[np.int_]:
        """
        Decrypt share

        :param data: the dataset (share) to decrypt
        :return: decrypted data set
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.decrypt, data)

    def decrypt(self, data: npt.NDArray[np.object_]) -> npt.NDArray[np.int_]:
        """
        Method to decrypt a dataset using the initialized Paillier scheme

        :param data: the dataset to decrypt
        :return: a decrypted dataset
        """
        self._logger.info("Decrypting data...")
        decrypted_data: npt.NDArray[np.int_] = np.fromiter(
            self.paillier_scheme.decrypt_sequence(data.flatten()),
            dtype=np.int_,
            count=data.size,
        )
        decrypted_data = decrypted_data.reshape(data.shape)
        self._logger.info("Done decrypting data")
        return decrypted_data


def randomize_ndarray(arr: npt.NDArray[np.object_]) -> None:
    """
    Randomize all elements in an numpy array with ciphertexts.

    This function calls 'RandomizableCiphertext.randomize' 'arr.size' times,
    as expected. Note that this contrasts with the 'arr.size+1' calls made by
    'np.vectorize(lambda _: _.randomize())(arr)'.

    :param arr: array to be randomized
    """
    for data in np.nditer(arr, flags=["refs_ok"]):
        data[()].randomize()  # type: ignore[call-overload]
