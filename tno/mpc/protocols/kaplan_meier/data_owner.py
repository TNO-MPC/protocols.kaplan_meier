"""
Module implements the data owners Alice and Bob.
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional, SupportsInt, Union, cast

import numpy as np
import pandas as pd

from tno.mpc.communication import Pool
from tno.mpc.encryption_schemes.paillier import Paillier, PaillierCiphertext
from tno.mpc.encryption_schemes.utils.fixed_point import FixedPoint

from .player import Player


class DataOwner(Player):
    """
    Data owner in the MPC protocol
    """

    def __init__(
        self,
        data: pd.DataFrame,
        pool: Pool,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Initializes data owner

        :param data: the data to use for this data owner
        :param pool: a communication pool
        :param args: arguments to pass on to base class
        :param kwargs: keyword arguments to pass on to base class
        """
        super().__init__(*args, **kwargs)
        self._paillier_scheme: Optional[Paillier] = None
        self._data = data
        self.pool = pool

    async def receive_message(self, party: str, msg_id: Optional[str] = None) -> Any:
        """
        Receives a message from a party (belonging to an optional message
        identifier)

        :param party: the party to receive a message from
        :param msg_id: the message id
        :return: the received message
        """
        return await self.pool.recv(party, msg_id=msg_id)

    async def send_message(
        self, receiver: str, message: Any, msg_id: Optional[str] = None
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
        return self.data.shape[0]

    @property
    def groups(self) -> int:
        """
        Number of groups in the loaded datasets

        :return: number of groups
        :raise NotImplementedError: raised when not implemented
        """
        raise NotImplementedError()

    @property
    def data(self) -> Union[pd.DataFrame, np.ndarray[np.int32]]:
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

    def stop_randomness_generation(self) -> None:
        """
        Stop generation of randomness.
        """
        self.paillier_scheme.randomness.shut_down()

    def encrypt(self, data: np.ndarray[np.float64]) -> np.ndarray[PaillierCiphertext]:  # type: ignore[type-var]
        """
        Method to encrypt a dataset using the initialized Paillier scheme

        :param data: the dataset to encrypt
        :return: an encrypted dataset
        """
        self._logger.info("Encrypting data...")
        encrypted_data: np.ndarray[PaillierCiphertext] = np.vectorize(self.paillier_scheme.encrypt)(data)  # type: ignore[attr-defined, type-var]
        self._logger.info("Done encrypting data")
        return encrypted_data

    def decrypt(self, data: np.ndarray[PaillierCiphertext]) -> np.ndarray[Any]:  # type: ignore[type-var]
        """
        Method to decrypt a dataset using the initialized Paillier scheme

        :param data: the dataset to decrypt
        :return: a decrypted dataset
        """
        self._logger.info("Decrypting data...")
        decrypted_data: np.ndarray[Any] = np.vectorize(self.paillier_scheme.decrypt)(data)  # type: ignore[attr-defined]
        self._logger.info("Done decrypting data")
        return decrypted_data


class Alice(DataOwner):
    """
    Alice player in the MPC protocol
    """

    def __init__(self, *args: Any, nr_of_threads: int = 4, **kwargs: Any) -> None:
        """
        Initializes player Alice

        :param nr_of_threads: the number of threads to use for randomness
            generation
        :param args: arguments to pass on to base class
        :param kwargs: keyword arguments to pass on to base class
        """
        super().__init__(*args, **kwargs)
        self.nr_of_threads = nr_of_threads
        self._encrypted_group_data_: Optional[np.ndarray[PaillierCiphertext]] = None  # type: ignore[type-var] # pylint: disable=unsubscriptable-object
        self._hidden_table: Optional[np.ndarray[PaillierCiphertext]] = None  # type: ignore[type-var] # pylint: disable=unsubscriptable-object
        self._plain_table: Optional[
            np.ndarray[np.int32]  # pylint: disable=unsubscriptable-object
        ] = None
        self._indices_events: pd.Series[bool] = pd.Series(None, dtype=bool)  # type: ignore[call-arg, type-var] # pylint: disable=unsubscriptable-object
        self._mask_ht = None
        self._number_of_groups = None

    @property
    def groups(self) -> int:
        """
        Number of groups in the datasets

        :return: number of groups
        :raise ValueError: raised when number of groups is not available (yet)
        """
        if self._number_of_groups is None:
            raise ValueError("Number of groups is not available yet")
        return self._number_of_groups

    @property
    def rows_in_hidden_table(self) -> int:
        """
        Number of rows in the hidden table. Equals number of unique event
        times (ignoring censorings).

        :return: number of rows in the hidden table
        """
        return self.data["time"].loc[self.data["event"].astype(bool)].nunique()

    @property
    def cols_in_hidden_table(self) -> int:
        """
        Number of columns in the hidden table. Two columns per group; one for
        the number of events on a given event time and one for the number of
        people at risk at that time.

        :return: number of columns in the hidden table
        """
        return 2 * self.groups

    @property
    def _encrypted_group_data(self) -> np.ndarray[PaillierCiphertext]:  # type: ignore[type-var] # pylint: disable=unsubscriptable-object
        """
        Encrypted group data

        :return: the encrypted group data
        :raise ValueError: raised when the encrypted group data is not yet available.
        """
        if self._encrypted_group_data_ is None:
            raise ValueError("Alice is missing some important data.")
        return self._encrypted_group_data_

    @_encrypted_group_data.setter
    def _encrypted_group_data(self, data: np.ndarray[PaillierCiphertext]) -> None:  # type: ignore[type-var] # pylint: disable=unsubscriptable-object
        self._encrypted_group_data_ = data

    @property
    def hidden_table(self) -> np.ndarray[PaillierCiphertext]:  # type: ignore[type-var] # pylint: disable=unsubscriptable-object
        """
        Hidden table

        :return: the constructed hidden table
        :raise ValueError: raised when hidden table is not yet available.
        """
        if self._hidden_table is None:
            raise ValueError("Hidden table is not set yet.")
        return self._hidden_table

    @property
    def plain_table(
        self,
    ) -> np.ndarray[np.int32]:  # pylint: disable=unsubscriptable-object
        """
        Plain table

        :return: plaintext result of some computation in table format
        :raise ValueError: raised when plain table is not yet available.
        """
        if self._plain_table is None:
            raise ValueError("Plain table is not set yet.")
        return self._plain_table

    async def start_protocol(self) -> None:
        """
        Starts and runs the protocol
        """
        await asyncio.gather(
            *[
                self.receive_paillier_scheme(),
                self.receive_number_of_groups(),
            ]
        )
        self.start_randomness_generation()
        await self.receive_encrypted_group_data()
        self.compute_hidden_table()
        self.compute_factors()
        self.re_randomise_ht()
        self.stop_randomness_generation()
        self.generate_share()
        await self.send_share()
        await self.run_mpyc()

    def start_randomness_generation(self) -> None:
        """
        Kicks off the randomness generation. This boosts performance.
        In particular will this decrease the total runtime (as data owners can
        already generate randomness before they need it).
        """
        # Total required randomness:
        # - once for re-randomizing the hidden table
        # - once for making an additive mask of the hidden table
        self.paillier_scheme.initialize_randomness(
            nr_of_threads=self.nr_of_threads,
            start_generation=True,
            max_size=2 * self.cols_in_hidden_table * self.rows_in_hidden_table,
            total=2 * self.cols_in_hidden_table * self.rows_in_hidden_table,
        )

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
        self._number_of_groups = await self.receive_message(
            self.party_B, msg_id="number_of_groups"
        )

    async def receive_encrypted_group_data(self) -> None:
        """
        Method to receive the encrypted group data from party Bob.
        """
        self._encrypted_group_data = await self.receive_message(
            self.party_B, msg_id="encrypted_group_data"
        )

    def compute_hidden_table(self) -> None:
        """
        Method to compute the hidden table of the protocol.
        """
        self._logger.info("Computing Kaplan-Meier features from encrypted data...")
        self._sort_data()
        self._determine_events()
        self._remove_censored_and_duplicates()
        self._logger.info("Done computing Kaplan-Meier features from encrypted data")

    def _sort_data(self) -> None:
        """
        Sort data by time (ascending), then by event (descending). To
        obtain correct result, first sort all data by event, then
        sort all data by time.

        :raise AttributeError: raised when data is not a pandas dataframe
        """
        if not isinstance(self.data, pd.DataFrame):
            raise AttributeError("Data is not a pandas dataframe")
        self._data = self.data.sort_values(
            by=["time", "event"], ascending=[True, False]
        )
        self._encrypted_group_data = self._encrypted_group_data[
            cast(slice, self.data.index)
        ]

    def _determine_events(self) -> None:
        """
        Determine the indices of the events.
        """
        self._indices_events = cast("pd.Series[bool]", self.data["event"] == 1)

    def _remove_censored_and_duplicates(self) -> None:
        """
        Removes censored data and processes duplicates.

        :raise ValueError: raised when event indices are not determined
        """
        diff: np.ndarray[np.int32] = np.diff(self.data[self._indices_events]["time"])  # type: ignore[arg-type, call-overload] # pylint: disable=unsubscriptable-object
        add = (np.nonzero(diff)[0] + 1).astype(np.int32)
        add = np.insert(add, 0, np.int32(0))

        grouped_data = np.c_[  # type: ignore[attr-defined]
            np.ones((self.data.shape[0], 1)), self._encrypted_group_data
        ]

        # Compute the result columns
        # Summing
        exposed_cols = grouped_data[::-1].cumsum(axis=0)[::-1]
        type_cols = np.add.reduceat(grouped_data[self._indices_events], add)  # type: ignore[attr-defined]
        # Removing
        if self._indices_events is None:
            raise ValueError("Indices of events are not determined (yet).")
        exposed_cols = exposed_cols[
            self._indices_events & ~self.data["time"].duplicated(keep="first")
        ]

        self._hidden_table = np.c_[type_cols[:, 1:], exposed_cols[:, 1:]]  # type: ignore[attr-defined]
        self._plain_table = np.c_[type_cols[:, 0], exposed_cols[:, 0]]  # type: ignore[attr-defined]

    def compute_factors(self) -> None:
        """Pre-computes several factors for in the computation of the log-
        rank statistic, leveraging information known by Alice only.

        Computes the following factors: dev_factors, var_factors,
        var_factors_2. These factors satisfy the following relations:

          Expected number of deaths in group i =
            dev_factors[i] * at_risk_group[i]

          Variance of deaths in group i =
            (var_factors_2[i] - var_factors[i] * at_risk_group[i])
            * at_risk_group[i]
        """
        at_risk_total = self.plain_table[:, 1]
        deaths_total = self.plain_table[:, 0]

        # Expected number of deaths(E)  =
        #   (deaths_total / at_risk_total) *
        #     [at_risk_group]
        dev_factors = deaths_total / at_risk_total

        # Variance =
        #   deaths_total * (at_risk_total - deaths_total) /
        #     (at_risk_total**2 * (at_risk_total - 1)) *
        #       [at_risk_group] * (at_risk_total - [at_risk_group])
        # Note here that the denominator equals zero if at_risk_total
        # equals one, which is only possible in the last event time.
        # The variance should then also equal zero. Since
        # deaths_total is always strictly positive, we find that
        # necessarily at_risk_total - deaths_total = 0 if
        # at_risk_total = 0. Therefore, the following produces the
        # correct variance for every event time without dividing by
        # zero.
        var_factors = deaths_total * (at_risk_total - deaths_total) / at_risk_total ** 2
        var_ind = at_risk_total != 1
        var_factors[var_ind] /= at_risk_total[var_ind] - 1
        var_factors_2 = var_factors * at_risk_total
        self._mpyc_factors = np.c_[dev_factors, var_factors, var_factors_2]  # type: ignore[attr-defined]

    def generate_share(self) -> None:
        """
        Generates additive secret shares.
        """
        self._mpyc_data: np.ndarray[FixedPoint] = np.vectorize(lambda _: self.signed_randomness())(np.ndarray(self.hidden_table.shape))  # type: ignore[attr-defined, type-var] # pylint: disable=unsubscriptable-object
        self._logger.info("Generated share")

    def mask_hidden_table(
        self,
    ) -> np.ndarray[np.float64]:  # pylint: disable=unsubscriptable-object
        """
        Masks the hidden table.

        :return: a masked hidden table
        """
        return cast("np.ndarray[np.float64]", self.hidden_table - self.share)

    async def send_share(self) -> None:
        """
        Sends additive secret share to party Bob.
        """
        loop = asyncio.get_event_loop()
        masked_hidden_table = await loop.run_in_executor(None, self.mask_hidden_table)
        await self.send_message(self.party_B, masked_hidden_table, msg_id="share")
        self._logger.info("Sent share")

    def signed_randomness(self) -> SupportsInt:
        """
        Returns a signed random plaintext value.

        :return: signed random plaintext value
        """
        return self.paillier_scheme.random_plaintext()

    @staticmethod
    def re_randomize(ciphertext: PaillierCiphertext) -> None:
        """
        Re-randomises a ciphertext

        :param ciphertext: ciphertext to randomize
        """
        ciphertext.randomize()

    def re_randomise_ht(self) -> None:
        """
        Re-randomises the hidden table
        """
        np.vectorize(self.re_randomize)(self.hidden_table)  # type: ignore[attr-defined]


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
        self.encrypted_data: Optional[np.ndarray[PaillierCiphertext]] = None  # type: ignore[type-var] # pylint: disable=unsubscriptable-object
        self._hidden_table: Optional[np.ndarray[PaillierCiphertext]] = None  # type: ignore[type-var] # pylint: disable=unsubscriptable-object

    @property
    def groups(self) -> int:
        """
        Number of groups in the loaded dataset

        :return: number of groups
        """
        return self._data.shape[1]

    async def start_protocol(self) -> None:
        """
        Starts and runs the protocol
        """
        await self.send_number_of_groups()
        loop = asyncio.get_event_loop()
        _, _, self.encrypted_data = await asyncio.gather(
            self.send_paillier_scheme(),
            self.send_number_of_groups(),
            loop.run_in_executor(None, self.encrypt, self.data),
        )
        self.stop_randomness_generation()
        await self.send_encrypted_data()
        await self.receive_share()
        await self.run_mpyc()

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
        await self.send_message(self.party_A, self.groups, msg_id="number_of_groups")

    async def send_encrypted_data(self) -> None:
        """
        Sends the encrypted dataset to party Alice.
        """
        await self.send_message(
            self.party_A, self.encrypted_data, msg_id="encrypted_group_data"
        )

    async def receive_share(self) -> None:
        """
        Receive additive secret share produced by party Alice.
        """
        encrypted_share = await self.receive_message(self.party_A, msg_id="share")
        self._mpyc_data = cast(
            "np.ndarray[FixedPoint]", await self.decrypt_share(encrypted_share)  # type: ignore[type-var]
        )
        self._mpyc_factors = np.zeros((len(self._mpyc_data), 3), dtype=np.float64)

    async def decrypt_share(self, data: np.ndarray[PaillierCiphertext]) -> Any:  # type: ignore[type-var] # pylint: disable=unsubscriptable-object
        """
        Decrypt share

        :param data: the dataset (share) to decrypt
        :return: decrypted data set
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.decrypt, data)
