"""
Module implements the base player class.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import reduce
from typing import List, Optional, Sequence, Tuple, Type, Union

import numpy as np
from lifelines.statistics import StatisticalResult
from mpyc.runtime import Party, mpc
from mpyc.sectypes import SecureFixedPoint
from scipy.stats import chi2

from tno.mpc.encryption_schemes.utils import FixedPoint
from tno.mpc.mpyc.matrix_inverse import matrix_inverse


@dataclass
class MPCProtocolMetadata:
    """
    This class contains some metadata that relates to the part of the protocol that relies on the MPyC framework.
    """

    data_parties: List[Party] = field(default_factory=list)
    _secfxp: Optional[Type[SecureFixedPoint]] = None

    @property
    def secfxp(self) -> Type[SecureFixedPoint]:
        """Type of SecureFixedPoint that we use throughout the protocol

        :return: SecureFixedPoint object
        :raise ValueError: raised when object has not yet been set
        """
        if self._secfxp is None:
            raise ValueError("secfxp is requested by has not been set")
        return self._secfxp

    @secfxp.setter
    def secfxp(self, secfxp: Type[SecureFixedPoint]) -> None:
        """Set the SecureFixedPoint object.

        :param secfxp: SecureFixedPoint object
        """
        self._secfxp = secfxp


class Player:
    """
    This class is the base class for players
    """

    def __init__(
        self,
        identifier: str,
        party_A: str = "Alice",
        party_B: str = "Bob",
        helper: str = "Helper",
    ) -> None:
        """
        Initializes player

        :param identifier: (unique) name of the player
        :param party_A: identifier of party Alice
        :param party_B: identifier of party Bob
        :param helper: identifier of the helper party
        """
        self._identifier = identifier
        self._party_A = party_A
        self._party_B = party_B
        self._helper = helper
        self.mpc_metadata = MPCProtocolMetadata()
        self._logger = logging.getLogger(self._identifier)
        self._logger.setLevel(logging.DEBUG)
        logging.basicConfig(
            format="%(asctime)s - %(name)s - " "%(levelname)s - %(message)s"
        )

        self._mpyc_data: Optional[np.ndarray[FixedPoint]] = None  # type: ignore
        self._mpyc_shares: Optional[Sequence[Sequence[SecureFixedPoint]]] = None
        self._mpyc_factors: Optional[np.ndarray[np.float64]] = None
        self._mpyc_factors_shares: Optional[Sequence[Sequence[SecureFixedPoint]]] = None
        self.statistic: Optional[StatisticalResult] = None

    async def start_protocol(self) -> None:
        """
        Starts and runs the protocol

        :raise NotImplementedError: raised when it is called but not implemented
        """
        raise NotImplementedError(
            "You need to implement the start_protocol function yourself."
        )

    @property
    def share(self) -> np.ndarray[FixedPoint]:  # type: ignore
        """
        Additive share used as an input for the Shamir secret sharing part of
        the protocol.

        :return: additive share
        :raise ValueError: raised when sharing is not set
        """
        if self._mpyc_data is None:
            raise ValueError("There is no share available")
        return self._mpyc_data

    @property
    def data_parties(self) -> Tuple[str, str]:
        """
        The identifiers of data parties involved in the protocol

        :return: a collection of data parties
        """
        return (self.party_A, self.party_B)

    @property
    def party_A(self) -> str:
        """
        The identifier of party Alice

        :return: the identifier of Alice
        """
        return self._party_A

    @property
    def party_B(self) -> str:
        """
        The identifier of party Bob

        :return: the identifier of Bob
        """
        return self._party_B

    @property
    def identifier(self) -> str:
        """
        Identifier of this instance

        :return: the identifier of this instance
        """
        return self._identifier

    @property
    def helper(self) -> str:
        """
        The identifier of the helper party

        :return: the identifier of the helper party
        """
        return self._helper

    async def run_mpyc(self) -> None:
        """
        Runs the Shamir secret sharing part of the protocol using the MPyC
        framework
        """
        async with mpc:
            assert len(mpc.parties) == 3, "Number of parties should be 3"
            await self._start_mpyc()
            await self.obtain_secret_sharings()
            await self.secure_multivariate_log_rank_test()

    async def obtain_secret_sharings(self) -> None:
        """
        Method to obtain Shamir secret sharings of the results of the
        homomorphic encryption part of the protocol.
        """
        await self.obtain_secret_sharing_additively_shared_database()
        await self.obtain_secret_sharing_factors()

    async def obtain_secret_sharing_additively_shared_database(self) -> None:
        """
        Method to obtain Shamir secret sharings of the additively shared results
        of the homomorphic encryption part of the protocol.
        """
        self._mpyc_shares = await self._reshare_dataframes(self._mpyc_data, typer=int)

    async def obtain_secret_sharing_factors(self) -> None:
        """
        Method to obtain Shamir secret sharings of factors needed in the
        secure computation of the log-rank statistic.

        These factors are pre-computed by Alice, leveraging her locally
        available information.
        """
        self._mpyc_factors_shares = await self._reshare_dataframes(
            self._mpyc_factors, typer=float
        )

    async def secure_multivariate_log_rank_test(self) -> None:
        """
        Computes the multivariate log rank statistics on the shares.

        :raise ValueError: raised when there are no shares available yet
        """
        self._logger.info("Computing log-rank statistics...")
        if self._mpyc_factors_shares is None or self._mpyc_shares is None:
            raise ValueError("Some shares are missing")
        dev_factors = self._mpyc_factors_shares[0]
        var_factors = self._mpyc_factors_shares[1]
        var_factors_2 = self._mpyc_factors_shares[2]
        deaths_array = self._mpyc_shares[: len(self._mpyc_shares) // 2]
        at_risk_array = self._mpyc_shares[len(self._mpyc_shares) // 2 :]
        self.statistic = await self._secure_multivariate_logrank_test(
            dev_factors, var_factors, var_factors_2, deaths_array, at_risk_array
        )
        self._logger.info("Done computing log-rank statistics")

    async def _start_mpyc(self) -> None:
        """
        Start MPyC and configure data parties.
        """
        self.mpc_metadata.secfxp = mpc.SecFxp(l=64, f=32)
        self._logger.info(f"In MPyC, you are player {mpc.pid}")
        self._logger.info(f"These are the other players: {mpc.parties}")
        data_parties = await mpc.transfer(self.identifier != self.helper)
        self.mpc_metadata.data_parties = [_ for _ in mpc.parties if data_parties[_.pid]]
        self._logger.info(
            f"These are the data_parties: {self.mpc_metadata.data_parties}"
        )

    async def _request_format_data(
        self, dataframe: Optional[Union[np.ndarray[FixedPoint], np.ndarray[np.float64]]]  # type: ignore
    ) -> Tuple[int, int]:
        """
        Method to determine the data format of the dataframe.
        The first data owner determines the data format (rows x cols).

        :param dataframe: the dataframe to determine data format of
        :raise ValueError: raised when data_parties metadata is not configured
        :return: the format of the dataframe (rows, columns)
        """
        if self.mpc_metadata.data_parties is None:
            raise ValueError("data parties metadata is not set (yet)")
        if mpc.pid == self.mpc_metadata.data_parties[0].pid:
            assert (
                dataframe is not None
            ), f"Party {mpc.pid} is missing some important data."
            rows, columns = (
                dataframe.shape[0],
                dataframe.shape[1],
            )
        else:
            rows, columns = 0, 0
        rows, columns = await mpc.transfer(
            (rows, columns), senders=self.mpc_metadata.data_parties[0].pid
        )
        return rows, columns

    async def _extract_and_convert_data_cols_from_np(
        self,
        dataframe: Optional[Union[np.ndarray[FixedPoint], np.ndarray[np.float64]]],  # type: ignore
        typer: type,
    ) -> List[List[Union[int, float]]]:
        """
        Extract the columns of a numpy dataframe and return them in a list.
        The elements of the returned lists are converted to type typer.

        Converting the type of the elements is particularly helpful if the
        elements are to be used with MPyC; MPyC SecureNumber objects expect
        type None, int, float, or finite field element. In particular, MPyC
        does not know how to deal with the types that numpy associates to the
        elements of a ndarray.

        :param dataframe: the dataframe to extract data from
        :param typer: the expected type in the dataframe
        :raise ValueError: raised when data_parties metadata is not configured
          or when data is missing
        :return: the extracted data columns
        """
        if self.mpc_metadata.data_parties is None:
            raise ValueError("data parties metadata is not set (yet)")
        rows, columns = await self._request_format_data(dataframe)
        if mpc.pid in [_.pid for _ in self.mpc_metadata.data_parties]:
            if dataframe is None:
                raise ValueError(f"Party {mpc.pid} is missing some important data.")
            return [
                list(map(typer, dataframe[..., i])) for i in range(dataframe.shape[1])
            ]
        else:
            return [[typer(0)] * rows for _ in range(columns)]

    async def _reshare_dataframes(
        self,
        dataframe: Optional[Union[np.ndarray[FixedPoint], np.ndarray[np.float64]]],  # type: ignore
        typer: type = float,
    ) -> List[List[SecureFixedPoint]]:
        """
        Re-share (in MPyC) the additively-shared inputs of the
        dataframe that is additively shared over the data owners.

        :param dataframe: the dataframe to reshare
        :param typer: the expected type in the dataframe
        :raise ValueError: raised when secfxp is not configured
        :return: Shamir secret sharing
        """
        if self.mpc_metadata.secfxp is None:
            raise ValueError("SecFxp is not configured (yet).")
        data_columns = await self._extract_and_convert_data_cols_from_np(
            dataframe, typer
        )
        shares = [
            reduce(
                mpc.vector_add,
                mpc.input(
                    list(map(lambda x: self.mpc_metadata.secfxp(x, integral=False), _))
                ),
            )
            for _ in data_columns
        ]
        return shares

    async def _secure_multivariate_logrank_test(
        self,
        dev_factors: Sequence[SecureFixedPoint],
        var_factors: Sequence[SecureFixedPoint],
        var_factors_2: Sequence[SecureFixedPoint],
        deaths_array: Sequence[Sequence[SecureFixedPoint]],
        at_risk_array: Sequence[Sequence[SecureFixedPoint]],
    ) -> StatisticalResult:
        """
        Computes the logrank statistics for the given input.

        :param dev_factors: The j-th element of this list indicates
            the quantity (total number of deaths) / (total number of
            patients at risk) at the j-th distinct event time.
        :param var_factors: The j-th element of this list indicates
            the quantity (total number of deaths) * (total number at risk -
            total number of deaths) / (total number at risk ** 2 *
            (total number at risk - 1)) at the j-th distinct event time.
        :param var_factors_2: The j-th element of this list indicates
            the quantity var_factors * (total number at risk) at the j-th
            distinct event time.
        :param deaths_array: list that contains one list per patient
            category. The j-th element of the i-th list indicates the
            number of patients in category i that died at the j-th distinct
            event time.
        :param at_risk_array: list that contains one list per patient
            category. The j-th element of the i-th list indicates the
            number of patients in category i that are at risk at the j-th
            distinct event time.
        :raise ValueError: raised when secfxp is not configured
        :return: logrank statistics.
        """
        if self.mpc_metadata.secfxp is None:
            raise ValueError("SecFxp is not configured (yet).")
        nr_groups = len(at_risk_array)
        nr_comparisons = nr_groups - 1
        secfxp = self.mpc_metadata.secfxp
        devs: List[SecureFixedPoint] = [secfxp(None)] * nr_comparisons
        pre_vars: List[List[SecureFixedPoint]] = [
            [secfxp(None)] * len(at_risk_array[0]) for _ in range(nr_comparisons)
        ]
        neg_pre_vars: List[List[SecureFixedPoint]] = [
            [secfxp(None)] * len(at_risk_array[0]) for _ in range(nr_comparisons)
        ]
        var_matrix: List[List[SecureFixedPoint]] = [
            [secfxp(None)] * nr_comparisons for _ in range(nr_comparisons)
        ]

        for i in range(nr_comparisons):
            pre_vars[i] = mpc.schur_prod(var_factors, at_risk_array[i])
            neg_pre_vars[i] = [-x for x in pre_vars[i]]

        for i, (deaths, at_risk) in enumerate(zip(deaths_array, at_risk_array)):
            if i == nr_comparisons:
                break
            # Compute deviations from expected number of deaths
            devs[i] = mpc.sum(
                mpc.vector_sub(deaths, mpc.schur_prod(dev_factors, at_risk))
            )

            # Compute variances
            var_matrix[i][i] = mpc.sum(
                mpc.schur_prod(mpc.vector_sub(var_factors_2, pre_vars[i]), at_risk)
            )
            for j in (jj for jj in range(nr_comparisons) if jj != i):
                var_matrix[i][j] = mpc.sum(mpc.schur_prod(neg_pre_vars[j], at_risk))

        # Compute chi-value
        if nr_groups == 2:
            chi_sec = devs[0] ** 2 / var_matrix[0][0]
        else:
            await mpc.barrier()
            vars_inv = matrix_inverse(var_matrix)
            await mpc.barrier()
            chi_sec = mpc.in_prod(mpc.matrix_prod([devs], vars_inv)[0], devs)
        chi = await mpc.output(chi_sec)
        p = chi2.sf(chi, len(at_risk_array) - 1)
        return StatisticalResult(
            p_value=p,
            test_statistic=chi,
            test_name="secure_multivariate_logrank_test",
            null_distribution="chi squared",
            degrees_of_freedom=len(at_risk_array) - 1,
        )
