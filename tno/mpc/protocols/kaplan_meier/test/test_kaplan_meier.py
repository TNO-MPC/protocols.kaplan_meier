"""
Tests that can be ran using pytest to test the kaplan-meier functionality
"""

import asyncio
from typing import Tuple, cast

import numpy as np
import pandas as pd
import pytest

from tno.mpc.communication import Pool
from tno.mpc.communication.test import (  # pylint: disable=unused-import
    fixture_pool_http_2p,
)
from tno.mpc.encryption_schemes.utils.fixed_point import FixedPoint
from tno.mpc.protocols.kaplan_meier import Alice, Bob
from tno.mpc.protocols.kaplan_meier.player import Player


@pytest.fixture(name="alice")
@pytest.mark.asyncio
async def fixture_alice(pool_http_2p: Tuple[Pool, Pool]) -> Alice:
    """
    Fixture that creates an instance of player Alice, initialized with test data

    :param pool_http_2p: communication pools
    :return: an instance of Alice
    """
    test_data = pd.DataFrame(
        {
            "time": [3, 5, 6, 8, 10, 14, 14, 18, 20, 22, 30, 30],
            "event": [1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1],
        }
    )
    return Alice(
        identifier="alice",
        party_A=list(pool_http_2p[1].pool_handlers)[0],
        party_B=list(pool_http_2p[0].pool_handlers)[0],
        data=test_data,
        pool=pool_http_2p[0],
    )


@pytest.fixture(name="bob")
@pytest.mark.asyncio
async def fixture_bob(pool_http_2p: Tuple[Pool, Pool]) -> Bob:
    """
    Fixture that creates an instance of player Bob, initialized with test data

    :param pool_http_2p: communication pools
    :return: an instance of Bob
    """
    test_data = pd.DataFrame(
        {
            "Group A": [1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0],
            "Group B": [0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1],
        }
    )
    return Bob(
        identifier="bob",
        party_A=list(pool_http_2p[1].pool_handlers)[0],
        party_B=list(pool_http_2p[0].pool_handlers)[0],
        data=test_data,
        pool=pool_http_2p[1],
    )


async def alice_protocol_paillier(alice: Alice) -> None:
    """
    Method that calls the protocol steps that Alice has to perform

    :param alice: the player that will perform the protocol steps
    """
    await asyncio.gather(
        *[
            alice.receive_paillier_scheme(),
            alice.receive_number_of_groups(),
        ]
    )
    alice.start_randomness_generation()
    await alice.receive_encrypted_group_data()
    alice.compute_hidden_table()
    alice.compute_factors()
    alice.re_randomise_ht()
    alice.generate_share()
    await alice.send_share()


async def bob_protocol_paillier(bob: Bob) -> None:
    """
    Method that calls the protocol steps that Bob has to perform

    :param bob: the player that will perform the protocol steps
    """
    await bob.send_number_of_groups()
    loop = asyncio.get_event_loop()
    _, _, bob.encrypted_data = await asyncio.gather(
        bob.send_paillier_scheme(),
        bob.send_number_of_groups(),
        loop.run_in_executor(None, bob.encrypt, bob.data),
    )
    await bob.send_encrypted_data()
    await bob.receive_share()


@pytest.mark.asyncio
async def test_protocol_paillier(alice: Alice, bob: Bob) -> None:
    """
    Tests the homomorphic encryption (using Paillier) part of the protocol

    :param alice: player alice in the protocol
    :param bob: player bob in the protocol
    """
    await asyncio.gather(
        *[
            alice_protocol_paillier(alice),
            bob_protocol_paillier(bob),
        ]
    )

    correct_outcome = np.array(
        [
            [1, 0, 5, 7],
            [1, 0, 4, 7],
            [0, 1, 3, 7],
            [0, 2, 2, 5],
            [1, 0, 2, 2],
            [1, 1, 1, 1],
        ]
    )
    np.testing.assert_array_equal(alice.share + bob.share, correct_outcome)


@pytest.mark.asyncio
async def test_protocol_mpyc() -> None:
    """
    Tests the Shamir secret sharing (using MPyC) part of the protocol
    """
    player = Player("Test_player")

    player._mpyc_data = cast(
        "np.ndarray[FixedPoint]",  # type: ignore
        np.array(
            [
                list(map(FixedPoint.fxp, dat))
                for dat in [
                    [1, 0, 5, 7],
                    [1, 0, 4, 7],
                    [0, 1, 3, 7],
                    [0, 2, 2, 5],
                    [1, 0, 2, 2],
                    [1, 1, 1, 1],
                ]
            ]
        ),
    )
    player._mpyc_factors = np.array(
        [
            [0.08333333333333333, 0.006944444444444445, 0.08333333333333334],
            [0.09090909090909091, 0.008264462809917356, 0.09090909090909091],
            [0.1, 0.01, 0.1],
            [0.2857142857142857, 0.034013605442176874, 0.2380952380952381],
            [0.25, 0.0625, 0.25],
            [1.0, 0.0, 0.0],
        ]
    )
    from mpyc.runtime import mpc

    async with mpc:
        await player._start_mpyc()
        await player.obtain_secret_sharings()
        await player.secure_multivariate_log_rank_test()
    assert player.statistic is not None
    assert player.statistic.test_statistic == pytest.approx(0.5645388)
    assert player.statistic.p_value == pytest.approx(0.4524372)
