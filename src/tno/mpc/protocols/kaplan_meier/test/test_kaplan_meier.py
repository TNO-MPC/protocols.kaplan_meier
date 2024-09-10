"""
Tests that can be ran using pytest to test the kaplan-meier functionality
"""

from __future__ import annotations

import asyncio
from typing import AsyncGenerator

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
import pytest_asyncio
from mpyc.runtime import mpc

from tno.mpc.communication import Pool
from tno.mpc.communication.test.pool_fixtures_http import (
    finish,
    generate_http_test_pools,
)
from tno.mpc.encryption_schemes.utils.fixed_point import FixedPoint

from tno.mpc.protocols.kaplan_meier import Alice, Bob
from tno.mpc.protocols.kaplan_meier.player import Player


@pytest_asyncio.fixture(name="pool_http_2p", scope="function")
async def fixture_pool_http_2p() -> AsyncGenerator[tuple[Pool, ...]]:
    """
    Creates a collection of 2 communication pools

    :return: a collection of communication pools
    """
    pools = generate_http_test_pools(2)
    yield await pools.asend(None)
    await finish(pools)


@pytest_asyncio.fixture(name="alice")
async def fixture_alice(pool_http_2p: tuple[Pool, Pool]) -> Alice:
    """
    Fixture that creates an instance of player Alice, initialized with test data

    :param pool_http_2p: communication pools
    :return: an instance of Alice
    """
    test_data = pd.DataFrame(  # type: ignore[attr-defined]
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


@pytest_asyncio.fixture(name="bob")
async def fixture_bob(pool_http_2p: tuple[Pool, Pool]) -> Bob:
    """
    Fixture that creates an instance of player Bob, initialized with test data

    :param pool_http_2p: communication pools
    :return: an instance of Bob
    """
    test_data = pd.DataFrame(  # type: ignore[attr-defined]
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
    alice.start_randomness_generation(
        amount=alice.nr_unique_event_times * 2 * (alice.nr_groups - 1)
    )
    await alice.receive_encrypted_group_data()
    alice.compute_kaplan_meier_minimal_statistics()
    alice.compute_logrank_factors()
    alice.generate_additive_share()
    await alice.send_additive_share()
    alice.shut_down_paillier()


async def bob_protocol_paillier(bob: Bob) -> None:
    """
    Method that calls the protocol steps that Bob has to perform

    :param bob: the player that will perform the protocol steps
    """
    bob.start_randomness_generation(amount=bob.data.shape[0] * (bob.nr_groups - 1))
    loop = asyncio.get_event_loop()
    _, _, bob.encrypted_data = await asyncio.gather(
        bob.send_paillier_scheme(),
        bob.send_number_of_groups(),
        loop.run_in_executor(None, bob.encrypt, bob.data.iloc[:, :-1]),
    )
    await bob.send_encrypted_group_data()
    # Alice and Bob share a Paillier scheme. Alice uses it last, so she gets to shut it down.
    await bob.receive_additive_share()


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

    correct_outcome: npt.NDArray[np.int_] = np.array(
        [
            [1, 5],
            [1, 4],
            [0, 3],
            [0, 2],
            [1, 2],
            [1, 1],
        ]
    )
    np.testing.assert_array_equal(
        alice.additive_share + bob.additive_share, correct_outcome
    )


@pytest.mark.asyncio
async def test_protocol_mpyc() -> None:
    """
    Tests the Shamir secret sharing (using MPyC) part of the protocol
    """
    player = Player("Test_player")

    player._additive_share = np.array(
        [
            list(map(FixedPoint.fxp, dat))
            for dat in [
                [1, 5],
                [1, 4],
                [0, 3],
                [0, 2],
                [1, 2],
                [1, 1],
            ]
        ]
    )
    player._logrank_computation_factors = np.array(
        [
            [0.08333333333333333, 0.006944444444444445, 0.08333333333333334],
            [0.09090909090909091, 0.008264462809917356, 0.09090909090909091],
            [0.1, 0.01, 0.1],
            [0.2857142857142857, 0.034013605442176874, 0.2380952380952381],
            [0.25, 0.0625, 0.25],
            [1.0, 0.0, 0.0],
        ]
    )

    async with mpc:
        await player._start_mpyc()
        await player.obtain_secret_shared_logrank_input()
        await player.secure_multivariate_log_rank_test()
    assert player.statistic is not None
    assert player.statistic.test_statistic == pytest.approx(0.5645388)
    assert player.statistic.p_value == pytest.approx(0.4524372)
