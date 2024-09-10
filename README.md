# TNO PET Lab - secure Multi-Party Computation (MPC) - Protocols - Kaplan Meier

An implementation of the Kaplan-Meier Estimator.
Details about the protocol can be found here: [CONVINCED -- Enabling privacy-preserving survival analyses using Multi-Party Computation](https://repository.tno.nl/islandora/object/uuid:1c4885d6-8cf3-4443-b952-e887e1b41207).

### PET Lab

The TNO PET Lab consists of generic software components, procedures, and functionalities developed and maintained on a regular basis to facilitate and aid in the development of PET solutions. The lab is a cross-project initiative allowing us to integrate and reuse previously developed PET functionalities to boost the development of new protocols and solutions.

The package `tno.mpc.protocols.kaplan_meier` is part of the [TNO Python Toolbox](https://github.com/TNO-PET).

_Limitations in (end-)use: the content of this software package may solely be used for applications that comply with international export control laws._  
_This implementation of cryptographic software has not been audited. Use at your own risk._

## Documentation

Documentation of the `tno.mpc.protocols.kaplan_meier` package can be found
[here](https://docs.pet.tno.nl/mpc/protocols/kaplan_meier/1.0.4).

## Install

Easily install the `tno.mpc.protocols.kaplan_meier` package using `pip`:

```console
$ python -m pip install tno.mpc.protocols.kaplan_meier
```

_Note:_ If you are cloning the repository and wish to edit the source code, be
sure to install the package in editable mode:

```console
$ python -m pip install -e 'tno.mpc.protocols.kaplan_meier'
```

If you wish to run the tests you can use:

```console
$ python -m pip install 'tno.mpc.protocols.kaplan_meier[tests]'
```
_Note:_ A significant performance improvement can be achieved by installing the GMPY2 library.

```console
$ python -m pip install 'tno.mpc.protocols.kaplan_meier[gmpy]'
```

## Protocol description

A more elaborate protocol description can be found in [CONVINCED -- Enabling privacy-preserving survival analyses using Multi-Party Computation](https://repository.tno.nl/islandora/object/uuid:1c4885d6-8cf3-4443-b952-e887e1b41207).
In [ERCIM News 126 (July 2021)](https://ercim-news.ercim.eu/en126/special/oncological-research-on-distributed-patient-data-privacy-can-be-preserved), we presented some extra context.

<figure>
  <img src="https://raw.githubusercontent.com/TNO-MPC/protocols.kaplan_meier/main/assets/kaplan-meier-overview.svg" width=100% alt="Kaplan-Meier High Level Overview"/>
  <figcaption>

**Figure 1.** _The protocol to securely compute the log-rank statistic for vertically-partitioned data. One party (Blue) owns data on patient groups, the other party (Orange) owns data on event times (did the patient experience an event ‘1’ or not ‘0’, and when did this occur). Protocol outline: Blue encrypts its data using additive homomorphic encryption and the encrypted data is sent to Orange. Orange is able to securely, without decryption, split its data in the patient groups specified by Blue (1) using the additive homomorphic properties of the encryptions. Orange performs some preparatory, local, computations (2) and with the help of Blue secret-shares the data (3) between Blue, Orange and Purple, where Purple is introduced for efficiency purposes. All parties together securely compute the log-rank statistic associated with the (never revealed) Kaplan-Meier curves (4) and only reveal the final statistical result (5)._

  </figcaption>
</figure>

## Usage

The protocol is asymmetric. To run the protocol you need to run three separate instances.

`scripts/example_usage.py`

```py
"""
Example usage for performing Kaplan-Meier analysis
Run three separate instances e.g.,
    $ python ./scripts/example_usage.py -M3 -I0 -p alice
    $ python ./scripts/example_usage.py -M3 -I1 -p bob
    $ python ./scripts/example_usage.py -M3 -I2 -p helper
All but the last argument are passed to MPyC.
"""

from __future__ import annotations

import argparse
import asyncio
from enum import Enum

import lifelines
import pandas as pd

from tno.mpc.communication import Pool
from tno.mpc.protocols.kaplan_meier import Alice, Bob, Helper


class KnownPlayers(Enum):
    ALICE = "alice"
    BOB = "bob"
    HELPER = "helper"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--player",
        help="Name of the sending player",
        type=str,
        required=True,
        choices=list(p.value.lower() for p in KnownPlayers),
    )
    args = parser.parse_args()
    return args


async def main(player_instance: Alice | Bob | Helper) -> None:
    await player_instance.run_protocol()


if __name__ == "__main__":
    # Parse arguments and acquire configuration parameters
    args = parse_args()
    player = KnownPlayers(args.player)
    player_config: dict[KnownPlayers, dict[str, str]] = {
        KnownPlayers.ALICE: {"address": "127.0.0.1", "port": "8080"},
        KnownPlayers.BOB: {"address": "127.0.0.1", "port": "8081"},
    }

    test_data = pd.DataFrame(  # type: ignore[attr-defined]
        {
            "time": [3, 5, 6, 8, 10, 14, 14, 18, 20, 22, 30, 30],
            "event": [1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1],
            "Group A": [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            "Group B": [0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1],
            "Group C": [0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0],
        }
    )

    player_instance: Alice | Bob | Helper
    if player in player_config.keys():
        pool = Pool()
        pool.add_http_server(port=int(player_config[player]["port"]))

        for player_, config in player_config.items():
            if player_ is player:
                continue
            pool.add_http_client(
                player_.value,
                config["address"],
                port=int(config["port"]) if "port" in config else 80,
            )  # default port=80
        if player is KnownPlayers.ALICE:
            event_times = test_data[["time", "event"]]
            player_instance = Alice(
                identifier=player.value,
                data=event_times,
                pool=pool,
            )
        elif player is KnownPlayers.BOB:
            groups = test_data[["Group A", "Group B", "Group C"]]
            player_instance = Bob(
                identifier=player.value,
                data=groups,
                pool=pool,
            )
    elif player is KnownPlayers.HELPER:
        player_instance = Helper(player.value)

    loop = asyncio.get_event_loop()
    loop.run_until_complete(main(player_instance))

    print("-" * 32)
    print(player_instance.statistic)
    print("-" * 32)

    # Validate results
    event_times = test_data[["time", "event"]]
    groups = (
        test_data["Group B"].to_numpy() + 2 * test_data["Group C"].to_numpy()
    )  # convert from binary to categorical
    print(
        lifelines.statistics.multivariate_logrank_test(
            event_times["time"],
            groups,
            event_times["event"],
        )
    )
    print("-" * 32)
```
