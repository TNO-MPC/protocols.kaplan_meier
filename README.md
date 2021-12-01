# TNO MPC Lab - Protocols - Kaplan-Meier

The TNO MPC lab consists of generic software components, procedures, and functionalities developed and maintained on a regular basis to facilitate and aid in the development of MPC solutions. The lab is a cross-project initiative allowing us to integrate and reuse previously developed MPC functionalities to boost the development of new protocols and solutions.

The package tno.mpc.protocols.kaplan_meier is part of the TNO Python Toolbox.

This MPC protocol securely computes the log-rank statistics associated with the Kaplan-Meier estimator on vertically partitioned data. It has been developed within the [CONVINCED](https://www.tno.nl/en/tno-insights/articles/privacy-friendly-data-technology-expands-oncology-research-opportunities/) project, a research collaboration between TNO and IKNL, and the project SELECTED. These projects were partly funded by TNOs Appl.AI and the Netherlands AI Coalition. In the future, this will be integrated in IKNLs open source [Vantage6](https://vantage6.ai) infrastructure.

*Limitations in (end-)use: the content of this software package may solely be used for applications that comply with international export control laws.*  
*This implementation of cryptographic software has not been audited. Use at your own risk.*

## Documentation

Documentation of the tno.mpc.protocols.kaplan_meier package can be found [here](https://docs.mpc.tno.nl/protocols/kaplan_meier/0.2.0).
In [ERCIM News 126 (July 2021)](https://ercim-news.ercim.eu/en126/special/oncological-research-on-distributed-patient-data-privacy-can-be-preserved), we presented some extra context.

## Install

Easily install the tno.mpc.protocols.kaplan_meier package using pip:
```console
$ python -m pip install tno.mpc.protocols.kaplan_meier
```

### Note:
A significant performance improvement can be achieved by installing the GMPY2 library.
```console
$ python -m pip install 'tno.mpc.protocols.kaplan_meier[gmpy]'
```

If you wish to run the tests you can use:
```console
$ python -m pip install 'tno.mpc.protocols.kaplan_meier[tests]'
```

## Protocol description
A more elaborate protocol description can be found in [CONVINCED -- Enabling privacy-preserving survival analyses using Multi-Party Computation](https://repository.tno.nl/islandora/object/uuid:1c4885d6-8cf3-4443-b952-e887e1b41207).

<figure>
  <img src="https://raw.githubusercontent.com/TNO-MPC/protocols.kaplan_meier/main/assets/kaplan-meier-overview.svg" width=100% alt="Kaplan-Meier High Level Overview"/>
  <figcaption>

__Figure 1.__ _The protocol to securely compute the log-rank statistic for vertically-partitioned data. One party (Blue) owns data on patient groups, the other party (Orange) owns data on event times (did the patient experience an event ‘1’ or not ‘0’, and when did this occur). Protocol outline: Blue encrypts its data using additive homomorphic encryption and the encrypted data is sent to Orange. Orange is able to securely, without decryption, split its data in the patient groups specified by Blue (1) using the additive homomorphic properties of the encryptions. Orange performs some preparatory, local, computations (2) and with the help of Blue secret-shares the data (3) between Blue, Orange and Purple, where Purple is introduced for efficiency purposes. All parties together securely compute the log-rank statistic associated with the (never revealed) Kaplan-Meier curves (4) and only reveal the final statistical result (5)._
  </figcaption>
</figure>

## Usage

The protocol is asymmetric. To run the protocol you need to run three separate instances.

>`example_usage.py`
>```python
>"""
>Example usage for performing Kaplan-Meier analysis
>Run three separate instances e.g.,
>    $ ./script/example_usage.py -M3 -I0 -p Alice
>    $ ./script/example_usage.py -M3 -I1 -p Bob
>    $ ./script/example_usage.py -M3 -I2 -p Helper
>All but the last argument are passed to MPyC.
>"""
>
>import argparse
>import asyncio
>import lifelines
>import pandas as pd
>
>from tno.mpc.communication import Pool
>
>from tno.mpc.protocols.kaplan_meier import Alice, Bob, Helper
>
>
>def parse_args():
>    parser = argparse.ArgumentParser()
>    parser.add_argument(
>        "-p", "--player", help="Name of the sending player", type=str, required=True
>    )
>    args = parser.parse_args()
>    return args
>
>
>async def main(player_instance):
>    await player_instance.start_protocol()
>
>
>if __name__ == "__main__":
>    # Parse arguments and acquire configuration parameters
>    args = parse_args()
>    player = args.player
>    parties = {
>        "Alice": {"address": "127.0.0.1", "port": 8080},
>        "Bob": {"address": "127.0.0.1", "port": 8081},
>    }
>
>    test_data = pd.DataFrame(
>        {
>            "time": [3, 5, 6, 8, 10, 14, 14, 18, 20, 22, 30, 30],
>            "event": [1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1],
>            "Group A": [1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0],
>            "Group B": [0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1],
>        }
>    )
>
>    if player in parties.keys():
>        port = parties[player]["port"]
>        del parties[player]
>
>        pool = Pool()
>        pool.add_http_server(port=port)
>        for name, party in parties.items():
>            assert "address" in party
>            pool.add_http_client(
>                name, party["address"], port=party["port"] if "port" in party else 80
>            )  # default port=80
>        if player == "Alice":
>            event_times = test_data[["time", "event"]]
>            player_instance = Alice(
>                identifier=player,
>                data=event_times,
>                pool=pool,
>            )
>        elif player == "Bob":
>            groups = test_data[["Group A", "Group B"]]
>            player_instance = Bob(
>                identifier=player,
>                data=groups,
>                pool=pool,
>            )
>    elif player == "Helper":
>        player_instance = Helper(player)
>    else:
>        raise ValueError(f"Unknown player was provided: '{player}'")
>
>    loop = asyncio.get_event_loop()
>    loop.run_until_complete(main(player_instance))
>
>    print("-" * 32)
>    print(player_instance.statistic)
>    print("-" * 32)
>
>    # Validate results
>    event_times = test_data[["time", "event"]]
>    groups = test_data[["Group A", "Group B"]]
>    print(
>        lifelines.statistics.multivariate_logrank_test(
>            event_times["time"], groups["Group B"], event_times["event"]
>        )
>    )
>    print("-" * 32)
>```
