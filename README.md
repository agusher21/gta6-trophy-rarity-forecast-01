# gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/agusher21/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/agusher21/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts each Grand Theft Auto VI trophy's Exophase earn rate one week after launch, as five quantiles, from the page's refresh history — in counts, not rates. Exophase's page is frozen between its own refreshes and jumps at them: a refresh brings a wave of newly tracked players who hold far fewer trophies, so every rate except a 100% one falls. This program forecasts players tracked and per-trophy earned counts separately (hold rates + progression fitted per trophy over the refreshes, with a documented launch prior when the history is a single frozen page), then takes earned / players as the rate, with a spread from a one-step-ahead backtest inside the history.

## Run it

```sh
python3 main.py
```

Reads every `*.in` case in `/input` and writes `<id>.out` to `/output` (override both with `MAKEFUN_INPUT_DIR` and `MAKEFUN_OUTPUT_DIR`). Python 3.13, standard library only, no network, deterministic, ~0.1s per case.

---

<sub>Entered on [make.fun](https://make.fun/?ref=w513bxaz&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
