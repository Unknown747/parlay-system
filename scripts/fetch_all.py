"""Central league registry used by ingestion, validation, training, and backtests."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LeagueSpec:
    name: str
    football_data_code: str | None = None
    odds_sport_key: str | None = None
    enabled: bool = True


LEAGUE_REGISTRY = {
    "EPL": LeagueSpec("EPL", "E0", "soccer_epl"),
    "LaLiga": LeagueSpec("LaLiga", "SP1", "soccer_spain_la_liga"),
    "SerieA": LeagueSpec("SerieA", "I1", "soccer_italy_serie_a"),
    "Bundesliga": LeagueSpec("Bundesliga", "D1", "soccer_germany_bundesliga"),
    "Ligue1": LeagueSpec("Ligue1", "F1", "soccer_france_ligue_one"),
    "Eredivisie": LeagueSpec("Eredivisie", "N1", "soccer_netherlands_eredivisie"),
    "PrimeiraLiga": LeagueSpec("PrimeiraLiga", "P1", "soccer_portugal_primeira_liga"),
    "Championship": LeagueSpec("Championship", "E1", "soccer_efl_champ"),
    "LigaMX": LeagueSpec("LigaMX", None, "soccer_mexico_ligamx"),
    "Brasileirao": LeagueSpec("Brasileirao", None, "soccer_brazil_campeonato"),
    "Belgian": LeagueSpec("Belgian", "B1", "soccer_belgium_first_div"),
    "Turkish": LeagueSpec("Turkish", "T1", "soccer_turkey_super_league"),
    "Greek": LeagueSpec("Greek", "G1", "soccer_greece_super_league"),
    "Swiss": LeagueSpec("Swiss", None, "soccer_switzerland_superleague"),
}

LEAGUE_NAMES = tuple(LEAGUE_REGISTRY)
FD_LEAGUE_CODES = {name: spec.football_data_code for name, spec in LEAGUE_REGISTRY.items() if spec.football_data_code}
ODDS_SPORT_KEYS = {name: spec.odds_sport_key for name, spec in LEAGUE_REGISTRY.items() if spec.odds_sport_key}


def configured_leagues(env_value: str | None) -> list[str]:
    """Return configured, known leagues while preserving user order."""
    requested = [x.strip() for x in (env_value or "").split(",") if x.strip()]
    return [name for name in requested if name in LEAGUE_REGISTRY and LEAGUE_REGISTRY[name].enabled]


def unknown_leagues(values) -> list[str]:
    return sorted({str(v) for v in values if str(v) not in LEAGUE_REGISTRY})
