"""
Normalisierung von Teamnamen und Turniernamen.

Viele Datensaetze nutzen unterschiedliche Schreibweisen ("USA" vs "United States",
"Iran" vs "IR Iran", "Ivory Coast" vs "Côte d'Ivoire"). Diese Datei liefert
Mapping-Tabellen, die im Martj42-Datensatz schon sauber sind, aber beim User-Input
im Inference-Modus gebraucht werden.
"""

from __future__ import annotations

# Mapping typischer alternativer Schreibweisen -> Martj42-canonical
TEAM_ALIASES: dict[str, str] = {
    # the-odds-api / Transfermarkt spellings for WC-2026 minnows (else live odds miss)
    "cabo verde": "Cape Verde",
    "curacao": "Curaçao",
    "congo dr": "DR Congo",
    "democratic republic of the congo": "DR Congo",
    "bosnia & herzegovina": "Bosnia and Herzegovina",
    "bosnia-herzegovina": "Bosnia and Herzegovina",
    # USA-Varianten
    "usa": "United States",
    "united states": "United States",
    "united states of america": "United States",
    "us": "United States",
    "usmnt": "United States",
    "vereinigte staaten": "United States",
    # Iran (Martj42 nutzt "Iran", nicht "IR Iran")
    "iran": "Iran",
    "ir iran": "Iran",
    "iran, islamic republic of": "Iran",
    "iran (islamic republic of)": "Iran",
    # Korea
    "south korea": "South Korea",
    "korea republic": "South Korea",
    "korea, republic of": "South Korea",
    "südkorea": "South Korea",
    "north korea": "North Korea",
    "korea dpr": "North Korea",
    "korea, democratic people's republic of": "North Korea",
    "nordkorea": "North Korea",
    # UK-Varianten
    "england": "England",
    "scotland": "Scotland",
    "wales": "Wales",
    "northern ireland": "Northern Ireland",
    "nordirland": "Northern Ireland",
    # China / Taiwan
    "china pr": "China PR",
    "china": "China PR",
    "peoples republic of china": "China PR",
    "chinese taipei": "China PR",
    "taiwan": "China PR",
    # Ivory Coast
    "ivory coast": "Ivory Coast",
    "cote d'ivoire": "Ivory Coast",
    "côte d'ivoire": "Ivory Coast",
    "elfenbeinküste": "Ivory Coast",
    "elfenbeinkueste": "Ivory Coast",
    # Tschechien / Tschechoslowakei
    "czechia": "Czech Republic",
    "czech republic": "Czech Republic",
    "tschechien": "Czech Republic",
    # Türkei
    "turkey": "Turkey",
    "türkiye": "Turkey",
    "turkei": "Turkey",
    "türkei": "Turkey",
    # Russland / UdSSR
    "russia": "Russia",
    "russland": "Russia",
    # Deutschland (zur Sicherheit, falls jemand eingibt)
    "germany": "Germany",
    "deutschland": "Germany",
    "bundesrepublik deutschland": "Germany",
    "west germany": "Germany",
    "east germany": "Germany",
    "brd": "Germany",
    "ddr": "Germany",
    # Andere haeufige Faelle
    "holland": "Netherlands",
    "the netherlands": "Netherlands",
    "niederlande": "Netherlands",
    "uae": "United Arab Emirates",
    "united arab emirates": "United Arab Emirates",
    "saudi": "Saudi Arabia",
    "saudi arabia": "Saudi Arabia",
    "katar": "Qatar",
    "qatar": "Qatar",
    "marokko": "Morocco",
    "morocco": "Morocco",
    "brasilien": "Brazil",
    "brazil": "Brazil",
    "frankreich": "France",
    "france": "France",
    "spanien": "Spain",
    "spain": "Spain",
    "spanish": "Spain",
    "argentinien": "Argentina",
    "argentina": "Argentina",
    "italien": "Italy",
    "italy": "Italy",
    "italia": "Italy",
    "portugal": "Portugal",
    "belgien": "Belgium",
    "belgium": "Belgium",
    "kroatien": "Croatia",
    "croatia": "Croatia",
    "dänemark": "Denmark",
    "denmark": "Denmark",
    "schweden": "Sweden",
    "sweden": "Sweden",
    "norwegen": "Norway",
    "norway": "Norway",
    "finnland": "Finland",
    "finland": "Finland",
    "polen": "Poland",
    "poland": "Poland",
    "österreich": "Austria",
    "oesterreich": "Austria",
    "austria": "Austria",
    "schweiz": "Switzerland",
    "switzerland": "Switzerland",
    "ungarn": "Hungary",
    "hungary": "Hungary",
    "rumänien": "Romania",
    "romania": "Romania",
    "serbien": "Serbia",
    "serbia": "Serbia",
    "albanien": "Albania",
    "albania": "Albania",
    "griechenland": "Greece",
    "greece": "Greece",
    "türkei": "Türkiye",
}


def normalize_team_name(name: str) -> str:
    """Mappt alternative Schreibweisen auf den Martj42-canonical Namen."""
    if not name:
        return name
    key = name.strip().lower()
    return TEAM_ALIASES.get(key, name.strip())


# Encoding von Turnieren in Wichtigkeits-Klassen.
# Quelle: angelehnt an Elo-Standard (FiveThirtyEight / World Football Elo Ratings)
TOURNAMENT_TIERS: dict[str, int] = {
    # Tier 1: Top-Turniere
    "FIFA World Cup": 60,
    "UEFA Euro": 50,
    "Copa América": 50,
    "CONCACAF Gold Cup": 45,
    "Africa Cup of Nations": 50,
    "Asian Cup": 45,
    "Oceania Nations Cup": 30,
    # Tier 2: Qualifier
    "FIFA World Cup qualification": 40,
    "UEFA Euro qualification": 40,
    "CONCACAF Nations League": 35,
    "CONCACAF Nations League Qualification": 35,
    "UEFA Nations League": 40,
    "Copa América qualification": 40,
    "African Cup of Nations qualification": 35,
    "Asian Cup qualification": 35,
    # Tier 3: Friendly (default)
}


def tournament_weight(tournament: str) -> int:
    """Gewicht (K-Faktor) eines Spiels abhängig vom Turnier."""
    if not tournament:
        return 20
    return TOURNAMENT_TIERS.get(tournament, 20)  # default = friendly weight
