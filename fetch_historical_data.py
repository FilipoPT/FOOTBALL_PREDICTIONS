#!/usr/bin/env python3
"""
fetch_historical_data.py

Descarrega resultados historicos (com golos reais) de football-data.co.uk
para varias ligas e epocas, e gera um unico CSV normalizado no formato
que o daily_predictions.py espera para treinar o Dixon-Coles:

    date,league,home_team,away_team,home_goals,away_goals

Uso:
    python3 fetch_historical_data.py
    python3 fetch_historical_data.py --leagues E0 E1 SP1
    python3 fetch_historical_data.py --seasons 5
    python3 fetch_historical_data.py --output historical_results.csv

IMPORTANTE - Liga Portugal 2:
    football-data.co.uk NAO cobre a 2a divisao portuguesa (so tem P1 =
    Primeira Liga). Esta liga (PT2) fica sem historico por esta via --
    o script avisa e salta. Alternativas: API-Football (RapidAPI, paga
    a partir de certo volume) ou outra fonte manual.
"""

import argparse
import io
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

try:
    import requests
except ImportError:
    requests = None


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

# Mapeia o teu codigo de liga interno (o mesmo usado em daily_predictions.py)
# para o codigo de divisao usado por football-data.co.uk.
# None = nao disponivel nesta fonte.
FD_LEAGUE_MAP = {
    "E0": "E0",     # Premier League
    "E1": "E1",     # Championship
    "SP1": "SP1",   # La Liga
    "SP2": "SP2",   # La Liga 2 (Segunda Division)
    "PT1": "P1",    # Primeira Liga
    "PT2": None,    # Liga Portugal 2 -- NAO disponivel em football-data.co.uk
}

DEFAULT_LEAGUES = ["E0", "E1", "SP1", "SP2", "PT1", "PT2"]

BASE_URL = "https://www.football-data.co.uk/mmz4281"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; DixonColesDataFetch/1.0)"}


# ---------------------------------------------------------------------------
# SEASON CODES
# ---------------------------------------------------------------------------

def season_codes(n_seasons=4):
    """Gera codigos de epoca tipo '2324','2425',... para as ultimas n_seasons
    epocas, incluindo a epoca atual (assume epoca de futebol europeia,
    inicio ~julho/agosto)."""
    now = datetime.now()
    start_year = now.year if now.month >= 7 else now.year - 1
    codes = []
    for i in range(n_seasons):
        y = start_year - i
        codes.append(f"{str(y)[2:]}{str(y + 1)[2:]}")
    return list(reversed(codes))


# ---------------------------------------------------------------------------
# DOWNLOAD + NORMALIZE
# ---------------------------------------------------------------------------

def download_one(league_code, fd_div, season_code, session, max_retries=3):
    url = f"{BASE_URL}/{season_code}/{fd_div}.csv"
    for attempt in range(1, max_retries + 1):
        try:
            resp = session.get(url, headers=HEADERS, timeout=20)
        except Exception as e:
            print(f"  [!] {league_code} {season_code}: erro de rede ({e}), tentativa {attempt}/{max_retries}")
            time.sleep(1.5)
            continue

        if resp.status_code == 404:
            # Epoca/div nao existe (ex: liga ainda nao comecou essa epoca)
            return None
        if resp.status_code != 200:
            print(f"  [!] {league_code} {season_code}: HTTP {resp.status_code}, tentativa {attempt}/{max_retries}")
            time.sleep(1.5)
            continue

        try:
            df = pd.read_csv(io.StringIO(resp.text), on_bad_lines="skip")
        except Exception as e:
            print(f"  [!] {league_code} {season_code}: falha a parsear CSV ({e})")
            return None

        return df

    print(f"  [!] {league_code} {season_code}: desisti apos {max_retries} tentativas.")
    return None


def normalize(df, league_code, season_code):
    """Reduz o CSV de football-data.co.uk as colunas que precisamos e
    normaliza nomes/datas. As colunas variam ligeiramente entre epocas,
    por isso procuramos por variantes conhecidas."""
    required = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"  [!] {league_code} {season_code}: faltam colunas {missing}, a saltar epoca.")
        return None

    out = df[required].copy()
    out.columns = ["date", "home_team", "away_team", "home_goals", "away_goals"]

    # datas em football-data.co.uk vem como DD/MM/YY ou DD/MM/YYYY consoante a epoca
    out["date"] = pd.to_datetime(out["date"], dayfirst=True, errors="coerce")

    out = out.dropna(subset=["date", "home_team", "away_team", "home_goals", "away_goals"])
    out["home_goals"] = out["home_goals"].astype(int)
    out["away_goals"] = out["away_goals"].astype(int)
    out["league"] = league_code

    return out[["date", "league", "home_team", "away_team", "home_goals", "away_goals"]]


def fetch_league(league_code, seasons, session):
    fd_div = FD_LEAGUE_MAP.get(league_code)
    if fd_div is None:
        print(f"[skip] {league_code}: sem fonte configurada em football-data.co.uk. "
              f"Precisas de outra fonte para esta liga.")
        return []

    frames = []
    for season_code in seasons:
        df = download_one(league_code, fd_div, season_code, session)
        if df is None or df.empty:
            continue
        norm = normalize(df, league_code, season_code)
        if norm is not None and not norm.empty:
            frames.append(norm)
            print(f"  {league_code} {season_code}: {len(norm)} jogos")
    return frames


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Descarrega historico de resultados (com golos) para varias ligas.")
    parser.add_argument("--leagues", nargs="+", default=DEFAULT_LEAGUES,
                         help=f"Codigos de liga a descarregar (default: {' '.join(DEFAULT_LEAGUES)})")
    parser.add_argument("--seasons", type=int, default=4, help="Numero de epocas a recuar (default 4)")
    parser.add_argument("--output", default="historical_results.csv", help="Ficheiro CSV de saida")
    args = parser.parse_args()

    if requests is None:
        sys.exit("Erro: precisas de instalar 'requests' (pip install requests --break-system-packages).")

    leagues = [l.upper() for l in args.leagues]
    seasons = season_codes(args.seasons)
    print(f"Epocas a descarregar: {seasons}\n")

    session = requests.Session()
    all_frames = []
    skipped_leagues = []

    for league_code in leagues:
        print(f"=== {league_code} ===")
        frames = fetch_league(league_code, seasons, session)
        if not frames:
            skipped_leagues.append(league_code)
            continue
        all_frames.extend(frames)

    if not all_frames:
        sys.exit("Erro: nao consegui obter dados para nenhuma liga.")

    result = pd.concat(all_frames, ignore_index=True)
    result = result.sort_values(["league", "date"]).reset_index(drop=True)
    result.to_csv(args.output, index=False)

    print(f"\nGuardado: {args.output}")
    print(f"Total de jogos: {len(result)}")
    print(result.groupby("league").size().to_string())

    if skipped_leagues:
        print(f"\n[!] Ligas sem historico obtido: {', '.join(skipped_leagues)}")
        if "PT2" in skipped_leagues:
            print("    A Liga Portugal 2 nao esta disponivel em football-data.co.uk.")
            print("    Precisas de outra fonte (ex: API-Football) para a treinar.")


if __name__ == "__main__":
    main()
