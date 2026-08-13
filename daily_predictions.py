#!/usr/bin/env python3
"""
daily_predictions.py

Mostra a probabilidade 1X2 e a probabilidade de Over/Under golos para
os jogos de hoje em varias ligas de uma vez, usando um modelo Dixon-Coles
treinado a partir de resultados historicos. Nao usa nem mostra odds —
so fixtures (via endpoint gratuito /events) + previsoes do modelo.

Uso:
    python3 daily_predictions.py                  -> corre a DEFAULT_LEAGUES toda
    python3 daily_predictions.py --leagues E0 SP1  -> so estas ligas
    python3 daily_predictions.py --force-retrain
    python3 daily_predictions.py --goal-line 2.5
    python3 daily_predictions.py --list-sports     -> lista sport_keys validos na API (debug)

Configuracao (editar CONFIG abaixo ou usar variaveis de ambiente):
    ODDS_API_KEY          -> chave da The Odds API (theoddsapi.com) -- so para fixtures, endpoint gratis
    HISTORICAL_DATA_PATH  -> CSV com colunas: date,league,home_team,away_team,home_goals,away_goals
    MODEL_CACHE_DIR       -> onde guardar os parametros treinados (default ./model_cache)
    CACHE_MAX_AGE_DAYS    -> idade maxima do cache antes de retreinar (default 7)
    TELEGRAM_BOT_TOKEN    -> token do bot (obtido via @BotFather no Telegram)
    TELEGRAM_CHAT_ID      -> id do chat/utilizador para onde enviar as mensagens

Como criar o bot do Telegram (5 min):
    1. Abre o Telegram, procura @BotFather, manda /newbot e segue as instrucoes.
    2. O BotFather devolve um token tipo "123456789:AAF...". Isso e o TELEGRAM_BOT_TOKEN.
    3. Manda uma mensagem qualquer ao teu novo bot (procura-o pelo username que deste).
    4. Visita: https://api.telegram.org/bot<TEU_TOKEN>/getUpdates
       O "chat":{"id": ...} que aparece ai e o teu TELEGRAM_CHAT_ID.
    5. Define as duas variaveis de ambiente (ou GitHub Secrets) e corre o script.

Mapeamento de ligas (codigo interno -> sport_key da The Odds API):
    ver LEAGUE_MAP abaixo. Os sport_keys de La Liga 2 e Liga Portugal 2
    nao estao 100% confirmados -- corre --list-sports e confirma/ajusta
    antes da primeira utilizacao.
"""

import argparse
import json
import os
import sys
import time
import difflib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

try:
    import requests
except ImportError:
    requests = None


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

CONFIG = {
    "ODDS_API_KEY": os.environ.get("ODDS_API_KEY", ""),
    "HISTORICAL_DATA_PATH": os.environ.get("HISTORICAL_DATA_PATH", "historical_results.csv"),
    "MODEL_CACHE_DIR": os.environ.get("MODEL_CACHE_DIR", "./model_cache"),
    "CACHE_MAX_AGE_DAYS": int(os.environ.get("CACHE_MAX_AGE_DAYS", "7")),
    "TELEGRAM_BOT_TOKEN": os.environ.get("TELEGRAM_BOT_TOKEN", ""),
    "TELEGRAM_CHAT_ID": os.environ.get("TELEGRAM_CHAT_ID", ""),
}

TELEGRAM_MAX_LEN = 4000  # limite real da Telegram e 4096; deixa margem

# Mapeia o teu codigo de liga interno para o sport_key usado pela The Odds API.
# Consulta a lista completa em: https://the-odds-api.com/sports-odds-data/sports-apis.html
# Os marcados com (?) nao vieram confirmados na documentacao publica -- corre
# `python3 daily_predictions.py --list-sports` e confirma o key exato antes de usar.
LEAGUE_MAP = {
    "E0": "soccer_epl",                        # Premier League
    "E1": "soccer_efl_champ",                  # Championship
    "SP1": "soccer_spain_la_liga",              # La Liga
    "SP2": "soccer_spain_segunda_division",     # La Liga 2 (?)
    "PT1": "soccer_portugal_primeira_liga",     # Primeira Liga
    "PT2": "soccer_portugal_liga_2",            # Liga Portugal 2 (?)
}

# Ligas a correr quando nao especificas --leagues. Ajusta a vontade.
DEFAULT_LEAGUES = ["E0", "E1", "SP1", "SP2", "PT1", "PT2"]

ODDS_API_BASE = "https://api.the-odds-api.com/v4"


# ---------------------------------------------------------------------------
# DIXON-COLES MODEL
# ---------------------------------------------------------------------------

def rho_correction(home_goals, away_goals, home_exp, away_exp, rho):
    """Ajuste de correlacao Dixon-Coles para resultados de baixo scoring."""
    if home_goals == 0 and away_goals == 0:
        return 1 - (home_exp * away_exp * rho)
    elif home_goals == 0 and away_goals == 1:
        return 1 + (home_exp * rho)
    elif home_goals == 1 and away_goals == 0:
        return 1 + (away_exp * rho)
    elif home_goals == 1 and away_goals == 1:
        return 1 - rho
    return 1.0


def rho_correction_vec(home_goals, away_goals, home_exp, away_exp, rho):
    """Versao vetorizada do ajuste de correlacao Dixon-Coles (ver rho_correction)."""
    corr = np.ones_like(home_exp, dtype=float)
    m00 = (home_goals == 0) & (away_goals == 0)
    m01 = (home_goals == 0) & (away_goals == 1)
    m10 = (home_goals == 1) & (away_goals == 0)
    m11 = (home_goals == 1) & (away_goals == 1)
    corr[m00] = 1 - (home_exp[m00] * away_exp[m00] * rho)
    corr[m01] = 1 + (home_exp[m01] * rho)
    corr[m10] = 1 + (away_exp[m10] * rho)
    corr[m11] = 1 - rho
    return corr


def dc_log_likelihood(params, teams, home_idx, away_idx, home_goals, away_goals, weights):
    n = len(teams)
    attack = params[:n]
    defense = params[n:2 * n]
    home_adv = params[2 * n]
    rho = params[2 * n + 1]

    home_exp = np.exp(attack[home_idx] + defense[away_idx] + home_adv)
    away_exp = np.exp(attack[away_idx] + defense[home_idx])

    p_home = poisson.pmf(home_goals, home_exp)
    p_away = poisson.pmf(away_goals, away_exp)
    corr = rho_correction_vec(home_goals, away_goals, home_exp, away_exp, rho)
    corr = np.clip(corr, 1e-10, None)
    prob = np.clip(p_home * p_away * corr, 1e-10, None)

    return -np.sum(weights * np.log(prob))


def fit_dixon_coles(df, half_life_days=180):
    """
    Treina o modelo Dixon-Coles a partir de um dataframe com colunas:
    date, home_team, away_team, home_goals, away_goals
    Aplica peso de decaimento temporal (jogos recentes pesam mais).
    """
    teams = sorted(set(df["home_team"]) | set(df["away_team"]))
    team_idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)

    home_idx = df["home_team"].map(team_idx).values
    away_idx = df["away_team"].map(team_idx).values
    home_goals = df["home_goals"].values
    away_goals = df["away_goals"].values

    max_date = df["date"].max()
    days_ago = (max_date - df["date"]).dt.days.values
    weights = 0.5 ** (days_ago / half_life_days)

    x0 = np.concatenate([
        np.zeros(n),        # attack
        np.zeros(n),        # defense
        [0.3],               # home advantage
        [0.0],               # rho
    ])

    res = minimize(
        dc_log_likelihood,
        x0,
        args=(teams, home_idx, away_idx, home_goals, away_goals, weights),
        method="L-BFGS-B",
        options={"maxiter": 300},
    )

    attack = res.x[:n]
    defense = res.x[n:2 * n]
    home_adv = res.x[2 * n]
    rho = res.x[2 * n + 1]

    # centrar attack em 0 para identificabilidade
    attack = attack - attack.mean()

    return {
        "teams": teams,
        "attack": {t: float(attack[i]) for i, t in enumerate(teams)},
        "defense": {t: float(defense[i]) for i, t in enumerate(teams)},
        "home_advantage": float(home_adv),
        "rho": float(rho),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_matches": len(df),
    }


def predict_match(model, home_team, away_team, goal_line=2.5, max_goals=10):
    """Devolve probabilidades 1X2 e Over/Under para um jogo, dado o modelo treinado."""
    attack = model["attack"]
    defense = model["defense"]
    home_adv = model["home_advantage"]
    rho = model["rho"]

    if home_team not in attack or away_team not in attack:
        return None

    lam = np.exp(attack[home_team] + defense[away_team] + home_adv)
    mu = np.exp(attack[away_team] + defense[home_team])

    score_matrix = np.zeros((max_goals + 1, max_goals + 1))
    for hg in range(max_goals + 1):
        for ag in range(max_goals + 1):
            p = poisson.pmf(hg, lam) * poisson.pmf(ag, mu)
            p *= rho_correction(hg, ag, lam, mu, rho)
            score_matrix[hg, ag] = max(p, 0)

    score_matrix /= score_matrix.sum()  # normalizar

    p_home = np.tril(score_matrix, -1).sum()
    p_draw = np.trace(score_matrix)
    p_away = np.triu(score_matrix, 1).sum()

    goal_totals = np.add.outer(np.arange(max_goals + 1), np.arange(max_goals + 1))
    p_over = score_matrix[goal_totals > goal_line].sum()
    p_under = score_matrix[goal_totals < goal_line].sum()

    # P(equipa marca >=1 golo) -- lido diretamente da score_matrix, nao assume
    # independencia entre equipas (o rho do Dixon-Coles ja esta refletido aqui).
    p_home_scores = 1 - score_matrix[0, :].sum()   # linha 0 = casa fez 0 golos
    p_away_scores = 1 - score_matrix[:, 0].sum()   # coluna 0 = fora fez 0 golos
    p_btts_yes = 1 - score_matrix[0, :].sum() - score_matrix[:, 0].sum() + score_matrix[0, 0]
    p_btts_no = 1 - p_btts_yes

    return {
        "home_exp_goals": round(float(lam), 2),
        "away_exp_goals": round(float(mu), 2),
        "p_home": round(float(p_home), 4),
        "p_draw": round(float(p_draw), 4),
        "p_away": round(float(p_away), 4),
        "p_over": round(float(p_over), 4),
        "p_under": round(float(p_under), 4),
        "goal_line": goal_line,
        "p_home_scores": round(float(p_home_scores), 4),
        "p_away_scores": round(float(p_away_scores), 4),
        "p_btts_yes": round(float(p_btts_yes), 4),
        "p_btts_no": round(float(p_btts_no), 4),
    }


# ---------------------------------------------------------------------------
# CACHE
# ---------------------------------------------------------------------------

def cache_path(league_code):
    Path(CONFIG["MODEL_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
    return Path(CONFIG["MODEL_CACHE_DIR"]) / f"dixon_coles_{league_code}.json"


def load_cached_model(league_code, max_age_days):
    path = cache_path(league_code)
    if not path.exists():
        return None
    with open(path) as f:
        model = json.load(f)
    trained_at = datetime.fromisoformat(model["trained_at"])
    age_days = (datetime.now(timezone.utc) - trained_at).total_seconds() / 86400
    if age_days > max_age_days:
        return None
    return model


def save_model(league_code, model):
    with open(cache_path(league_code), "w") as f:
        json.dump(model, f, indent=2)


def get_model(league_code, force_retrain=False):
    if not force_retrain:
        cached = load_cached_model(league_code, CONFIG["CACHE_MAX_AGE_DAYS"])
        if cached is not None:
            print(f"[cache] A usar modelo treinado em {cached['trained_at']} "
                  f"({cached['n_matches']} jogos).")
            return cached

    print(f"[treino] Cache ausente ou expirado. A treinar modelo para {league_code}...")
    hist_path = CONFIG["HISTORICAL_DATA_PATH"]
    if not os.path.exists(hist_path):
        sys.exit(
            f"Erro: nao encontrei o ficheiro historico em '{hist_path}'.\n"
            f"Define HISTORICAL_DATA_PATH com colunas: date,league,home_team,away_team,home_goals,away_goals"
        )

    df = pd.read_csv(hist_path, parse_dates=["date"])
    df = df[df["league"] == league_code].copy()
    if len(df) < 100:
        sys.exit(f"Erro: apenas {len(df)} jogos historicos para {league_code}. Preciso de mais dados para treinar.")

    model = fit_dixon_coles(df)
    save_model(league_code, model)
    print(f"[treino] Modelo treinado e guardado em {cache_path(league_code)} ({len(df)} jogos).")
    return model


# ---------------------------------------------------------------------------
# FIXTURES / ODDS (The Odds API)
# ---------------------------------------------------------------------------

def _require_requests():
    if requests is None:
        sys.exit("Erro: precisas de instalar 'requests' (pip install requests --break-system-packages).")


def _require_api_key():
    api_key = CONFIG["ODDS_API_KEY"]
    if not api_key:
        sys.exit("Erro: define a variavel de ambiente ODDS_API_KEY com a tua chave de theoddsapi.com")
    return api_key


def list_available_sports():
    """Lista os sport_keys em vigor na API (endpoint gratis, nao gasta quota). Util para
    confirmares os keys corretos de La Liga 2 / Liga Portugal 2 antes de correr o script a serio."""
    _require_requests()
    api_key = _require_api_key()
    resp = requests.get(f"{ODDS_API_BASE}/sports", params={"apiKey": api_key, "all": "true"}, timeout=15)
    if resp.status_code != 200:
        sys.exit(f"Erro na API ({resp.status_code}): {resp.text[:300]}")
    for s in resp.json():
        if s.get("group") == "Soccer":
            print(f"  {s['key']:<40} {s['title']}")


def fetch_todays_fixtures(league_code):
    """Usa o endpoint /events -- so devolve fixtures (id, equipas, hora), sem odds,
    e NAO conta para a quota da API."""
    _require_requests()
    sport_key = LEAGUE_MAP.get(league_code)
    if sport_key is None:
        print(f"  [!] Liga '{league_code}' nao esta em LEAGUE_MAP. A saltar.")
        return []

    api_key = _require_api_key()
    url = f"{ODDS_API_BASE}/sports/{sport_key}/events"
    params = {"apiKey": api_key, "dateFormat": "iso"}
    resp = requests.get(url, params=params, timeout=15)
    if resp.status_code != 200:
        print(f"  [!] Erro na API para {league_code} ({resp.status_code}): {resp.text[:200]}")
        return []

    data = resp.json()
    today = datetime.now(timezone.utc).date()
    todays_games = []
    for game in data:
        commence = datetime.fromisoformat(game["commence_time"].replace("Z", "+00:00"))
        if commence.date() == today:
            todays_games.append(game)
    return todays_games


def match_team_name(name, known_teams, cutoff=0.6):
    """Fuzzy match entre nomes de equipa da API e do dataset historico."""
    matches = difflib.get_close_matches(name, known_teams, n=1, cutoff=cutoff)
    return matches[0] if matches else None


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# TELEGRAM
# ---------------------------------------------------------------------------

def send_telegram_message(text):
    """Envia uma mensagem via Telegram Bot API. Faz chunking automatico se o
    texto ultrapassar o limite da Telegram (4096 chars, usamos margem 4000)."""
    _require_requests()
    token = CONFIG["TELEGRAM_BOT_TOKEN"]
    chat_id = CONFIG["TELEGRAM_CHAT_ID"]
    if not token or not chat_id:
        print("[telegram] TELEGRAM_BOT_TOKEN ou TELEGRAM_CHAT_ID nao definidos -- a saltar envio.")
        return

    url = f"https://api.telegram.org/bot{token}/sendMessage"

    # parte o texto em blocos que cabem no limite, sem cortar a meio de uma linha
    chunks = []
    current = ""
    for line in text.split("\n"):
        if len(current) + len(line) + 1 > TELEGRAM_MAX_LEN:
            chunks.append(current)
            current = ""
        current += line + "\n"
    if current:
        chunks.append(current)

    for chunk in chunks:
        try:
            resp = requests.post(url, data={
                "chat_id": chat_id,
                "text": chunk,
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
            }, timeout=15)
            if resp.status_code != 200:
                print(f"[telegram] Erro ao enviar ({resp.status_code}): {resp.text[:300]}")
        except Exception as e:
            print(f"[telegram] Erro de rede ao enviar: {e}")
        time.sleep(0.5)  # evitar rate limit


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def build_predictions_for_league(league_code, goal_line, force_retrain):
    """Gera as previsoes de uma liga e devolve o texto formatado (HTML, para
    Telegram) em vez de o imprimir diretamente -- assim serve tanto para
    consola como para envio."""
    lines = []
    lines.append(f"\n<b>=== {league_code} ===</b>")

    model = get_model(league_code, force_retrain=force_retrain)
    known_teams = model["teams"]

    games = fetch_todays_fixtures(league_code)
    if not games:
        lines.append("Sem jogos hoje para esta liga.")
        return "\n".join(lines)

    for game in games:
        home_raw = game["home_team"]
        away_raw = game["away_team"]
        home = match_team_name(home_raw, known_teams) or home_raw
        away = match_team_name(away_raw, known_teams) or away_raw

        pred = predict_match(model, home, away, goal_line=goal_line)

        lines.append(f"\n<b>{home_raw} vs {away_raw}</b>")
        if pred is None:
            lines.append("  [!] Equipa nao encontrada no historico treinado — sem previsao.")
            continue
        lines.append(f"  Golos esperados: {pred['home_exp_goals']} - {pred['away_exp_goals']}")
        lines.append(f"  1X2: {pred['p_home']*100:.1f}% / {pred['p_draw']*100:.1f}% / {pred['p_away']*100:.1f}%")
        lines.append(f"  Over/Under {pred['goal_line']}: {pred['p_over']*100:.1f}% / {pred['p_under']*100:.1f}%")
        lines.append(f"  {home_raw} marca: {pred['p_home_scores']*100:.1f}%  |  {away_raw} marca: {pred['p_away_scores']*100:.1f}%")
        lines.append(f"  Ambas marcam: {pred['p_btts_yes']*100:.1f}% sim / {pred['p_btts_no']*100:.1f}% nao")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Probabilidades 1X2 e Over/Under para os jogos do dia, em varias ligas.")
    parser.add_argument("--leagues", nargs="+", default=None,
                         help=f"Codigos de liga a correr (default: {' '.join(DEFAULT_LEAGUES)})")
    parser.add_argument("--goal-line", type=float, default=2.5, help="Linha de golos para Over/Under (default 2.5)")
    parser.add_argument("--force-retrain", action="store_true", help="Ignora o cache e retreina o modelo")
    parser.add_argument("--list-sports", action="store_true", help="Lista sport_keys de futebol validos na API e sai")
    parser.add_argument("--no-telegram", action="store_true", help="Nao envia para o Telegram, so imprime na consola")
    args = parser.parse_args()

    if args.list_sports:
        list_available_sports()
        return

    leagues = [l.upper() for l in (args.leagues or DEFAULT_LEAGUES)]

    today_str = datetime.now().strftime("%Y-%m-%d")
    full_text = f"<b>Previsoes {today_str}</b>"

    for league_code in leagues:
        try:
            block = build_predictions_for_league(league_code, args.goal_line, args.force_retrain)
        except SystemExit as e:
            block = f"\n<b>=== {league_code} ===</b>\n[!] {e}"
        print(block)
        full_text += "\n" + block

    if not args.no_telegram:
        send_telegram_message(full_text)


if __name__ == "__main__":
    main()
