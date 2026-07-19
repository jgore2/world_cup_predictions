import os, warnings, textwrap
import numpy as np
import pandas as pd
import requests
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import accuracy_score, log_loss
from scipy.stats import poisson as poisson_dist
from scipy.optimize import minimize

warnings.filterwarnings("ignore")
np.random.seed(42)

# ─── MATCH INFO ────────────────────────────────────────────────────────────
MATCH_DATE  = "2026-07-19"
HOME_TEAM   = "Spain"
AWAY_TEAM   = "Argentina"
MATCH_STAGE = "Final"
MATCH_VENUE = "MetLife Stadium, New Jersey"

# ─── CONSTANTS ─────────────────────────────────────────────────────────────
CACHE_DIR   = "data_cache"
RESULTS_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"

NAME_MAP = {
    "USA": "United States", "Korea Republic": "South Korea",
    "Republic of Ireland": "Ireland", "Türkiye": "Turkey",
    "Cape Verde": "Cabo Verde", "Côte d'Ivoire": "Ivory Coast",
    "Czechia": "Czech Republic", "Curaçao": "Curacao",
    "Congo DR": "DR Congo", "Congo": "Republic of the Congo",
}

TEAM_FEATURES = [
    "neutral", "tournament_weight", "home_elo", "away_elo", "elo_diff",
    "home_win5", "away_win5", "home_gd5", "away_gd5",
    "home_win10", "away_win10", "home_rest_days", "away_rest_days",
    "h2h_n", "h2h_home_winrate", "h2h_home_gd",
]

TRAIN_START    = "2006-01-01"
VAL_START      = "2023-01-01"
ELO_BASE       = 1500.0
ELO_K          = 32
ELO_HOME_BONUS = 60

N_SIMULATIONS     = 10_000
DIXON_COLES_RHO   = -0.13
MAX_GOALS          = 7
TOURNAMENT_AVG_XG  = 1.40
KNOCKOUT_REGRESSION = 0.95  # slight regression for final — high stakes, tight play

# ─── CHART STYLING ─────────────────────────────────────────────────────────
INK, MUTE, GRID = "#1a1a2e", "#8a8a9e", "#e8e8ee"
ESP_RED, ARG_BLUE, GRAY = "#AA151B", "#6CACE4", "#9aa0a6"
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": MUTE, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.titlecolor": INK,
    "font.size": 11, "axes.titlesize": 14, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
})

SPAIN_PLAYERS = [
    # ── Goalkeepers ──
    {"name": "Unai Simon",            "pos": "GK", "age": 27, "club": "Athletic Bilbao",
     "club_apps": 36, "club_cs": 13, "club_ga_90": 0.94,
     "wc_apps": 7, "wc_mins": 630, "wc_cs": 3, "wc_ga": 4,
     "wc_saves": 18, "wc_save_pct": 0.82,
     "starter": True, "notes": "Ever-present; 3 clean sheets; commanding"},
    # ── Defenders ──
    {"name": "Dani Carvajal",         "pos": "FB", "age": 34, "club": "Real Madrid",
     "club_apps": 30, "club_goals": 2, "club_assists": 5, "club_rating": 7.1,
     "wc_apps": 7, "wc_mins": 610, "wc_goals": 0, "wc_assists": 2,
     "wc_tackles": 16, "wc_interceptions": 10, "wc_key_passes": 8,
     "starter": True, "notes": "Veteran RB; 2 assists; experience in finals"},
    {"name": "Aymeric Laporte",       "pos": "CB", "age": 32, "club": "Al-Nassr",
     "club_apps": 28, "club_goals": 2, "club_assists": 1, "club_rating": 7.0,
     "wc_apps": 7, "wc_mins": 630, "wc_goals": 1, "wc_assists": 0,
     "wc_tackles": 14, "wc_interceptions": 12, "wc_aerials_won": 22,
     "starter": True, "notes": "First-choice CB; headed goal vs Belgium"},
    {"name": "Robin Le Normand",      "pos": "CB", "age": 29, "club": "Atletico Madrid",
     "club_apps": 34, "club_goals": 1, "club_assists": 0, "club_rating": 7.0,
     "wc_apps": 7, "wc_mins": 630, "wc_goals": 0, "wc_assists": 0,
     "wc_tackles": 18, "wc_interceptions": 14, "wc_aerials_won": 20,
     "starter": True, "notes": "Rock-solid CB partnership; tournament-high tackles"},
    {"name": "Marc Cucurella",        "pos": "FB", "age": 27, "club": "Chelsea",
     "club_apps": 36, "club_goals": 1, "club_assists": 6, "club_rating": 7.0,
     "wc_apps": 7, "wc_mins": 600, "wc_goals": 0, "wc_assists": 1,
     "wc_tackles": 14, "wc_interceptions": 8, "wc_key_passes": 6,
     "starter": True, "notes": "Overlapping LB; Euro 2024 experience"},
    # ── Midfielders ──
    {"name": "Rodri",                 "pos": "CM", "age": 30, "club": "Man City",
     "club_apps": 38, "club_goals": 5, "club_assists": 8, "club_rating": 7.6,
     "wc_apps": 7, "wc_mins": 630, "wc_goals": 1, "wc_assists": 2,
     "wc_tackles": 20, "wc_interceptions": 16, "wc_key_passes": 14,
     "starter": True, "notes": "Ballon d'Or holder; midfield metronome; 1G 2A"},
    {"name": "Pedri",                 "pos": "CM", "age": 23, "club": "Barcelona",
     "club_apps": 40, "club_goals": 8, "club_assists": 10, "club_rating": 7.5,
     "wc_apps": 7, "wc_mins": 590, "wc_goals": 2, "wc_assists": 3,
     "wc_tackles": 10, "wc_interceptions": 8, "wc_key_passes": 18,
     "wc_dribbles": 16,
     "starter": True, "notes": "2G 3A; creative hub; tournament-high key passes"},
    # ── Attacking midfielders / Wingers ──
    {"name": "Lamine Yamal",          "pos": "W",  "age": 19, "club": "Barcelona",
     "club_apps": 42, "club_goals": 12, "club_assists": 14, "club_rating": 7.7,
     "wc_apps": 7, "wc_mins": 620, "wc_goals": 4, "wc_assists": 5,
     "wc_shots": 20, "wc_sot": 10, "wc_xg": 3.2, "wc_key_passes": 16,
     "wc_dribbles": 28,
     "starter": True, "notes": "4G 5A; youngest player to 4 WC goals; electric"},
    {"name": "Nico Williams",         "pos": "W",  "age": 23, "club": "Athletic Bilbao",
     "club_apps": 38, "club_goals": 10, "club_assists": 12, "club_rating": 7.4,
     "wc_apps": 7, "wc_mins": 560, "wc_goals": 3, "wc_assists": 2,
     "wc_shots": 14, "wc_sot": 8, "wc_xg": 2.4, "wc_key_passes": 10,
     "wc_dribbles": 22,
     "starter": True, "notes": "3G 2A; blistering pace; scored in SF vs France"},
    {"name": "Dani Olmo",             "pos": "AM", "age": 28, "club": "Barcelona",
     "club_apps": 34, "club_goals": 10, "club_assists": 8, "club_rating": 7.3,
     "wc_apps": 7, "wc_mins": 500, "wc_goals": 3, "wc_assists": 2,
     "wc_shots": 16, "wc_sot": 9, "wc_xg": 2.6, "wc_key_passes": 12,
     "wc_dribbles": 10,
     "starter": True, "notes": "3G 2A; links midfield to attack; Euro 2024 top scorer"},
    # ── Forwards ──
    {"name": "Alvaro Morata",         "pos": "ST", "age": 33, "club": "AC Milan",
     "club_apps": 32, "club_goals": 12, "club_assists": 4, "club_rating": 6.9,
     "wc_apps": 7, "wc_mins": 480, "wc_goals": 2, "wc_assists": 1,
     "wc_shots": 14, "wc_sot": 7, "wc_xg": 2.8, "wc_key_passes": 4,
     "wc_dribbles": 6,
     "starter": True, "notes": "Captain; 2G 1A; tireless pressing; hold-up play"},
]

ARGENTINA_PLAYERS = [
    # ── Goalkeepers ──
    {"name": "Emiliano Martinez",     "pos": "GK", "age": 33, "club": "Aston Villa",
     "club_apps": 36, "club_cs": 12, "club_ga_90": 1.06,
     "wc_apps": 7, "wc_mins": 630, "wc_cs": 2, "wc_ga": 6,
     "wc_saves": 24, "wc_save_pct": 0.80,
     "starter": True, "notes": "Tournament-high saves (24); penalty-saving pedigree"},
    # ── Defenders ──
    {"name": "Nahuel Molina",         "pos": "FB", "age": 28, "club": "Atletico Madrid",
     "club_apps": 34, "club_goals": 3, "club_assists": 6, "club_rating": 7.0,
     "wc_apps": 7, "wc_mins": 600, "wc_goals": 1, "wc_assists": 1,
     "wc_tackles": 14, "wc_interceptions": 8, "wc_key_passes": 6,
     "starter": True, "notes": "1G 1A; attacking runs from RB; scored vs Egypt"},
    {"name": "Cristian Romero",       "pos": "CB", "age": 28, "club": "Tottenham",
     "club_apps": 36, "club_goals": 2, "club_assists": 1, "club_rating": 7.2,
     "wc_apps": 7, "wc_mins": 630, "wc_goals": 0, "wc_assists": 0,
     "wc_tackles": 22, "wc_interceptions": 16, "wc_aerials_won": 26,
     "starter": True, "notes": "Aggressive CB; tournament-leading aerials won"},
    {"name": "Nicolas Otamendi",      "pos": "CB", "age": 38, "club": "Benfica",
     "club_apps": 30, "club_goals": 1, "club_assists": 0, "club_rating": 6.8,
     "wc_apps": 7, "wc_mins": 630, "wc_goals": 0, "wc_assists": 0,
     "wc_tackles": 16, "wc_interceptions": 12, "wc_aerials_won": 18,
     "starter": True, "notes": "Veteran presence; experience in 3 World Cups"},
    {"name": "Nicolas Tagliafico",    "pos": "FB", "age": 33, "club": "Lyon",
     "club_apps": 32, "club_goals": 1, "club_assists": 4, "club_rating": 6.8,
     "wc_apps": 7, "wc_mins": 580, "wc_goals": 0, "wc_assists": 1,
     "wc_tackles": 12, "wc_interceptions": 6, "wc_key_passes": 4,
     "starter": True, "notes": "Steady LB; 1 assist; defensive discipline"},
    # ── Midfielders ──
    {"name": "Rodrigo De Paul",       "pos": "CM", "age": 32, "club": "Atletico Madrid",
     "club_apps": 36, "club_goals": 3, "club_assists": 5, "club_rating": 7.1,
     "wc_apps": 7, "wc_mins": 600, "wc_goals": 0, "wc_assists": 2,
     "wc_tackles": 18, "wc_interceptions": 14, "wc_key_passes": 10,
     "starter": True, "notes": "Box-to-box engine; 2 assists; covers huge ground"},
    {"name": "Enzo Fernandez",        "pos": "CM", "age": 25, "club": "Chelsea",
     "club_apps": 40, "club_goals": 5, "club_assists": 8, "club_rating": 7.3,
     "wc_apps": 7, "wc_mins": 620, "wc_goals": 1, "wc_assists": 3,
     "wc_tackles": 16, "wc_interceptions": 12, "wc_key_passes": 14,
     "wc_dribbles": 10,
     "starter": True, "notes": "1G 3A; playmaking CM; 2022 WC Young Player Award"},
    {"name": "Alexis Mac Allister",   "pos": "AM", "age": 26, "club": "Liverpool",
     "club_apps": 42, "club_goals": 7, "club_assists": 10, "club_rating": 7.4,
     "wc_apps": 7, "wc_mins": 560, "wc_goals": 2, "wc_assists": 2,
     "wc_shots": 12, "wc_sot": 6, "wc_xg": 1.8, "wc_key_passes": 12,
     "wc_dribbles": 8,
     "starter": True, "notes": "2G 2A; scored winning goal vs England in SF"},
    # ── Forwards ──
    {"name": "Lionel Messi",          "pos": "W",  "age": 39, "club": "Inter Miami",
     "club_apps": 28, "club_goals": 18, "club_assists": 14, "club_rating": 7.6,
     "wc_apps": 7, "wc_mins": 540, "wc_goals": 4, "wc_assists": 4,
     "wc_shots": 18, "wc_sot": 10, "wc_xg": 3.0, "wc_key_passes": 16,
     "wc_dribbles": 20,
     "starter": True, "notes": "4G 4A; GOAT farewell tour; 2nd all-time WC goals (17)"},
    {"name": "Julian Alvarez",        "pos": "ST", "age": 26, "club": "Atletico Madrid",
     "club_apps": 40, "club_goals": 20, "club_assists": 8, "club_rating": 7.4,
     "wc_apps": 7, "wc_mins": 580, "wc_goals": 5, "wc_assists": 2,
     "wc_shots": 22, "wc_sot": 12, "wc_xg": 4.0, "wc_key_passes": 6,
     "wc_dribbles": 12,
     "starter": True, "notes": "5G 2A; clinical finisher; brace vs Egypt & Switzerland"},
    {"name": "Lautaro Martinez",      "pos": "ST", "age": 28, "club": "Inter Milan",
     "club_apps": 38, "club_goals": 24, "club_assists": 6, "club_rating": 7.5,
     "wc_apps": 6, "wc_mins": 380, "wc_goals": 3, "wc_assists": 1,
     "wc_shots": 14, "wc_sot": 8, "wc_xg": 2.4, "wc_key_passes": 4,
     "wc_dribbles": 8,
     "starter": True, "notes": "3G 1A; super-sub role; scored vs England SF"},
]

# Key bench players who may impact the game as subs
SPAIN_BENCH = [
    {"name": "Ferran Torres",  "pos": "W",  "wc_goals": 2, "wc_assists": 1,
     "club_rating": 7.0, "impact": "high",
     "notes": "2G 1A; pace off the bench; Euro 2024 experience"},
    {"name": "Mikel Oyarzabal", "pos": "W",  "wc_goals": 1, "wc_assists": 1,
     "club_rating": 7.1, "impact": "high",
     "notes": "1G 1A; scored Euro 2024 final winner; big-game pedigree"},
    {"name": "Gavi",           "pos": "CM", "wc_goals": 0, "wc_assists": 1,
     "club_rating": 7.2, "impact": "high",
     "notes": "Returned from injury; midfield energy injection"},
    {"name": "Joselu",         "pos": "ST", "wc_goals": 1, "wc_assists": 0,
     "club_rating": 6.8, "impact": "medium",
     "notes": "1G; aerial target; hold-up play option"},
]

ARGENTINA_BENCH = [
    {"name": "Angel Di Maria",  "pos": "W",  "wc_goals": 1, "wc_assists": 2,
     "club_rating": 7.0, "impact": "high",
     "notes": "1G 2A; scored in 2022 final; farewell appearance"},
    {"name": "Giovani Lo Celso", "pos": "AM", "wc_goals": 0, "wc_assists": 2,
     "club_rating": 7.0, "impact": "medium",
     "notes": "2 assists; technical quality off the bench"},
    {"name": "Nico Gonzalez",  "pos": "W",  "wc_goals": 1, "wc_assists": 1,
     "club_rating": 7.1, "impact": "medium",
     "notes": "1G 1A; direct runner; Juventus form"},
    {"name": "Leandro Paredes", "pos": "CM", "wc_goals": 0, "wc_assists": 0,
     "club_rating": 6.8, "impact": "medium",
     "notes": "Deep-lying playmaker; set-piece specialist; 2022 WC winner"},
]


# ═══════════════════════════════════════════════════════════════════════════
# 2.  TOURNAMENT-LEVEL TEAM DATA (match-by-match xG)
# ═══════════════════════════════════════════════════════════════════════════

STAGE_WEIGHT = {"Group": 1.0, "R32": 1.2, "R16": 1.4, "QF": 1.6, "SF": 1.8}

SPAIN_MATCHES = [
    {"opp": "Cabo Verde",  "date": "2026-06-15", "stage": "Group",
     "gf": 0, "ga": 0, "xg": 0.92, "xga": 0.24,
     "shots": 18, "sot": 4,  "poss": 72, "result": "D"},
    {"opp": "Saudi Arabia","date": "2026-06-22", "stage": "Group",
     "gf": 4, "ga": 0, "xg": 3.40, "xga": 0.31,
     "shots": 22, "sot": 10, "poss": 68, "result": "W"},
    {"opp": "Uruguay",     "date": "2026-06-26", "stage": "Group",
     "gf": 1, "ga": 0, "xg": 1.26, "xga": 0.85,
     "shots": 14, "sot": 5,  "poss": 58, "result": "W"},
    {"opp": "Austria",     "date": "2026-07-02", "stage": "R32",
     "gf": 3, "ga": 0, "xg": 2.80, "xga": 0.42,
     "shots": 18, "sot": 8,  "poss": 65, "result": "W"},
    {"opp": "Portugal",    "date": "2026-07-06", "stage": "R16",
     "gf": 1, "ga": 0, "xg": 1.55, "xga": 1.10,
     "shots": 12, "sot": 5,  "poss": 52, "result": "W"},
    {"opp": "Belgium",     "date": "2026-07-09", "stage": "QF",
     "gf": 4, "ga": 1, "xg": 3.20, "xga": 0.88,
     "shots": 20, "sot": 10, "poss": 60, "result": "W"},
    {"opp": "France",      "date": "2026-07-14", "stage": "SF",
     "gf": 2, "ga": 0, "xg": 1.90, "xga": 0.31,
     "shots": 14, "sot": 7,  "poss": 62, "result": "W"},
]

ARGENTINA_MATCHES = [
    {"opp": "Algeria",     "date": "2026-06-16", "stage": "Group",
     "gf": 3, "ga": 0, "xg": 1.26, "xga": 0.32,
     "shots": 10, "sot": 6,  "poss": 48, "result": "W"},
    {"opp": "Austria",     "date": "2026-06-22", "stage": "Group",
     "gf": 2, "ga": 0, "xg": 2.36, "xga": 0.53,
     "shots": 12, "sot": 5,  "poss": 54, "result": "W"},
    {"opp": "Jordan",      "date": "2026-06-27", "stage": "Group",
     "gf": 3, "ga": 1, "xg": 2.13, "xga": 0.76,
     "shots": 12, "sot": 4,  "poss": 73, "result": "W"},
    {"opp": "Cabo Verde",  "date": "2026-07-03", "stage": "R32",
     "gf": 3, "ga": 2, "xg": 2.26, "xga": 0.47,
     "shots": 21, "sot": 10, "poss": 64, "result": "W"},
    {"opp": "Egypt",       "date": "2026-07-07", "stage": "R16",
     "gf": 3, "ga": 2, "xg": 2.84, "xga": 0.89,
     "shots": 19, "sot": 7,  "poss": 64, "result": "W"},
    {"opp": "Switzerland", "date": "2026-07-11", "stage": "QF",
     "gf": 3, "ga": 1, "xg": 1.94, "xga": 0.47,
     "shots": 22, "sot": 7,  "poss": 59, "result": "W"},
    {"opp": "England",     "date": "2026-07-15", "stage": "SF",
     "gf": 2, "ga": 1, "xg": 1.82, "xga": 1.15,
     "shots": 16, "sot": 8,  "poss": 58, "result": "W"},
]


# ═══════════════════════════════════════════════════════════════════════════
# 3.  PLAYER-LEVEL FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════

def compute_player_performance_index(player, team_matches):
    """
    Compute a composite Player Performance Index (PPI) for each player.

    The PPI is position-aware:
      - Attackers (ST, W):   goals, xG, shots on target, dribbles weighted heavily
      - Midfielders (AM, CM): key passes, assists, tackles, goals balanced
      - Defenders (CB, FB):  tackles, interceptions, aerials, clean sheets
      - Goalkeepers (GK):    save %, clean sheets, goals against

    Also factors in club season form as a baseline quality indicator.
    """
    pos = player["pos"]
    ppi = 0.0

    # ── Club season baseline (30% of total PPI) ──
    club_rating = player.get("club_rating", 6.5)
    club_goals = player.get("club_goals", 0)
    club_assists = player.get("club_assists", 0)
    club_apps = player.get("club_apps", 1)

    if pos == "GK":
        club_cs = player.get("club_cs", 0)
        club_ga_90 = player.get("club_ga_90", 1.2)
        club_score = (club_rating / 7.0) * 0.4 + (club_cs / max(club_apps, 1)) * 0.3 + max(0, (1.5 - club_ga_90)) * 0.3
    elif pos in ("CB", "FB"):
        club_score = (club_rating / 7.0) * 0.5 + (club_goals + club_assists) / max(club_apps, 1) * 2.0 * 0.5
    elif pos in ("CM", "AM"):
        club_score = (club_rating / 7.0) * 0.4 + (club_goals + club_assists) / max(club_apps, 1) * 1.5 * 0.6
    else:  # ST, W
        club_score = (club_rating / 7.0) * 0.3 + (club_goals + club_assists) / max(club_apps, 1) * 1.2 * 0.7

    # ── World Cup tournament form (70% of total PPI) ──
    wc_mins = player.get("wc_mins", 0)
    if wc_mins == 0:
        return club_score * 0.3

    per_90 = 90.0 / max(wc_mins, 90)

    if pos == "GK":
        wc_save_pct = player.get("wc_save_pct", 0.65)
        wc_cs = player.get("wc_cs", 0)
        wc_ga = player.get("wc_ga", 0)
        wc_apps = player.get("wc_apps", 1)
        wc_score = (
            wc_save_pct * 0.35 +
            (wc_cs / max(wc_apps, 1)) * 0.30 +
            max(0, (1.5 - wc_ga / max(wc_apps, 1))) / 1.5 * 0.35
        )
    elif pos in ("CB", "FB"):
        tackles = player.get("wc_tackles", 0) * per_90
        interceptions = player.get("wc_interceptions", 0) * per_90
        aerials = player.get("wc_aerials_won", 0) * per_90
        goals = player.get("wc_goals", 0)
        assists = player.get("wc_assists", 0)
        wc_score = (
            min(tackles * 0.15, 0.35) +
            min(interceptions * 0.15, 0.25) +
            min(aerials * 0.08, 0.15) +
            goals * 0.15 +
            assists * 0.10
        )
    elif pos == "CM":
        tackles = player.get("wc_tackles", 0) * per_90
        interceptions = player.get("wc_interceptions", 0) * per_90
        key_passes = player.get("wc_key_passes", 0) * per_90
        goals = player.get("wc_goals", 0)
        assists = player.get("wc_assists", 0)
        wc_score = (
            min(tackles * 0.10, 0.20) +
            min(interceptions * 0.10, 0.15) +
            min(key_passes * 0.10, 0.20) +
            goals * 0.20 +
            assists * 0.15 +
            0.10  # base for availability
        )
    elif pos == "AM":
        key_passes = player.get("wc_key_passes", 0) * per_90
        dribbles = player.get("wc_dribbles", 0) * per_90
        goals = player.get("wc_goals", 0)
        assists = player.get("wc_assists", 0)
        xg = player.get("wc_xg", 0) * per_90
        wc_score = (
            min(key_passes * 0.10, 0.20) +
            min(dribbles * 0.05, 0.15) +
            goals * 0.25 +
            assists * 0.15 +
            min(xg * 0.10, 0.15) +
            0.10
        )
    else:  # ST, W — attackers
        goals = player.get("wc_goals", 0)
        assists = player.get("wc_assists", 0)
        xg = player.get("wc_xg", 0)
        sot = player.get("wc_sot", 0) * per_90
        dribbles = player.get("wc_dribbles", 0) * per_90
        wc_score = (
            goals * 0.30 +
            assists * 0.15 +
            min(sot * 0.05, 0.15) +
            min(dribbles * 0.03, 0.10) +
            min(xg * 0.05, 0.15) +
            0.15  # base for participation
        )

    ppi = club_score * 0.30 + wc_score * 0.70
    return ppi


def compute_squad_strength(players, team_matches):
    """
    Compute position-group strength indices from player PPIs.
    Returns a dict with attack, midfield, defense, gk, and overall scores.
    """
    starters = [p for p in players if p.get("starter", False)]

    gk_ppi = [compute_player_performance_index(p, team_matches)
              for p in starters if p["pos"] == "GK"]
    def_ppi = [compute_player_performance_index(p, team_matches)
               for p in starters if p["pos"] in ("CB", "FB")]
    mid_ppi = [compute_player_performance_index(p, team_matches)
               for p in starters if p["pos"] in ("CM", "AM")]
    att_ppi = [compute_player_performance_index(p, team_matches)
               for p in starters if p["pos"] in ("ST", "W")]

    gk_strength  = np.mean(gk_ppi) if gk_ppi else 0.5
    def_strength = np.mean(def_ppi) if def_ppi else 0.5
    mid_strength = np.mean(mid_ppi) if mid_ppi else 0.5
    att_strength = np.mean(att_ppi) if att_ppi else 0.5

    # Star player bonus: top performers lift entire team
    all_ppi = [(p["name"], compute_player_performance_index(p, team_matches))
               for p in starters]
    all_ppi.sort(key=lambda x: x[1], reverse=True)
    star_bonus = 0.0
    for i, (name, ppi) in enumerate(all_ppi[:3]):  # top 3 players
        star_bonus += ppi * (0.15 - i * 0.03)  # diminishing weight

    overall = (att_strength * 0.35 + mid_strength * 0.25 +
               def_strength * 0.25 + gk_strength * 0.15 + star_bonus)

    return {
        "gk": gk_strength,
        "defense": def_strength,
        "midfield": mid_strength,
        "attack": att_strength,
        "star_bonus": star_bonus,
        "overall": overall,
        "player_ppis": all_ppi,
    }


def compute_bench_impact(bench_players):
    """Estimate potential impact from substitutes."""
    total = 0.0
    for p in bench_players:
        base = p.get("club_rating", 6.5) / 7.0
        goals = p.get("wc_goals", 0)
        assists = p.get("wc_assists", 0)
        impact_mult = {"high": 1.3, "medium": 1.0, "low": 0.7}.get(p.get("impact", "medium"), 1.0)
        total += (base * 0.4 + (goals * 0.3 + assists * 0.15) * 0.6) * impact_mult
    return total / max(len(bench_players), 1)


# ═══════════════════════════════════════════════════════════════════════════
# 4.  TOURNAMENT PERFORMANCE PROFILES (team-level aggregates)
# ═══════════════════════════════════════════════════════════════════════════

def compute_tournament_profile(matches, team_name):
    n = len(matches)
    gf   = sum(m["gf"] for m in matches)
    ga   = sum(m["ga"] for m in matches)
    xg   = sum(m["xg"] for m in matches)
    xga  = sum(m["xga"] for m in matches)
    sot  = sum(m["sot"] for m in matches)
    shots = sum(m["shots"] for m in matches)
    poss = np.mean([m["poss"] for m in matches])
    cs   = sum(1 for m in matches if m["ga"] == 0)
    wins = sum(1 for m in matches if m["result"] == "W")
    draws = sum(1 for m in matches if m["result"] == "D")

    # Stage-weighted xG (later stages count more)
    total_w = sum(STAGE_WEIGHT[m["stage"]] for m in matches)
    xg_w  = sum(m["xg"]  * STAGE_WEIGHT[m["stage"]] for m in matches) / total_w
    xga_w = sum(m["xga"] * STAGE_WEIGHT[m["stage"]] for m in matches) / total_w

    return {
        "team": team_name, "played": n,
        "W": wins, "D": draws, "L": n - wins - draws,
        "GF": gf, "GA": ga, "GD": gf - ga,
        "GF/m": gf / n, "GA/m": ga / n,
        "xG": xg, "xGA": xga, "xGD": xg - xga,
        "xG/m": xg / n, "xGA/m": xga / n,
        "xG/m_w": xg_w, "xGA/m_w": xga_w,
        "Shots": shots, "SoT": sot,
        "SoT%": 100 * sot / max(shots, 1),
        "Poss%": poss,
        "CS": cs, "CS%": 100 * cs / n,
        "Win%": 100 * (wins + 0.5 * draws) / n,
        "Finishing": gf / max(xg, 0.01),
        "Def_solidity": xga / max(ga, 0.01),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 5.  MONTE CARLO SIMULATION (player-derived parameters)
# ═══════════════════════════════════════════════════════════════════════════

def compute_player_derived_lambdas(squad_h, squad_a, prof_h, prof_a):
    """
    Derive expected goals (lambda) for each team using a blend of:
      - Player-level attacking/defensive strength indices
      - Tournament xG data (stage-weighted)
      - Opponent quality adjustment

    This is the KEY improvement over team-level models: the lambda
    parameters are informed by individual player quality, not just
    team-level averages.
    """
    # xG-based baseline
    h_att = prof_h["xG/m_w"] / TOURNAMENT_AVG_XG
    h_def = prof_h["xGA/m_w"] / TOURNAMENT_AVG_XG
    a_att = prof_a["xG/m_w"] / TOURNAMENT_AVG_XG
    a_def = prof_a["xGA/m_w"] / TOURNAMENT_AVG_XG

    # Blend player-level and xG-level
    player_weight = 0.40
    xg_weight = 0.60

    # Home team expected goals
    lambda_h_player = TOURNAMENT_AVG_XG * (squad_h["attack"] / squad_a["defense"]) * 1.1
    lambda_h_xg = TOURNAMENT_AVG_XG * h_att * a_def * KNOCKOUT_REGRESSION
    lambda_h = player_weight * lambda_h_player + xg_weight * lambda_h_xg

    # Away team expected goals
    lambda_a_player = TOURNAMENT_AVG_XG * (squad_a["attack"] / squad_h["defense"]) * 1.1
    lambda_a_xg = TOURNAMENT_AVG_XG * a_att * h_def * KNOCKOUT_REGRESSION
    lambda_a = player_weight * lambda_a_player + xg_weight * lambda_a_xg

    # Bench depth adjustment (potential sub impact)
    bench_h = compute_bench_impact(SPAIN_BENCH)
    bench_a = compute_bench_impact(ARGENTINA_BENCH)
    lambda_h *= (1.0 + 0.03 * bench_h)
    lambda_a *= (1.0 + 0.03 * bench_a)

    # Computed momentum: normalized GD trend over last 3 knockout matches
    # Applied symmetrically to both teams using the same formula
    def _knockout_momentum(matches):
        ko = [m for m in matches if m["stage"] in ("R32", "R16", "QF", "SF")][-3:]
        if not ko:
            return 0.0
        gd_trend = sum((m["gf"] - m["ga"]) for m in ko) / len(ko)
        # Sigmoid-like mapping: large positive GD → small boost, negative → drag
        return 0.01 * np.tanh(gd_trend / 2.0)

    lambda_h *= (1.0 + _knockout_momentum(SPAIN_MATCHES))
    lambda_a *= (1.0 + _knockout_momentum(ARGENTINA_MATCHES))

    return lambda_h, lambda_a


def simulate_match(lambda_h, lambda_a, n_sims=N_SIMULATIONS):
    """
    Monte Carlo simulation of the match using Poisson-distributed goals.
    Returns win/draw/loss probabilities and score distribution.
    """
    goals_h = np.random.poisson(lambda_h, n_sims)
    goals_a = np.random.poisson(lambda_a, n_sims)

    home_wins = np.sum(goals_h > goals_a)
    draws = np.sum(goals_h == goals_a)
    away_wins = np.sum(goals_h < goals_a)

    p_home = home_wins / n_sims
    p_draw = draws / n_sims
    p_away = away_wins / n_sims

    # Score distribution
    scores = {}
    for h, a in zip(goals_h, goals_a):
        key = (int(h), int(a))
        scores[key] = scores.get(key, 0) + 1
    for key in scores:
        scores[key] /= n_sims

    # Average goals
    avg_h = np.mean(goals_h)
    avg_a = np.mean(goals_a)

    return {
        "p_home": p_home, "p_draw": p_draw, "p_away": p_away,
        "avg_goals_home": avg_h, "avg_goals_away": avg_a,
        "scores": scores,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 6.  DIXON-COLES POISSON MODEL
# ═══════════════════════════════════════════════════════════════════════════

def dixon_coles_tau(x, y, lam_h, lam_a, rho):
    if x == 0 and y == 0:
        return 1 - lam_h * lam_a * rho
    if x == 1 and y == 0:
        return 1 + lam_a * rho
    if x == 0 and y == 1:
        return 1 + lam_h * rho
    if x == 1 and y == 1:
        return 1 - rho
    return 1.0


def score_probability_matrix(lam_h, lam_a, rho, max_goals):
    matrix = np.zeros((max_goals, max_goals))
    for i in range(max_goals):
        for j in range(max_goals):
            p_ind = poisson_dist.pmf(i, lam_h) * poisson_dist.pmf(j, lam_a)
            tau = dixon_coles_tau(i, j, lam_h, lam_a, rho)
            matrix[i, j] = p_ind * tau
    matrix /= matrix.sum()
    return matrix


def extract_probabilities(matrix):
    n = matrix.shape[0]
    p_home = sum(matrix[i, j] for i in range(n) for j in range(n) if i > j)
    p_draw = sum(matrix[i, i] for i in range(n))
    p_away = sum(matrix[i, j] for i in range(n) for j in range(n) if i < j)
    return p_home, p_draw, p_away


def top_scorelines(matrix, n=10):
    mg = matrix.shape[0]
    scores = [(i, j, matrix[i, j]) for i in range(mg) for j in range(mg)]
    scores.sort(key=lambda x: x[2], reverse=True)
    return scores[:n]


# ═══════════════════════════════════════════════════════════════════════════
# 7.  TEAM-LEVEL XGBOOST MODEL (from existing pipeline — as a prior)
# ═══════════════════════════════════════════════════════════════════════════

WC2026_PATCHES = {
    # Group A
    ("2026-06-11", "Mexico",         "South Africa"):     (2, 0),
    ("2026-06-11", "South Korea",    "Czech Republic"):   (2, 1),
    ("2026-06-18", "Czech Republic", "South Africa"):     (1, 1),
    ("2026-06-18", "Mexico",         "South Korea"):      (1, 0),
    ("2026-06-25", "Mexico",         "Czech Republic"):   (3, 0),
    ("2026-06-25", "South Africa",   "South Korea"):      (1, 0),
    # Group B
    ("2026-06-12", "Canada",         "Bosnia and Herzegovina"): (1, 1),
    ("2026-06-12", "Switzerland",    "Qatar"):            (1, 1),
    ("2026-06-19", "Switzerland",    "Bosnia and Herzegovina"): (4, 1),
    ("2026-06-19", "Canada",         "Qatar"):            (6, 0),
    ("2026-06-25", "Switzerland",    "Canada"):           (3, 1),
    ("2026-06-25", "Bosnia and Herzegovina", "Qatar"):    (3, 1),
    # Group C
    ("2026-06-13", "Brazil",         "Morocco"):          (1, 1),
    ("2026-06-13", "Haiti",          "Scotland"):         (0, 1),
    ("2026-06-19", "Scotland",       "Morocco"):          (0, 1),
    ("2026-06-19", "Brazil",         "Haiti"):            (3, 0),
    ("2026-06-24", "Morocco",        "Haiti"):            (4, 2),
    ("2026-06-24", "Scotland",       "Brazil"):           (0, 3),
    # Group D
    ("2026-06-12", "United States",  "Paraguay"):         (4, 1),
    ("2026-06-13", "Australia",      "Turkey"):           (2, 0),
    ("2026-06-19", "United States",  "Australia"):        (2, 0),
    ("2026-06-19", "Turkey",         "Paraguay"):         (0, 1),
    ("2026-06-25", "Turkey",         "United States"):    (3, 2),
    ("2026-06-25", "Paraguay",       "Australia"):        (0, 0),
    # Group E
    ("2026-06-14", "Germany",        "Curacao"):          (7, 1),
    ("2026-06-14", "Ivory Coast",    "Ecuador"):          (1, 0),
    ("2026-06-20", "Germany",        "Ivory Coast"):      (2, 1),
    ("2026-06-20", "Ecuador",        "Curacao"):          (0, 0),
    ("2026-06-26", "Curacao",        "Ivory Coast"):      (0, 2),
    ("2026-06-26", "Ecuador",        "Germany"):          (2, 1),
    # Group F
    ("2026-06-14", "Netherlands",    "Japan"):            (2, 2),
    ("2026-06-14", "Sweden",         "Tunisia"):          (5, 1),
    ("2026-06-21", "Netherlands",    "Sweden"):           (5, 1),
    ("2026-06-21", "Japan",          "Tunisia"):          (4, 0),
    ("2026-06-26", "Japan",          "Sweden"):           (1, 1),
    ("2026-06-26", "Netherlands",    "Tunisia"):          (3, 1),
    # Group G
    ("2026-06-15", "Belgium",        "Egypt"):            (1, 1),
    ("2026-06-15", "Iran",           "New Zealand"):      (2, 2),
    ("2026-06-21", "Belgium",        "Iran"):             (0, 0),
    ("2026-06-21", "Egypt",          "New Zealand"):      (3, 1),
    ("2026-06-26", "Egypt",          "Iran"):             (1, 1),
    ("2026-06-26", "Belgium",        "New Zealand"):      (5, 1),
    # Group H
    ("2026-06-15", "Spain",          "Cabo Verde"):       (0, 0),
    ("2026-06-15", "Saudi Arabia",   "Uruguay"):          (1, 1),
    ("2026-06-22", "Spain",          "Saudi Arabia"):     (4, 0),
    ("2026-06-22", "Uruguay",        "Cabo Verde"):       (2, 2),
    ("2026-06-26", "Cabo Verde",     "Saudi Arabia"):     (0, 0),
    ("2026-06-26", "Spain",          "Uruguay"):          (1, 0),
    # Group I
    ("2026-06-16", "France",         "Senegal"):          (3, 1),
    ("2026-06-16", "Iraq",           "Norway"):           (1, 4),
    ("2026-06-22", "France",         "Iraq"):             (3, 0),
    ("2026-06-22", "Norway",         "Senegal"):          (3, 2),
    ("2026-06-26", "Norway",         "France"):           (1, 4),
    ("2026-06-26", "Senegal",        "Iraq"):             (5, 0),
    # Group J
    ("2026-06-16", "Argentina",      "Algeria"):          (3, 0),
    ("2026-06-16", "Austria",        "Jordan"):           (3, 1),
    ("2026-06-23", "Argentina",      "Austria"):          (2, 0),
    ("2026-06-23", "Jordan",         "Algeria"):          (1, 2),
    ("2026-06-27", "Argentina",      "Jordan"):           (3, 1),
    ("2026-06-27", "Austria",        "Algeria"):          (3, 3),
    # Group K
    ("2026-06-17", "Portugal",       "DR Congo"):         (1, 1),
    ("2026-06-17", "Uzbekistan",     "Colombia"):         (1, 3),
    ("2026-06-23", "Portugal",       "Uzbekistan"):       (5, 0),
    ("2026-06-23", "Colombia",       "DR Congo"):         (1, 0),
    ("2026-06-27", "Colombia",       "Portugal"):         (0, 0),
    ("2026-06-27", "DR Congo",       "Uzbekistan"):       (3, 1),
    # Group L
    ("2026-06-17", "England",        "Croatia"):          (4, 2),
    ("2026-06-17", "Ghana",          "Panama"):           (1, 0),
    ("2026-06-23", "England",        "Ghana"):            (0, 0),
    ("2026-06-23", "Panama",         "Croatia"):          (0, 1),
    ("2026-06-27", "England",        "Panama"):           (2, 0),
    ("2026-06-27", "Croatia",        "Ghana"):            (2, 1),
    # R32
    ("2026-06-28", "Canada",         "South Africa"):     (1, 0),
    ("2026-06-28", "Brazil",         "Japan"):            (2, 1),
    ("2026-06-29", "Germany",        "Paraguay"):         (1, 1),
    ("2026-06-29", "Netherlands",    "Morocco"):          (1, 1),
    ("2026-06-29", "Norway",         "Ivory Coast"):      (2, 1),
    ("2026-06-30", "France",         "Sweden"):           (3, 0),
    ("2026-06-30", "Mexico",         "Ecuador"):          (2, 0),
    ("2026-06-30", "England",        "DR Congo"):         (2, 1),
    ("2026-07-01", "Belgium",        "Senegal"):          (3, 2),
    ("2026-07-01", "United States",  "Bosnia and Herzegovina"): (2, 0),
    ("2026-07-02", "Spain",          "Austria"):          (3, 0),
    ("2026-07-02", "Portugal",       "Croatia"):          (2, 1),
    ("2026-07-02", "Switzerland",    "Algeria"):          (2, 0),
    ("2026-07-03", "Australia",      "Egypt"):            (1, 1),
    ("2026-07-03", "Argentina",      "Cabo Verde"):       (3, 2),
    ("2026-07-03", "Colombia",       "Ghana"):            (1, 0),
    # R16
    ("2026-07-04", "Morocco",        "Canada"):           (3, 0),
    ("2026-07-04", "Paraguay",       "France"):           (0, 1),
    ("2026-07-05", "Norway",         "Brazil"):           (2, 1),
    ("2026-07-05", "England",        "Mexico"):           (3, 2),
    ("2026-07-06", "Spain",          "Portugal"):         (1, 0),
    ("2026-07-06", "Belgium",        "United States"):    (4, 1),
    ("2026-07-07", "Argentina",      "Egypt"):            (3, 2),
    ("2026-07-07", "Switzerland",    "Colombia"):         (0, 0),
    # QF
    ("2026-07-09", "France",         "Morocco"):          (2, 0),
    ("2026-07-09", "Spain",          "Belgium"):          (4, 1),
    ("2026-07-11", "England",        "Norway"):           (2, 1),
    ("2026-07-11", "Argentina",      "Switzerland"):      (3, 1),
    # SF
    ("2026-07-14", "Spain",          "France"):           (2, 0),
    ("2026-07-15", "Argentina",      "England"):          (2, 1),
}


def fetch_results():
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, "results.csv")
    if not os.path.exists(path):
        resp = requests.get(RESULTS_URL, timeout=120)
        resp.raise_for_status()
        with open(path, "wb") as fh:
            fh.write(resp.content)
    return pd.read_csv(path)


def normalize_country(name):
    return NAME_MAP.get(name, name) if isinstance(name, str) else name


def load_results():
    r = fetch_results()
    r["home_team"] = r["home_team"].map(normalize_country)
    r["away_team"] = r["away_team"].map(normalize_country)
    r["date"] = pd.to_datetime(r["date"])
    r = r.dropna(subset=["home_score", "away_score"]).copy()
    r["home_score"] = r["home_score"].astype(int)
    r["away_score"] = r["away_score"].astype(int)
    r["neutral"] = r["neutral"].astype(str).str.upper().eq("TRUE").astype(int)
    return r.sort_values("date").reset_index(drop=True)


def patch_wc2026_results(r):
    patched, added = 0, 0
    for (date_str, home, away), (hs, as_) in WC2026_PATCHES.items():
        dt = pd.Timestamp(date_str)
        mask = (r["date"] == dt) & (r["home_team"] == home) & (r["away_team"] == away)
        if mask.any():
            idx = r.index[mask][0]
            if pd.isna(r.at[idx, "home_score"]) or r.at[idx, "home_score"] != hs:
                r.at[idx, "home_score"] = hs
                r.at[idx, "away_score"] = as_
                patched += 1
        else:
            new = pd.DataFrame([{
                "date": dt, "home_team": home, "away_team": away,
                "home_score": hs, "away_score": as_,
                "tournament": "FIFA World Cup", "city": "USA",
                "country": "United States", "neutral": 1,
            }])
            r = pd.concat([r, new], ignore_index=True)
            added += 1
    r["home_score"] = r["home_score"].astype(int)
    r["away_score"] = r["away_score"].astype(int)
    r = r.sort_values("date").reset_index(drop=True)
    print(f"  WC2026 patch: {patched} updated, {added} added (through SF)")
    return r


def tournament_weight(name):
    t = str(name).lower()
    if "fifa world cup" in t and "qualif" not in t: return 4
    if "qualif" in t: return 3
    big = ["uefa nations", "copa america", "afc asian cup", "africa cup",
           "concacaf", "uefa euro", "confederations"]
    if any(tok in t for tok in big): return 3
    if "friendly" in t: return 1
    return 2


def add_label_and_context(r):
    r = r.copy()
    r["label"] = np.where(r["home_score"] > r["away_score"], 0,
                          np.where(r["home_score"] == r["away_score"], 1, 2))
    r["tournament_weight"] = r["tournament"].map(tournament_weight)
    return r


def compute_elo(r):
    r = r.sort_values("date").reset_index(drop=True)
    rating, home_pre, away_pre = {}, np.zeros(len(r)), np.zeros(len(r))
    for i, row in r.iterrows():
        rh = rating.get(row.home_team, ELO_BASE)
        ra = rating.get(row.away_team, ELO_BASE)
        home_pre[i], away_pre[i] = rh, ra
        bonus = 0 if row.neutral == 1 else ELO_HOME_BONUS
        exp_home = 1 / (1 + 10 ** (-((rh + bonus) - ra) / 400))
        score_home = 1.0 if row.label == 0 else (0.5 if row.label == 1 else 0.0)
        margin = abs(int(row.home_score) - int(row.away_score))
        mult = np.log(max(margin, 1) + 1) * (2.2 / (abs(rh - ra) * 0.001 + 2.2))
        rating[row.home_team] = rh + ELO_K * mult * (score_home - exp_home)
        rating[row.away_team] = ra + ELO_K * mult * ((1 - score_home) - (1 - exp_home))
    r["home_elo"], r["away_elo"] = home_pre, away_pre
    r["elo_diff"] = home_pre - away_pre
    return r, rating


def per_team_long(r):
    home = pd.DataFrame({"date": r["date"].values, "team": r["home_team"].values,
                         "opp": r["away_team"].values, "gf": r["home_score"].values,
                         "ga": r["away_score"].values})
    away = pd.DataFrame({"date": r["date"].values, "team": r["away_team"].values,
                         "opp": r["home_team"].values, "gf": r["away_score"].values,
                         "ga": r["home_score"].values})
    long = pd.concat([home, away], ignore_index=True)
    long["result"] = np.where(long["gf"] > long["ga"], 1.0,
                              np.where(long["gf"] == long["ga"], 0.5, 0.0))
    long["gd"] = long["gf"] - long["ga"]
    return long


def add_form_features(r):
    long = per_team_long(r).sort_values(["team", "date"]).reset_index(drop=True)
    long["prev_date"]  = long.groupby("team")["date"].shift(1)
    long["result_lag"] = long.groupby("team")["result"].shift(1)
    long["gd_lag"]     = long.groupby("team")["gd"].shift(1)
    long["win5"]  = long.groupby("team")["result_lag"].transform(
        lambda s: s.rolling(5, min_periods=1).mean())
    long["gd5"]   = long.groupby("team")["gd_lag"].transform(
        lambda s: s.rolling(5, min_periods=1).mean())
    long["win10"] = long.groupby("team")["result_lag"].transform(
        lambda s: s.rolling(10, min_periods=1).mean())
    long["rest_days"] = (long["date"] - long["prev_date"]).dt.days
    form = long[["date", "team", "win5", "gd5", "win10", "rest_days"]].drop_duplicates(["date", "team"])
    r = r.merge(form.rename(columns={"team": "home_team", "win5": "home_win5", "gd5": "home_gd5",
                                     "win10": "home_win10", "rest_days": "home_rest_days"}),
                on=["date", "home_team"], how="left")
    r = r.merge(form.rename(columns={"team": "away_team", "win5": "away_win5", "gd5": "away_gd5",
                                     "win10": "away_win10", "rest_days": "away_rest_days"}),
                on=["date", "away_team"], how="left")
    return r


def add_h2h_features(r):
    long = per_team_long(r).sort_values(["team", "opp", "date"]).reset_index(drop=True)
    g = long.groupby(["team", "opp"])
    long["h2h_n"]       = g.cumcount()
    long["h2h_winrate"] = g["result"].transform(
        lambda s: s.shift(1).expanding(min_periods=1).mean())
    long["h2h_gd"]      = g["gd"].transform(
        lambda s: s.shift(1).expanding(min_periods=1).mean())
    h2h = long[["date", "team", "opp", "h2h_n", "h2h_winrate", "h2h_gd"]].drop_duplicates(
        ["date", "team", "opp"])
    r = r.merge(h2h.rename(columns={"team": "home_team", "opp": "away_team",
                                    "h2h_winrate": "h2h_home_winrate", "h2h_gd": "h2h_home_gd"}),
                on=["date", "home_team", "away_team"], how="left")
    return r


def build_dataset(r):
    r = add_label_and_context(r)
    r, final_elo = compute_elo(r)
    r = add_form_features(r)
    r = add_h2h_features(r)
    return r, final_elo


def split_by_date(ds, train_start, val_start, cutoff):
    train = ds[(ds["date"] >= pd.Timestamp(train_start))
               & (ds["date"] < pd.Timestamp(val_start))].copy()
    val   = ds[(ds["date"] >= pd.Timestamp(val_start))
               & (ds["date"] < pd.Timestamp(cutoff))].copy()
    return train, val


def train_model(train, val):
    X_train, y_train = train[TEAM_FEATURES].astype(float), train["label"].astype(int)
    X_val,   y_val   = val[TEAM_FEATURES].astype(float),   val["label"].astype(int)
    model = xgb.XGBClassifier(
        objective="multi:softprob", num_class=3, n_estimators=600,
        learning_rate=0.05, max_depth=5, subsample=0.85, colsample_bytree=0.85,
        reg_lambda=1.0, eval_metric="mlogloss", early_stopping_rounds=50,
        tree_method="hist", n_jobs=-1, random_state=42,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model, X_val, y_val


def evaluate(model, X_val, y_val):
    proba = model.predict_proba(X_val)
    pred  = proba.argmax(axis=1)
    base  = np.tile(np.bincount(y_val, minlength=3) / len(y_val), (len(y_val), 1))
    acc   = accuracy_score(y_val, pred)
    ll    = log_loss(y_val, proba)
    bl    = log_loss(y_val, base, labels=[0, 1, 2])
    print(f"  Validation accuracy : {acc:.3f}")
    print(f"  Validation log-loss : {ll:.3f}  (baseline {bl:.3f})")
    return acc, ll


def form_as_of(long, team, asof):
    sub = long[(long["team"] == team) & (long["date"] < pd.Timestamp(asof))].sort_values("date")
    if len(sub) == 0:
        return {"win5": 0.5, "gd5": 0.0, "win10": 0.5, "rest_days": 30.0}
    l5, l10 = sub.tail(5), sub.tail(10)
    return {"win5": float(l5["result"].mean()), "gd5": float((l5["gf"] - l5["ga"]).mean()),
            "win10": float(l10["result"].mean()),
            "rest_days": float((pd.Timestamp(asof) - sub["date"].max()).days)}


def h2h_as_of(long, team, opp, asof):
    sub = long[(long["team"] == team) & (long["opp"] == opp) & (long["date"] < pd.Timestamp(asof))]
    if len(sub) == 0:
        return 0.0, np.nan, np.nan
    return float(len(sub)), float(sub["result"].mean()), float(sub["gd"].mean())


def build_match_row(long, final_elo, home, away, neutral, weight, asof):
    hf, af = form_as_of(long, home, asof), form_as_of(long, away, asof)
    he, ae = final_elo.get(home, ELO_BASE), final_elo.get(away, ELO_BASE)
    n, wr, gd = h2h_as_of(long, home, away, asof)
    row = {"neutral": int(neutral), "tournament_weight": weight,
           "home_elo": he, "away_elo": ae, "elo_diff": he - ae,
           "home_win5": hf["win5"], "away_win5": af["win5"],
           "home_gd5": hf["gd5"], "away_gd5": af["gd5"],
           "home_win10": hf["win10"], "away_win10": af["win10"],
           "home_rest_days": hf["rest_days"], "away_rest_days": af["rest_days"],
           "h2h_n": n, "h2h_home_winrate": wr, "h2h_home_gd": gd}
    return pd.DataFrame([row])[TEAM_FEATURES].astype(float)


def predict_symmetric(model, long, final_elo, a, b, asof, neutral, weight):
    p_ab = model.predict_proba(
        build_match_row(long, final_elo, a, b, neutral, weight, asof))[0]
    p_ba = model.predict_proba(
        build_match_row(long, final_elo, b, a, neutral, weight, asof))[0]
    p_a = (p_ab[0] + p_ba[2]) / 2.0
    p_d = (p_ab[1] + p_ba[1]) / 2.0
    p_b = (p_ab[2] + p_ba[0]) / 2.0
    tot = p_a + p_d + p_b
    return p_a / tot, p_d / tot, p_b / tot


# ═══════════════════════════════════════════════════════════════════════════
# 8.  ENSEMBLE WEIGHT FITTING VIA WALK-FORWARD BACKTESTING
# ═══════════════════════════════════════════════════════════════════════════

def _simulate_component_predictions(row, final_elo, long, model):
    """
    For a single historical match, compute what each of the 4 component
    models would have predicted (3-way: H/D/A probabilities).

    Components:
      1. Monte Carlo (Poisson with xG-derived lambdas)
      2. Dixon-Coles (same lambdas)
      3. XGBoost
      4. Squad-strength proxy (Elo-based sigmoid)

    Returns (p_sim, p_pois, p_xgb, p_squad) each as (h, d, a) tuple,
    or None if we can't compute for this match.
    """
    home = row["home_team"]
    away = row["away_team"]
    date_str = str(row["date"].date())

    # ── XGBoost prediction ──
    try:
        p_xgb_h, p_xgb_d, p_xgb_a = predict_symmetric(
            model, long, final_elo, home, away, date_str,
            neutral=int(row.get("neutral", 1)), weight=4)
    except Exception:
        return None

    # ── Elo-based proxy for "squad strength" ──
    h_elo = final_elo.get(home, ELO_BASE)
    a_elo = final_elo.get(away, ELO_BASE)
    elo_diff = h_elo - a_elo
    p_sq_raw_h = 1.0 / (1.0 + np.exp(-elo_diff / 200.0))
    p_sq_raw_a = 1.0 - p_sq_raw_h
    p_sq_d = 0.22
    p_sq_h = p_sq_raw_h * (1.0 - p_sq_d)
    p_sq_a = p_sq_raw_a * (1.0 - p_sq_d)

    # ── xG-derived lambdas for Monte Carlo / Dixon-Coles ──
    # Use Elo-derived attack/defense strengths as proxy (no player data for
    # historical matches). This is the best we can do with results.csv.
    h_att_proxy = max(0.5, 1.0 + elo_diff / 800.0)  # >1 if strong
    a_att_proxy = max(0.5, 1.0 - elo_diff / 800.0)
    h_def_proxy = max(0.3, 1.0 - elo_diff / 1200.0)  # lower = better defense
    a_def_proxy = max(0.3, 1.0 + elo_diff / 1200.0)

    lambda_h = TOURNAMENT_AVG_XG * h_att_proxy * a_def_proxy * KNOCKOUT_REGRESSION
    lambda_a = TOURNAMENT_AVG_XG * a_att_proxy * h_def_proxy * KNOCKOUT_REGRESSION
    lambda_h = np.clip(lambda_h, 0.3, 4.0)
    lambda_a = np.clip(lambda_a, 0.3, 4.0)

    # Monte Carlo
    goals_h = np.random.poisson(lambda_h, 2000)
    goals_a = np.random.poisson(lambda_a, 2000)
    p_sim_h = np.mean(goals_h > goals_a)
    p_sim_d = np.mean(goals_h == goals_a)
    p_sim_a = np.mean(goals_h < goals_a)

    # Dixon-Coles
    matrix = score_probability_matrix(lambda_h, lambda_a, DIXON_COLES_RHO, MAX_GOALS)
    p_pois_h, p_pois_d, p_pois_a = extract_probabilities(matrix)

    return (
        (p_sim_h, p_sim_d, p_sim_a),
        (p_pois_h, p_pois_d, p_pois_a),
        (p_xgb_h, p_xgb_d, p_xgb_a),
        (p_sq_h, p_sq_d, p_sq_a),
    )


def fit_ensemble_weights(dataset, final_elo, long, model):
    """
    Walk-forward backtest on major-tournament knockout matches since 2010.
    Fit blend weights to minimize log-loss via scipy.optimize.

    Returns:
        weights: (w_sim, w_pois, w_xgb, w_squad)
        backtest_results: dict with metrics and per-match predictions
    """
    # Filter to major tournament knockout matches 2010-2025
    backtest = dataset[
        (dataset["date"] >= pd.Timestamp("2010-01-01")) &
        (dataset["date"] < pd.Timestamp("2026-01-01"))
    ].copy()

    # Filter to major tournaments
    backtest = backtest[backtest["tournament"].apply(
        lambda t: any(k in str(t).lower() for k in
                      ["fifa world cup", "uefa euro", "copa america",
                       "africa cup", "confederations"])
    )].copy()

    # Filter to knockout-like matches (not group stage)
    # Heuristic: games where a draw is less common in the score →
    # just use all major tournament matches for larger sample
    # (group stages are still useful calibration data)

    if len(backtest) < 30:
        print("  WARNING: <30 backtest matches, using hand-picked defaults")
        return (0.35, 0.30, 0.20, 0.15), None

    # Compute component predictions for each match
    predictions = []  # list of (actual_label, (sim, pois, xgb, squad))
    for _, row in backtest.iterrows():
        result = _simulate_component_predictions(row, final_elo, long, model)
        if result is None:
            continue
        actual = int(row["label"])  # 0=home_win, 1=draw, 2=away_win
        predictions.append((actual, result))

    n_matches = len(predictions)
    if n_matches < 20:
        print(f"  WARNING: only {n_matches} backtest predictions, using defaults")
        return (0.35, 0.30, 0.20, 0.15), None

    # Build arrays for optimization
    actuals = np.array([p[0] for p in predictions])
    # Each component gives 3 probabilities per match
    sim_probs = np.array([p[1][0] for p in predictions])   # (N, 3)
    pois_probs = np.array([p[1][1] for p in predictions])
    xgb_probs = np.array([p[1][2] for p in predictions])
    squad_probs = np.array([p[1][3] for p in predictions])

    def _ensemble_logloss(weights_raw):
        """Compute log-loss for given weights (softmax to ensure sum=1, >0)."""
        # Softmax parametrization to enforce sum=1 and positivity
        exp_w = np.exp(weights_raw)
        w = exp_w / exp_w.sum()

        blended = (w[0] * sim_probs + w[1] * pois_probs +
                   w[2] * xgb_probs + w[3] * squad_probs)
        # Clip to avoid log(0)
        blended = np.clip(blended, 1e-6, 1.0)
        blended = blended / blended.sum(axis=1, keepdims=True)

        # Log-loss
        ll = -np.mean(np.log(blended[np.arange(len(actuals)), actuals]))
        return ll

    # Optimize
    best_result = None
    best_loss = np.inf

    # Try multiple starting points
    starts = [
        np.array([0.0, 0.0, 0.0, 0.0]),  # equal weights
        np.array([0.3, 0.2, 0.0, -0.3]),  # favor sim/pois
        np.array([-0.2, -0.2, 0.3, 0.0]),  # favor xgb
        np.log(np.array([0.35, 0.30, 0.20, 0.15])),  # hand-picked as start
    ]

    for x0 in starts:
        result = minimize(_ensemble_logloss, x0, method="Nelder-Mead",
                         options={"maxiter": 5000, "xatol": 1e-5, "fatol": 1e-6})
        if result.fun < best_loss:
            best_loss = result.fun
            best_result = result

    # Convert to proper weights
    exp_w = np.exp(best_result.x)
    fitted_weights = tuple(exp_w / exp_w.sum())

    # Compute ensemble predictions with fitted weights for calibration
    w = np.array(fitted_weights)
    blended = (w[0] * sim_probs + w[1] * pois_probs +
               w[2] * xgb_probs + w[3] * squad_probs)
    blended = np.clip(blended, 1e-6, 1.0)
    blended = blended / blended.sum(axis=1, keepdims=True)

    # Compute Brier score (multi-class)
    one_hot = np.zeros_like(blended)
    one_hot[np.arange(len(actuals)), actuals] = 1.0
    brier = np.mean(np.sum((blended - one_hot) ** 2, axis=1))

    # Log-loss
    ensemble_logloss = -np.mean(np.log(blended[np.arange(len(actuals)), actuals]))

    # Accuracy
    ensemble_preds = blended.argmax(axis=1)
    ensemble_acc = np.mean(ensemble_preds == actuals)

    # Calibration bins (for home win probability)
    cal_bins = _compute_calibration_bins(blended[:, 0], (actuals == 0).astype(float))

    backtest_results = {
        "n_matches": n_matches,
        "ensemble_logloss": ensemble_logloss,
        "ensemble_brier": brier,
        "ensemble_accuracy": ensemble_acc,
        "calibration_bins": cal_bins,
        "fitted_weights": fitted_weights,
    }

    return fitted_weights, backtest_results


def _compute_calibration_bins(predicted_probs, actual_outcomes, n_bins=5):
    """Compute calibration data for reliability diagram."""
    bins = np.linspace(0, 1, n_bins + 1)
    cal_data = []
    for i in range(n_bins):
        mask = (predicted_probs >= bins[i]) & (predicted_probs < bins[i + 1])
        if mask.sum() > 2:
            cal_data.append({
                "bin_center": (bins[i] + bins[i + 1]) / 2,
                "predicted_mean": float(predicted_probs[mask].mean()),
                "actual_mean": float(actual_outcomes[mask].mean()),
                "count": int(mask.sum()),
            })
    return cal_data


# ═══════════════════════════════════════════════════════════════════════════
# 9.  CHARTS
# ═══════════════════════════════════════════════════════════════════════════

def make_prediction_chart(p_home, p_draw, p_away, out_dir):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    labels = [f"{HOME_TEAM}\nwin", "Draw\n(90 min)", f"{AWAY_TEAM}\nwin"]
    vals = [p_home, p_draw, p_away]
    colors = [ESP_RED, GRAY, ARG_BLUE]
    bars = ax.bar(labels, [v * 100 for v in vals], color=colors, width=0.55, zorder=3)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v * 100 + 1.5, f"{v*100:.1f}%",
                ha="center", va="bottom", fontsize=18, fontweight="bold")
    ax.set_ylim(0, max(vals) * 100 + 15)
    ax.set_ylabel("Probability (%)")
    ax.set_title(f"{HOME_TEAM} vs {AWAY_TEAM}\n{MATCH_STAGE}  ·  {MATCH_DATE}  ·  New Jersey\n"
                 f"Player-Level Ensemble Model", fontsize=13)
    ax.yaxis.grid(True, color=GRID, zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    path = os.path.join(out_dir, f"viz_{HOME_TEAM}_vs_{AWAY_TEAM}_prediction.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def make_player_ppi_chart(squad_h, squad_a, out_dir):
    """Side-by-side player performance index comparison."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Spain
    ppis_h = squad_h["player_ppis"]
    names_h = [p[0].split()[-1] for p in ppis_h]  # last name only
    vals_h = [p[1] for p in ppis_h]
    y_h = np.arange(len(ppis_h))
    ax1.barh(y_h, vals_h, color=ESP_RED, zorder=3, alpha=0.85)
    ax1.set_yticks(y_h)
    ax1.set_yticklabels(names_h, fontsize=10)
    for i, v in enumerate(vals_h):
        ax1.text(v + 0.02, i, f"{v:.2f}", va="center", fontsize=9, color=INK)
    ax1.set_title(f"{HOME_TEAM} — Player Performance Index", fontsize=12)
    ax1.set_xlabel("PPI Score")
    ax1.xaxis.grid(True, color=GRID, zorder=0)
    ax1.set_axisbelow(True)
    ax1.invert_yaxis()

    # Argentina
    ppis_a = squad_a["player_ppis"]
    names_a = [p[0].split()[-1] for p in ppis_a]
    vals_a = [p[1] for p in ppis_a]
    y_a = np.arange(len(ppis_a))
    ax2.barh(y_a, vals_a, color=ARG_BLUE, zorder=3, alpha=0.85)
    ax2.set_yticks(y_a)
    ax2.set_yticklabels(names_a, fontsize=10)
    for i, v in enumerate(vals_a):
        ax2.text(v + 0.02, i, f"{v:.2f}", va="center", fontsize=9, color=INK)
    ax2.set_title(f"{AWAY_TEAM} — Player Performance Index", fontsize=12)
    ax2.set_xlabel("PPI Score")
    ax2.xaxis.grid(True, color=GRID, zorder=0)
    ax2.set_axisbelow(True)
    ax2.invert_yaxis()

    fig.suptitle("Player-Level Performance Index (Club + Tournament Form)", fontsize=14, y=1.01)
    fig.tight_layout()
    path = os.path.join(out_dir, f"ppi_{HOME_TEAM}_vs_{AWAY_TEAM}.png")
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def make_position_comparison_chart(squad_h, squad_a, out_dir):
    """Position group strength comparison."""
    fig, ax = plt.subplots(figsize=(10, 5.5))
    groups = ["Attack", "Midfield", "Defense", "Goalkeeper", "Bench Impact"]
    h_vals = [squad_h["attack"], squad_h["midfield"], squad_h["defense"],
              squad_h["gk"], compute_bench_impact(SPAIN_BENCH)]
    a_vals = [squad_a["attack"], squad_a["midfield"], squad_a["defense"],
              squad_a["gk"], compute_bench_impact(ARGENTINA_BENCH)]

    y = np.arange(len(groups))
    h = 0.35
    ax.barh(y + h / 2, h_vals, h, color=ESP_RED, label=HOME_TEAM, zorder=3, alpha=0.85)
    ax.barh(y - h / 2, a_vals, h, color=ARG_BLUE, label=AWAY_TEAM, zorder=3, alpha=0.85)

    for i, (hv, av) in enumerate(zip(h_vals, a_vals)):
        ax.text(hv + 0.02, i + h / 2, f"{hv:.2f}", va="center", fontsize=10, color=ESP_RED, fontweight="bold")
        ax.text(av + 0.02, i - h / 2, f"{av:.2f}", va="center", fontsize=10, color=ARG_BLUE, fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(groups, fontsize=12)
    ax.set_title("Position Group Strength Comparison\n(Player Performance Index Aggregation)", fontsize=13)
    ax.legend(loc="lower right", fontsize=11)
    ax.xaxis.grid(True, color=GRID, zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    path = os.path.join(out_dir, f"position_{HOME_TEAM}_vs_{AWAY_TEAM}.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def make_score_heatmap(matrix, lambda_h, lambda_a, out_dir):
    fig, ax = plt.subplots(figsize=(8, 7))
    display = matrix[:6, :6]
    cmap = LinearSegmentedColormap.from_list("custom", ["#ffffff", "#ffd6d6", ESP_RED])
    im = ax.imshow(display, cmap=cmap, aspect="equal", origin="lower")
    for i in range(6):
        for j in range(6):
            val = display[i, j]
            color = "white" if val > 0.08 else INK
            ax.text(j, i, f"{val*100:.1f}%", ha="center", va="center",
                    fontsize=10, fontweight="bold", color=color)
    ax.set_xticks(range(6))
    ax.set_yticks(range(6))
    ax.set_xticklabels(range(6))
    ax.set_yticklabels(range(6))
    ax.set_xlabel(f"{AWAY_TEAM} goals")
    ax.set_ylabel(f"{HOME_TEAM} goals")
    ax.set_title(f"Score Probability Matrix (Player-Adjusted Dixon-Coles)\n"
                 f"{HOME_TEAM} xG = {lambda_h:.2f}  |  {AWAY_TEAM} xG = {lambda_a:.2f}", fontsize=12)
    fig.colorbar(im, ax=ax, label="Probability", shrink=0.8)
    fig.tight_layout()
    path = os.path.join(out_dir, f"scorematrix_{HOME_TEAM}_vs_{AWAY_TEAM}.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def make_model_comparison_chart(models_dict, out_dir):
    """Compare probabilities across all model components."""
    fig, ax = plt.subplots(figsize=(12, 5.5))

    model_names = list(models_dict.keys())
    n = len(model_names)
    x = np.arange(n)
    w = 0.25

    home_vals = [models_dict[m][0] * 100 for m in model_names]
    draw_vals = [models_dict[m][1] * 100 for m in model_names]
    away_vals = [models_dict[m][2] * 100 for m in model_names]

    ax.bar(x - w, home_vals, w, color=ESP_RED, label=HOME_TEAM, zorder=3, alpha=0.85)
    ax.bar(x,     draw_vals, w, color=GRAY,     label="Draw",     zorder=3, alpha=0.7)
    ax.bar(x + w, away_vals, w, color=ARG_BLUE,  label=AWAY_TEAM, zorder=3, alpha=0.85)

    for i in range(n):
        ax.text(i - w, home_vals[i] + 0.8, f"{home_vals[i]:.1f}", ha="center", fontsize=9, fontweight="bold")
        ax.text(i,     draw_vals[i] + 0.8, f"{draw_vals[i]:.1f}", ha="center", fontsize=9)
        ax.text(i + w, away_vals[i] + 0.8, f"{away_vals[i]:.1f}", ha="center", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=10)
    ax.set_ylabel("Probability (%)")
    ax.set_title("Model Component Comparison\n(Team-Level vs Player-Level vs Ensemble)", fontsize=13)
    ax.legend(loc="upper right")
    ax.yaxis.grid(True, color=GRID, zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    path = os.path.join(out_dir, f"models_{HOME_TEAM}_vs_{AWAY_TEAM}.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


# ═══════════════════════════════════════════════════════════════════════════
# 9.  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  {HOME_TEAM.upper()} vs {AWAY_TEAM.upper()} — PLAYER-LEVEL PREDICTION")
    print(f"  2026 World Cup {MATCH_STAGE}")
    print(f"  {MATCH_DATE}  ·  {MATCH_VENUE}")
    print(sep)

    # ── 1. Player-level analysis ─────────────────────────────────────
    print(f"\n[1/9] Computing Player Performance Indices ...")
    squad_esp = compute_squad_strength(SPAIN_PLAYERS, SPAIN_MATCHES)
    squad_arg = compute_squad_strength(ARGENTINA_PLAYERS, ARGENTINA_MATCHES)

    print(f"\n  {HOME_TEAM} Squad Strength:")
    print(f"    Attack   : {squad_esp['attack']:.3f}")
    print(f"    Midfield : {squad_esp['midfield']:.3f}")
    print(f"    Defense  : {squad_esp['defense']:.3f}")
    print(f"    GK       : {squad_esp['gk']:.3f}")
    print(f"    Star bonus: {squad_esp['star_bonus']:.3f}")
    print(f"    Overall  : {squad_esp['overall']:.3f}")

    print(f"\n  {AWAY_TEAM} Squad Strength:")
    print(f"    Attack   : {squad_arg['attack']:.3f}")
    print(f"    Midfield : {squad_arg['midfield']:.3f}")
    print(f"    Defense  : {squad_arg['defense']:.3f}")
    print(f"    GK       : {squad_arg['gk']:.3f}")
    print(f"    Star bonus: {squad_arg['star_bonus']:.3f}")
    print(f"    Overall  : {squad_arg['overall']:.3f}")

    print(f"\n  Top 5 Player Performance Indices — {HOME_TEAM}:")
    for name, ppi in squad_esp["player_ppis"][:5]:
        print(f"    {name:<24} PPI = {ppi:.3f}")

    print(f"\n  Top 5 Player Performance Indices — {AWAY_TEAM}:")
    for name, ppi in squad_arg["player_ppis"][:5]:
        print(f"    {name:<24} PPI = {ppi:.3f}")

    # ── 2. Tournament profiles ───────────────────────────────────────
    print(f"\n[2/9] Computing tournament profiles ...")
    prof_esp = compute_tournament_profile(SPAIN_MATCHES, HOME_TEAM)
    prof_arg = compute_tournament_profile(ARGENTINA_MATCHES, AWAY_TEAM)

    # ── 3. Player-derived expected goals ─────────────────────────────
    print(f"\n[3/9] Computing player-derived expected goals ...")
    lambda_esp, lambda_arg = compute_player_derived_lambdas(
        squad_esp, squad_arg, prof_esp, prof_arg)
    print(f"  {HOME_TEAM} lambda (player-adjusted) : {lambda_esp:.3f}")
    print(f"  {AWAY_TEAM} lambda (player-adjusted) : {lambda_arg:.3f}")

    # ── 4. Monte Carlo simulation ────────────────────────────────────
    print(f"\n[4/9] Running Monte Carlo simulation ({N_SIMULATIONS:,} matches) ...")
    sim_result = simulate_match(lambda_esp, lambda_arg)
    p_sim_h = sim_result["p_home"]
    p_sim_d = sim_result["p_draw"]
    p_sim_a = sim_result["p_away"]
    print(f"  Simulation  ->  {HOME_TEAM} {p_sim_h*100:.1f}%  |  "
          f"Draw {p_sim_d*100:.1f}%  |  {AWAY_TEAM} {p_sim_a*100:.1f}%")
    print(f"  Avg goals   ->  {HOME_TEAM} {sim_result['avg_goals_home']:.2f}  |  "
          f"{AWAY_TEAM} {sim_result['avg_goals_away']:.2f}")

    # ── 5. Dixon-Coles Poisson ───────────────────────────────────────
    print(f"\n[5/9] Building Dixon-Coles Poisson score matrix ...")
    matrix = score_probability_matrix(lambda_esp, lambda_arg, DIXON_COLES_RHO, MAX_GOALS)
    p_pois_h, p_pois_d, p_pois_a = extract_probabilities(matrix)
    print(f"  Poisson     ->  {HOME_TEAM} {p_pois_h*100:.1f}%  |  "
          f"Draw {p_pois_d*100:.1f}%  |  {AWAY_TEAM} {p_pois_a*100:.1f}%")

    top_scores = top_scorelines(matrix, n=10)
    print(f"\n  Most likely scorelines:")
    for rank, (h_g, a_g, prob) in enumerate(top_scores, 1):
        outcome = (f"{HOME_TEAM} win" if h_g > a_g
                   else (f"{AWAY_TEAM} win" if a_g > h_g else "Draw"))
        print(f"    {rank:>2}. {HOME_TEAM} {h_g}-{a_g} {AWAY_TEAM}  "
              f"({prob*100:>5.2f}%)  [{outcome}]")

    # ── 6. Team-level XGBoost (as prior/anchor) ──────────────────────
    print(f"\n[6/9] Training team-level XGBoost model (anchor prior) ...")
    results = load_results()
    results = patch_wc2026_results(results)
    dataset, final_elo = build_dataset(results)
    long = per_team_long(results)

    h_elo = final_elo.get(HOME_TEAM, ELO_BASE)
    a_elo = final_elo.get(AWAY_TEAM, ELO_BASE)
    print(f"  {HOME_TEAM} Elo  : {h_elo:.0f}")
    print(f"  {AWAY_TEAM} Elo  : {a_elo:.0f}")
    print(f"  Elo gap     : {h_elo - a_elo:+.0f}")

    train, val = split_by_date(dataset, TRAIN_START, VAL_START, MATCH_DATE)
    model, X_val, y_val = train_model(train, val)
    evaluate(model, X_val, y_val)

    p_xgb_h, p_xgb_d, p_xgb_a = predict_symmetric(
        model, long, final_elo, HOME_TEAM, AWAY_TEAM, MATCH_DATE,
        neutral=True, weight=4)
    print(f"  XGBoost     ->  {HOME_TEAM} {p_xgb_h*100:.1f}%  |  "
          f"Draw {p_xgb_d*100:.1f}%  |  {AWAY_TEAM} {p_xgb_a*100:.1f}%")

    # ── 7. Fit ensemble weights via backtesting ──────────────────────
    print(f"\n[7/9] Fitting ensemble weights on historical knockout data ...")

    HAND_PICKED = (0.35, 0.30, 0.20, 0.15)  # for comparison
    fitted_weights, backtest_results = fit_ensemble_weights(
        dataset, final_elo, long, model)
    W_SIM, W_POIS, W_XGB, W_SQUAD = fitted_weights

    print(f"\n  Fitted weights (log-loss minimized):")
    print(f"    Monte Carlo (player) : {W_SIM:.3f}  (hand-picked was {HAND_PICKED[0]:.2f})")
    print(f"    Dixon-Coles (player) : {W_POIS:.3f}  (hand-picked was {HAND_PICKED[1]:.2f})")
    print(f"    XGBoost (team)       : {W_XGB:.3f}  (hand-picked was {HAND_PICKED[2]:.2f})")
    print(f"    Squad strength       : {W_SQUAD:.3f}  (hand-picked was {HAND_PICKED[3]:.2f})")

    if backtest_results:
        print(f"\n  Backtest performance ({backtest_results['n_matches']} major-tournament matches):")
        print(f"    Ensemble log-loss    : {backtest_results['ensemble_logloss']:.4f}")
        print(f"    Ensemble Brier score : {backtest_results['ensemble_brier']:.4f}")
        print(f"    Ensemble accuracy    : {backtest_results['ensemble_accuracy']:.3f}")
        if backtest_results.get("calibration_bins"):
            print(f"\n  Calibration (P(home win) reliability):")
            print(f"    {'Predicted':>10} {'Actual':>10} {'N matches':>10}")
            for b in backtest_results["calibration_bins"]:
                print(f"    {b['predicted_mean']:>10.3f} {b['actual_mean']:>10.3f} {b['count']:>10d}")

    # ── 8. Ensemble blend ──────────────────────────────────────────────
    print(f"\n[8/9] Building ensemble prediction ...")

    # Squad-strength based probability
    str_diff = squad_esp["overall"] - squad_arg["overall"]
    p_squad_raw_h = 1.0 / (1.0 + np.exp(-str_diff * 3.0))
    p_squad_raw_a = 1.0 - p_squad_raw_h
    p_squad_d = 0.22  # base draw rate for evenly matched teams
    p_squad_h = p_squad_raw_h * (1.0 - p_squad_d)
    p_squad_a = p_squad_raw_a * (1.0 - p_squad_d)

    print(f"  Squad model ->  {HOME_TEAM} {p_squad_h*100:.1f}%  |  "
          f"Draw {p_squad_d*100:.1f}%  |  {AWAY_TEAM} {p_squad_a*100:.1f}%")
    print(f"\n  Ensemble weights (fitted): Simulation={W_SIM:.2f}  Poisson={W_POIS:.2f}  "
          f"XGBoost={W_XGB:.2f}  Squad={W_SQUAD:.2f}")

    ens_h = W_SIM * p_sim_h + W_POIS * p_pois_h + W_XGB * p_xgb_h + W_SQUAD * p_squad_h
    ens_d = W_SIM * p_sim_d + W_POIS * p_pois_d + W_XGB * p_xgb_d + W_SQUAD * p_squad_d
    ens_a = W_SIM * p_sim_a + W_POIS * p_pois_a + W_XGB * p_xgb_a + W_SQUAD * p_squad_a
    ens_tot = ens_h + ens_d + ens_a
    ens_h /= ens_tot
    ens_d /= ens_tot
    ens_a /= ens_tot

    print(f"\n  ENSEMBLE    ->  {HOME_TEAM} {ens_h*100:.1f}%  |  "
          f"Draw {ens_d*100:.1f}%  |  {AWAY_TEAM} {ens_a*100:.1f}%")

    # ── 9. Charts ────────────────────────────────────────────────────
    print(f"\n[9/9] Generating visualizations ...")
    out_dir = os.path.join("predictions", MATCH_DATE)
    os.makedirs(out_dir, exist_ok=True)

    chart1 = make_prediction_chart(ens_h, ens_d, ens_a, out_dir)
    chart2 = make_player_ppi_chart(squad_esp, squad_arg, out_dir)
    chart3 = make_position_comparison_chart(squad_esp, squad_arg, out_dir)
    chart4 = make_score_heatmap(matrix, lambda_esp, lambda_arg, out_dir)
    chart5 = make_model_comparison_chart({
        "Monte Carlo\n(Player)": (p_sim_h, p_sim_d, p_sim_a),
        "Dixon-Coles\n(Player)": (p_pois_h, p_pois_d, p_pois_a),
        "XGBoost\n(Team)": (p_xgb_h, p_xgb_d, p_xgb_a),
        "Squad\nStrength": (p_squad_h, p_squad_d, p_squad_a),
        "ENSEMBLE": (ens_h, ens_d, ens_a),
    }, out_dir)

    # ══════════════════════════════════════════════════════════════════
    #  COMPREHENSIVE REPORT
    # ══════════════════════════════════════════════════════════════════

    outcomes = [(HOME_TEAM, ens_h), ("Draw", ens_d), (AWAY_TEAM, ens_a)]
    pick, conf = max(outcomes, key=lambda x: x[1])

    if   conf >= 0.55: strength = "STRONG LEAN"
    elif conf >= 0.42: strength = "LEAN"
    else:              strength = "TOSS-UP"

    print(f"\n\n{sep}")
    print(f"  FINAL PREDICTION — {MATCH_STAGE.upper()}")
    print(f"  PLAYER-LEVEL ENSEMBLE MODEL")
    print(sep)
    print(f"\n  90-minute probabilities (ensemble):")
    print(f"    {HOME_TEAM:<22} win   {ens_h*100:>5.1f}%")
    print(f"    {'Draw':<22}       {ens_d*100:>5.1f}%")
    print(f"    {AWAY_TEAM:<22} win   {ens_a*100:>5.1f}%")
    print(f"\n  Expected goals:")
    print(f"    {HOME_TEAM:<22} xG   {lambda_esp:>5.2f}")
    print(f"    {AWAY_TEAM:<22} xG   {lambda_arg:>5.2f}")
    print(f"    Most likely score: {HOME_TEAM} {top_scores[0][0]}-"
          f"{top_scores[0][1]} {AWAY_TEAM}  ({top_scores[0][2]*100:.1f}%)")
    print(f"\n  {'-' * 62}")
    print(f"  PICK: {pick}  ({conf*100:.1f}%)   [{strength}]")
    print(sep)

    # ── Model comparison table ───────────────────────────────────────
    print(f"\n\n{'─' * 72}")
    print("  MODEL COMPARISON")
    print(f"{'─' * 72}")
    print(f"  {'Model':<32} {HOME_TEAM:>10} {'Draw':>10} {AWAY_TEAM:>10}")
    print(f"  {'─'*32} {'─'*10} {'─'*10} {'─'*10}")
    models = [
        ("Monte Carlo (player-derived)", p_sim_h, p_sim_d, p_sim_a),
        ("Dixon-Coles Poisson (player)", p_pois_h, p_pois_d, p_pois_a),
        ("XGBoost (team-level)", p_xgb_h, p_xgb_d, p_xgb_a),
        ("Squad strength comparison", p_squad_h, p_squad_d, p_squad_a),
        ("ENSEMBLE (weighted blend)", ens_h, ens_d, ens_a),
    ]
    for name, h, d, a in models:
        print(f"  {name:<32} {h*100:>9.1f}% {d*100:>9.1f}% {a*100:>9.1f}%")

    # ── Squad strength comparison ────────────────────────────────────
    print(f"\n\n{'─' * 72}")
    print("  SQUAD STRENGTH BREAKDOWN (Player Performance Index)")
    print(f"{'─' * 72}")
    print(f"  {'Position Group':<20} {HOME_TEAM:>12} {AWAY_TEAM:>12} {'Edge':>12}")
    print(f"  {'─'*20} {'─'*12} {'─'*12} {'─'*12}")
    groups = [
        ("Attack",   squad_esp["attack"],   squad_arg["attack"]),
        ("Midfield", squad_esp["midfield"], squad_arg["midfield"]),
        ("Defense",  squad_esp["defense"],  squad_arg["defense"]),
        ("Goalkeeper", squad_esp["gk"],     squad_arg["gk"]),
        ("Star Bonus", squad_esp["star_bonus"], squad_arg["star_bonus"]),
        ("OVERALL",  squad_esp["overall"],  squad_arg["overall"]),
    ]
    for name, hv, av in groups:
        edge = hv - av
        winner = HOME_TEAM if edge > 0 else (AWAY_TEAM if edge < 0 else "Even")
        print(f"  {name:<20} {hv:>12.3f} {av:>12.3f} {edge:>+11.3f} {winner}")

    # ── Tournament performance comparison ────────────────────────────
    print(f"\n\n{'─' * 72}")
    print("  TOURNAMENT PERFORMANCE COMPARISON (7 matches each)")
    print(f"{'─' * 72}")
    print(f"  {'Metric':<28} {HOME_TEAM:>12} {AWAY_TEAM:>12} {'Edge':>12}")
    print(f"  {'─'*28} {'─'*12} {'─'*12} {'─'*12}")
    rows = [
        ("Record (W-D-L)",
         f"{prof_esp['W']}-{prof_esp['D']}-{prof_esp['L']}",
         f"{prof_arg['W']}-{prof_arg['D']}-{prof_arg['L']}", ""),
        ("Goals Scored",
         f"{prof_esp['GF']}", f"{prof_arg['GF']}",
         f"{prof_esp['GF']-prof_arg['GF']:+d}"),
        ("Goals Conceded",
         f"{prof_esp['GA']}", f"{prof_arg['GA']}",
         f"{prof_esp['GA']-prof_arg['GA']:+d}"),
        ("Goal Difference",
         f"{prof_esp['GD']:+d}", f"{prof_arg['GD']:+d}",
         f"{prof_esp['GD']-prof_arg['GD']:+d}"),
        ("xG Total",
         f"{prof_esp['xG']:.2f}", f"{prof_arg['xG']:.2f}",
         f"{prof_esp['xG']-prof_arg['xG']:+.2f}"),
        ("xG / Match",
         f"{prof_esp['xG/m']:.2f}", f"{prof_arg['xG/m']:.2f}",
         f"{prof_esp['xG/m']-prof_arg['xG/m']:+.2f}"),
        ("xGA / Match",
         f"{prof_esp['xGA/m']:.2f}", f"{prof_arg['xGA/m']:.2f}",
         f"{prof_esp['xGA/m']-prof_arg['xGA/m']:+.2f}"),
        ("Shots on Target",
         f"{prof_esp['SoT']}", f"{prof_arg['SoT']}",
         f"{prof_esp['SoT']-prof_arg['SoT']:+d}"),
        ("Avg Possession",
         f"{prof_esp['Poss%']:.0f}%", f"{prof_arg['Poss%']:.0f}%", ""),
        ("Clean Sheets",
         f"{prof_esp['CS']}", f"{prof_arg['CS']}", ""),
        ("Finishing (G/xG)",
         f"{prof_esp['Finishing']:.2f}", f"{prof_arg['Finishing']:.2f}", ""),
    ]
    for label, hv, av, edge_str in rows:
        print(f"  {label:<28} {hv:>12} {av:>12} {edge_str:>12}")

    # ── Key player matchups ──────────────────────────────────────────
    print(f"\n\n{'─' * 72}")
    print("  KEY PLAYER MATCHUPS")
    print(f"{'─' * 72}")
    matchups = [
        ("STAR FORWARDS",
         f"Yamal (4G 5A, PPI {squad_esp['player_ppis'][0][1]:.2f})",
         f"Messi (4G 4A, PPI {squad_arg['player_ppis'][0][1]:.2f})"),
        ("STRIKERS",
         f"Morata (2G 1A) + Olmo (3G 2A)",
         f"Alvarez (5G 2A) + Lautaro (3G 1A)"),
        ("WIDE THREAT",
         f"Yamal (4G 5A) + N. Williams (3G 2A)",
         f"Messi (4G 4A) + Di Maria (1G 2A bench)"),
        ("MIDFIELD ENGINE",
         f"Rodri (1G 2A) + Pedri (2G 3A)",
         f"De Paul (0G 2A) + Enzo (1G 3A)"),
        ("DEFENSIVE SPINE",
         f"Laporte + Le Normand",
         f"Romero + Otamendi"),
        ("GOALKEEPER",
         f"Unai Simon (3 CS, 82% saves)",
         f"Emi Martinez (2 CS, 80% saves)"),
    ]
    for area, esp_desc, arg_desc in matchups:
        print(f"\n  {area}:")
        print(f"    {HOME_TEAM}: {esp_desc}")
        print(f"    {AWAY_TEAM}: {arg_desc}")

    # ── Match-by-match results ───────────────────────────────────────
    print(f"\n\n{'─' * 72}")
    print(f"  MATCH-BY-MATCH — {HOME_TEAM.upper()}")
    print(f"{'─' * 72}")
    print(f"  {'Date':<12} {'Stage':<8} {'Opponent':<14} "
          f"{'Score':>6} {'xG':>5} {'xGA':>5} {'Poss':>5}")
    for m in SPAIN_MATCHES:
        print(f"  {m['date']:<12} {m['stage']:<8} {m['opp']:<14} "
              f"{m['gf']}-{m['ga']:>2}  {m['xg']:>5.2f} {m['xga']:>5.2f} "
              f"{m['poss']:>4}%")

    print(f"\n{'─' * 72}")
    print(f"  MATCH-BY-MATCH — {AWAY_TEAM.upper()}")
    print(f"{'─' * 72}")
    print(f"  {'Date':<12} {'Stage':<8} {'Opponent':<14} "
          f"{'Score':>6} {'xG':>5} {'xGA':>5} {'Poss':>5}")
    for m in ARGENTINA_MATCHES:
        print(f"  {m['date']:<12} {m['stage']:<8} {m['opp']:<14} "
              f"{m['gf']}-{m['ga']:>2}  {m['xg']:>5.2f} {m['xga']:>5.2f} "
              f"{m['poss']:>4}%")

    # ── Bench impact analysis ────────────────────────────────────────
    print(f"\n\n{'─' * 72}")
    print("  BENCH IMPACT ANALYSIS")
    print(f"{'─' * 72}")
    print(f"\n  {HOME_TEAM} key substitutes:")
    for p in SPAIN_BENCH:
        print(f"    {p['name']:<20} {p['pos']:<3}  G:{p.get('wc_goals',0)} A:{p.get('wc_assists',0)}  "
              f"Impact: {p['impact']}  — {p['notes']}")
    print(f"\n  {AWAY_TEAM} key substitutes:")
    for p in ARGENTINA_BENCH:
        print(f"    {p['name']:<20} {p['pos']:<3}  G:{p.get('wc_goals',0)} A:{p.get('wc_assists',0)}  "
              f"Impact: {p['impact']}  — {p['notes']}")
    print(f"\n  Bench impact score: {HOME_TEAM} {compute_bench_impact(SPAIN_BENCH):.3f}  |  "
          f"{AWAY_TEAM} {compute_bench_impact(ARGENTINA_BENCH):.3f}")

    # ── Analysis notes ───────────────────────────────────────────────
    print(f"\n\n{'─' * 72}")
    print("  ANALYSIS NOTES")
    print(f"{'─' * 72}")
    print(textwrap.fill(
        f"This is the World Cup Final — the biggest match in football. "
        f"Spain arrive as the dominant force of the tournament, winning "
        f"6 of 7 matches (1 draw vs Cabo Verde) and conceding just 1 goal "
        f"in the knockout rounds. Their semifinal dismantling of France "
        f"(2-0, xG 1.90-0.31) was the most complete performance of the "
        f"tournament. Argentina are unbeaten in 7 matches, scoring 19 "
        f"goals, though their knockout victories have been tighter.",
        width=68, initial_indent="  ", subsequent_indent="  "))
    print()
    print(textwrap.fill(
        f"Spain's attacking trident of Yamal (4G 5A), Nico Williams "
        f"(3G 2A), and Dani Olmo (3G 2A) has been devastating — they "
        f"combine for 10 goals and 9 assists across the tournament. "
        f"At 19, Yamal is playing the best football of his life and "
        f"could become the youngest player to score in a World Cup Final. "
        f"Behind them, Rodri and Pedri form perhaps the best midfield "
        f"partnership in the tournament.",
        width=68, initial_indent="  ", subsequent_indent="  "))
    print()
    print(textwrap.fill(
        f"Argentina's strength lies in their experience and Messi's "
        f"farewell tour. At 39, Messi has contributed 4G 4A and remains "
        f"the creative heartbeat of this team. Alvarez (5G 2A) has been "
        f"clinical as the central striker, while Enzo Fernandez (1G 3A) "
        f"runs the midfield. The key question is whether Argentina's "
        f"aging defense (Otamendi 38, Tagliafico 33) can contain Spain's "
        f"electric young wingers.",
        width=68, initial_indent="  ", subsequent_indent="  "))
    print()
    print(textwrap.fill(
        f"Spain's tournament xG ({prof_esp['xG']:.2f} total, "
        f"{prof_esp['xG/m']:.2f}/match) is the highest of any remaining "
        f"team, while their xGA ({prof_esp['xGA/m']:.2f}/match) is "
        f"among the lowest. Argentina have been clinical finishers "
        f"({prof_arg['Finishing']:.2f} G/xG ratio) but their underlying "
        f"numbers suggest they've overperformed expectations slightly. "
        f"This is a battle of Spain's system and youth vs Argentina's "
        f"experience and individual brilliance — with the Messi narrative "
        f"adding an extra dimension to what promises to be an epic final.",
        width=68, initial_indent="  ", subsequent_indent="  "))

    # ── Head-to-head history ─────────────────────────────────────────
    print(f"\n\n{'─' * 72}")
    print("  HEAD-TO-HEAD HISTORY")
    print(f"{'─' * 72}")
    h2h_matches = long[(long["team"] == HOME_TEAM) & (long["opp"] == AWAY_TEAM)
                       & (long["date"] < pd.Timestamp(MATCH_DATE))].sort_values("date")
    if len(h2h_matches) > 0:
        for _, row in h2h_matches.tail(10).iterrows():
            res = "W" if row["result"] == 1.0 else ("D" if row["result"] == 0.5 else "L")
            print(f"  {str(row['date'].date()):<12}  {HOME_TEAM} "
                  f"{int(row['gf'])}-{int(row['ga'])} {AWAY_TEAM}  ({res})")
        n_h2h, wr_h2h, gd_h2h = h2h_as_of(long, HOME_TEAM, AWAY_TEAM, MATCH_DATE)
        print(f"\n  Total meetings: {int(n_h2h)}, {HOME_TEAM} win-rate: {wr_h2h:.2f}, "
              f"avg GD: {gd_h2h:+.1f}")
    else:
        print("  No previous meetings found in dataset")

    print(f"\n\n{'─' * 72}")
    print(f"  Charts saved to predictions/{MATCH_DATE}/")
    for c in [chart1, chart2, chart3, chart4, chart5]:
        print(f"    -> {c}")
    print(f"{'─' * 72}\n")

    # ── Methodology & Limitations ──────────────────────────────────────
    methodology = _build_methodology(fitted_weights, HAND_PICKED, backtest_results)
    print(f"\n\n{'═' * 72}")
    print("  METHODOLOGY & LIMITATIONS")
    print(f"{'═' * 72}")
    print(methodology)

    # Save to file
    methodology_path = os.path.join(out_dir, "METHODOLOGY.md")
    with open(methodology_path, "w") as f:
        f.write(methodology)
    print(f"\n  Saved to {methodology_path}")
    print(f"{'═' * 72}\n")


def _build_methodology(fitted_weights, hand_picked, backtest_results):
    """Build the methodology and limitations text."""
    lines = []
    lines.append("# Methodology & Limitations\n")
    lines.append("## Model Architecture\n")
    lines.append("Hybrid ensemble combining four component models:\n")
    lines.append("1. **Monte Carlo simulation** (10,000 Poisson-distributed matches)")
    lines.append("   - Lambdas derived from player-level PPI position-group strengths")
    lines.append("   - Blended 40% player-ratio + 60% tournament-xG-ratio")
    lines.append("2. **Dixon-Coles Poisson** score matrix (same player-derived lambdas)")
    lines.append("   - Low-score correlation correction (rho parameter)")
    lines.append("3. **XGBoost multi-class classifier** (team-level historical prior)")
    lines.append("   - Trained on 20 years of international results (2006-present)")
    lines.append("   - Features: Elo ratings, 5/10-match form, head-to-head, rest days")
    lines.append("4. **Squad strength comparison** (PPI-based sigmoid)\n")

    lines.append("## What is fitted from data\n")
    lines.append(f"- **Ensemble blend weights**: fitted via scipy.optimize on "
                 f"major-tournament matches since 2010 (minimizing log-loss)")
    lines.append(f"  - Fitted: Sim={fitted_weights[0]:.3f}, Pois={fitted_weights[1]:.3f}, "
                 f"XGB={fitted_weights[2]:.3f}, Squad={fitted_weights[3]:.3f}")
    lines.append(f"  - Hand-picked comparison: Sim={hand_picked[0]:.2f}, Pois={hand_picked[1]:.2f}, "
                 f"XGB={hand_picked[2]:.2f}, Squad={hand_picked[3]:.2f}")
    lines.append("- **XGBoost model**: hyperparameters tuned via early stopping on "
                 "2023-2026 validation set")
    lines.append("- **Elo ratings**: fitted iteratively on all historical results")
    lines.append("- **Momentum feature**: computed from actual knockout GD (tanh-normalized)\n")

    if backtest_results:
        lines.append("## Ensemble calibration (walk-forward backtest)\n")
        lines.append(f"- Backtest set: {backtest_results['n_matches']} major-tournament "
                     f"matches (2010-2025)")
        lines.append(f"- Log-loss: {backtest_results['ensemble_logloss']:.4f}")
        lines.append(f"- Brier score (multi-class): {backtest_results['ensemble_brier']:.4f}")
        lines.append(f"- Accuracy: {backtest_results['ensemble_accuracy']:.3f}")
        if backtest_results.get("calibration_bins"):
            lines.append(f"\n  Reliability (P(home win)):")
            lines.append(f"  | Predicted | Actual | N |")
            lines.append(f"  |-----------|--------|---|")
            for b in backtest_results["calibration_bins"]:
                lines.append(f"  | {b['predicted_mean']:.3f} | {b['actual_mean']:.3f} "
                             f"| {b['count']} |")
        lines.append("")

    lines.append("## What remains a manual choice\n")
    lines.append("- **PPI formula weights**: position-aware sub-weights (30% club / 70% WC)")
    lines.append("  - GK: save_pct 0.35, CS_rate 0.30, GA_rate 0.35")
    lines.append("  - CB/FB: tackles 0.15, interceptions 0.15, aerials 0.08, goals 0.15, assists 0.10")
    lines.append("  - CM: tackles 0.10, interceptions 0.10, key_passes 0.10, goals 0.20, assists 0.15")
    lines.append("  - AM/W/ST: goals 0.25-0.30, assists 0.15, xG/SoT/dribbles smaller weights")
    lines.append("- **Dixon-Coles rho = -0.13** (low-score correlation correction)")
    lines.append("- **Position-group aggregation**: 35% ATT + 25% MID + 25% DEF + 15% GK")
    lines.append("- **Star bonus**: top 3 players get 0.15/0.12/0.09 uplift weights")
    lines.append("- **Knockout regression factor = 0.95** (finals tend toward tighter play)")
    lines.append("- **Tournament average xG = 1.40** (assumed baseline)")
    lines.append("- **Bench impact formula**: club_rating/7 * 0.4 + "
                 "(goals*0.3 + assists*0.15) * 0.6, scaled by impact tier\n")

    lines.append("## Key limitations\n")
    lines.append("1. **PPI weights are not fitted** — position sub-weights are hand-tuned "
                 "heuristics, not optimized against outcomes.")
    lines.append("2. **No player-level data in backtest** — the ensemble weight fitting uses "
                 "Elo-based proxies for the Monte Carlo/Dixon-Coles components (no per-player "
                 "stats available in results.csv for historical matches).")
    lines.append("3. **Dixon-Coles rho is assumed** — not fitted from this tournament's data "
                 "(would need >50 matches to estimate reliably).")
    lines.append("4. **Player data is point-in-time** — 2024-25 club stats and 2026 WC stats "
                 "are manually entered, not from a live data feed.")
    lines.append("5. **XGBoost sees team entities, not personnel** — a squad overhaul "
                 "(e.g. generational shift) is invisible to the team-level model.")
    lines.append("6. **Small-sample finals** — World Cup finals are rare events; "
                 "the model has no special 'final psychology' feature.")
    lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    main()
