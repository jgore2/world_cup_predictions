"""
build_shootout_history.py — pulls each team's all-time World Cup penalty
shoot-out record into data_cache/penalty_shootouts.csv, for use with the
resolve_draw(shootout_rates=...) upgrade in predict_today.py.

Source: the "Penalty shoot-out statistics by team" table on Wikipedia's
"List of FIFA World Cup penalty shoot-outs" article. That table already
aggregates each team's full history (e.g. Germany's total includes West
Germany's shoot-outs, Serbia's includes Yugoslavia's) so no extra merging
is needed on your end.

Usage:
    pip install requests beautifulsoup4 pandas lxml
    python build_shootout_history.py
    python build_shootout_history.py --dry-run
"""

import argparse
import os

import pandas as pd
import requests

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_FIFA_World_Cup_penalty_shoot-outs"
CACHE_DIR = "data_cache"
OUT_PATH = os.path.join(CACHE_DIR, "penalty_shootouts.csv")
HEADERS = {"User-Agent": "world-cup-shootout-history/1.0 (personal script)"}

# reuse the same normalization predict_today.py applies to results.csv,
# so team names line up between the two files
NAME_MAP = {
    "USA": "United States", "Korea Republic": "South Korea", "South Korea": "South Korea",
    "Republic of Ireland": "Ireland", "Türkiye": "Turkey", "Turkey": "Turkey",
    "Cape Verde": "Cabo Verde", "Côte d'Ivoire": "Ivory Coast",
    "Czechia": "Czech Republic", "Curaçao": "Curacao",
    "Congo DR": "DR Congo", "Congo": "Republic of the Congo",
}


def fetch_table():
    resp = requests.get(WIKI_URL, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    tables = pd.read_html(resp.text)  # requires lxml
    for t in tables:
        cols = [str(c).strip() for c in t.columns]
        if "Team" in cols and "Played" in cols and "Win" in cols and "Loss" in cols:
            return t
    raise RuntimeError(
        "Couldn't find the 'Penalty shoot-out statistics by team' table — "
        "Wikipedia's page structure may have changed. Open the URL and check "
        "the table still has Team/Played/Win/Loss columns."
    )


def clean(df):
    df = df[["Team", "Played", "Win", "Loss"]].copy()
    df["Team"] = df["Team"].astype(str).str.strip()
    df = df[~df["Team"].str.contains("Total", case=False, na=False)]
    df["Team"] = df["Team"].map(lambda t: NAME_MAP.get(t, t))
    df["Played"] = pd.to_numeric(df["Played"], errors="coerce").fillna(0).astype(int)
    df["Win"] = pd.to_numeric(df["Win"], errors="coerce").fillna(0).astype(int)
    df["Loss"] = pd.to_numeric(df["Loss"], errors="coerce").fillna(0).astype(int)
    df = df.rename(columns={"Team": "team", "Played": "shootouts_played", "Win": "shootouts_won"})
    return df[["team", "shootouts_played", "shootouts_won"]].sort_values("team").reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print(f"Fetching {WIKI_URL} ...")
    raw = fetch_table()
    df = clean(raw)

    print(f"\nParsed {len(df)} teams with shoot-out history:")
    print(df.to_string(index=False))

    if args.dry_run:
        print(f"\nDry run — nothing written. Would save to {OUT_PATH}.")
        return

    os.makedirs(CACHE_DIR, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved -> {OUT_PATH}")


if __name__ == "__main__":
    main()