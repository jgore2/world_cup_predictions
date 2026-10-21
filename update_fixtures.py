"""
update_fixtures.py — refreshes data_cache/fixtures.csv with resolved knockout matchups
========================================================================================

The original fixtures.csv only has real team names for the 72 group-stage matches.
Knockout rounds start out as placeholders ("Winner Group L v Runner-up Group K") and
only get real team names once FIFA confirms the bracket slot. This script scrapes the
Wikipedia "2026 FIFA World Cup knockout stage" article — which editors update within
minutes of each match being confirmed/played — and rewrites any placeholder row whose
match has since been resolved, using the real team names, plus fills in the score if
the match has already been played.

Usage:
    python update_fixtures.py                  # updates data_cache/fixtures.csv in place
    python update_fixtures.py --dry-run         # shows what would change, writes nothing
    python update_fixtures.py --source URL      # override the Wikipedia URL if it moves

Run this before predict_today.py any time you want to predict a knockout match — it's
what keeps "England" vs "Norway" (or any other still-to-be-confirmed pairing) resolvable
instead of throwing "Couldn't find a World Cup match".
"""

import argparse
import csv
import os
import re
import sys
from datetime import datetime


import requests
from bs4 import BeautifulSoup

WIKI_URL = "https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_knockout_stage"
FIXTURES_PATH = os.path.join("data_cache", "fixtures.csv")
FIELDNAMES = ["date", "match_number", "teams", "group", "stadium", "date_dt"]

HEADERS = {"User-Agent": "world-cup-fixtures-updater/1.0 (personal script)"}


def fetch_soup(url):
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser")


def parse_footballboxes(soup):
    """
    Wikipedia World Cup match articles render each fixture as a
    <table class="fevent"> (older skin) or a "footballbox" div/table with
    home team / score / away team cells. This walks every such block and
    pulls out: home team, away team, score (if played), date, stadium, round.
    Falls back gracefully (returns []) if Wikipedia's markup has changed —
    check the CSS selectors below against the live page source if so.
    """
    matches = []
    boxes = soup.select("table.fevent, div.footballbox, table.footballbox")
    current_round = None

    for el in soup.find_all(["h2", "h3", "table", "div"]):
        if el.name in ("h2", "h3"):
            heading = el.get_text(strip=True)
            if heading:
                current_round = heading
            continue
        if el.name == "table" and "fevent" in (el.get("class") or []):
            box = el
        elif "footballbox" in (el.get("class") or []):
            box = el
        else:
            continue

        home = box.select_one(".fhome, .footballbox .team.hteam, th.fhome")
        away = box.select_one(".faway, .footballbox .team.ateam, th.faway")
        score = box.select_one(".fscore, .footballbox .score")
        date_cell = box.select_one(".fdate, .footballbox .fdate")
        stadium_cell = box.select_one(".fright, .footballbox .stadium")

        if not (home and away):
            continue

        home_name = home.get_text(strip=True)
        away_name = away.get_text(strip=True)
        score_text = score.get_text(strip=True) if score else ""
        date_text = date_cell.get_text(strip=True) if date_cell else ""
        stadium_text = stadium_cell.get_text(strip=True) if stadium_cell else ""

        # Skip rows that are still placeholders (bracket not resolved yet)
        if any(tok in home_name for tok in ["Winner", "Runner-up", "3rd", "TBD"]) or \
           any(tok in away_name for tok in ["Winner", "Runner-up", "3rd", "TBD"]):
            continue

        matches.append({
            "round": current_round or "Knockout",
            "home": home_name,
            "away": away_name,
            "score": score_text,
            "date_text": date_text,
            "stadium": stadium_text,
        })

    return matches


def normalize_date(date_text, fallback_year=2026):
    """Best-effort parse of Wikipedia's date strings into YYYY-MM-DD."""
    date_text = re.sub(r"\(.*?\)", "", date_text).strip()
    for fmt in ("%d %B %Y", "%B %d, %Y", "%d %B", "%A, %d %B %Y"):
        try:
            dt = datetime.strptime(date_text, fmt)
            if dt.year == 1900:
                dt = dt.replace(year=fallback_year)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            continue
    return ""


def load_fixtures():
    if not os.path.exists(FIXTURES_PATH):
        raise FileNotFoundError(f"{FIXTURES_PATH} not found — run predict_today.py once first.")
    with open(FIXTURES_PATH, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def save_fixtures(rows):
    with open(FIXTURES_PATH, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def is_placeholder(teams_field):
    return any(tok in teams_field for tok in ["Winner", "Runner-up", "3rd", "TBD"])


def merge(rows, resolved_matches, dry_run=False):
    changed = 0
    # Resolved matches are matched to placeholder rows by row order within each
    # round's block of matches, since match numbers in fixtures.csv (Match 73, 74...)
    # correspond 1:1 with Wikipedia's chronological bracket listing per round.
    placeholder_rows = [r for r in rows if is_placeholder(r["teams"])]

    for i, resolved in enumerate(resolved_matches):
        if i >= len(placeholder_rows):
            break
        row = placeholder_rows[i]
        old_teams = row["teams"]
        new_teams = f"{resolved['home']} v {resolved['away']}"
        if old_teams == new_teams:
            continue

        changed += 1
        print(f"  [{row['match_number']}] '{old_teams}' -> '{new_teams}'"
              + (f"  ({resolved['score']})" if resolved["score"] else ""))

        if not dry_run:
            row["teams"] = new_teams
            row["group"] = resolved["round"]
            if resolved["stadium"]:
                row["stadium"] = resolved["stadium"]
            nd = normalize_date(resolved["date_text"])
            if nd:
                row["date_dt"] = nd
                row["date"] = nd

    return changed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=WIKI_URL)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print(f"Fetching {args.source} ...")
    soup = fetch_soup(args.source)
    resolved = parse_footballboxes(soup)
    print(f"Found {len(resolved)} resolved knockout matches on the page.")

    if not resolved:
        print("  No resolved matches parsed — Wikipedia's markup may have changed.")
        print("  Open the page, inspect a match box's HTML, and adjust the CSS")
        print("  selectors in parse_footballboxes().")
        return

    rows = load_fixtures()
    changed = merge(rows, resolved, dry_run=args.dry_run)

    if changed == 0:
        print("Nothing to update — fixtures.csv already matches the current bracket.")
        return

    if args.dry_run:
        print(f"\n{changed} row(s) would be updated. Re-run without --dry-run to apply.")
    else:
        save_fixtures(rows)
        print(f"\nUpdated {changed} row(s) in {FIXTURES_PATH}.")


if __name__ == "__main__":
    main()
