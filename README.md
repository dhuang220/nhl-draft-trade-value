# NHL Draft Value & Trade Grader

A data-science project that builds an original NHL player value model from public data, then uses it to grade real and hypothetical trades and evaluate draft picks.

## What this does

1. **Composite player value model** - learns what makes an NHL skater valuable (production, defensive play, possession/expected goals, age, draft pedigree) by regressing against an existing performance-value stat (Point Shares), then separately compares that on-ice value to salary to find bargains and overpays.
2. **Draft outcome model** - predicts a drafted player's expected career value from what's known on draft day (pick number, age, position), with an honest treatment of the fact that recent draft classes haven't had time to prove themselves yet.
3. **Trade grader** - combines both models to value either side of any NHL trade from the trade history dataset, restricted in the dashboard to the 2000-2026 seasons. The 2016-17 season uses a locally precomputed exact model output (highest confidence); every other season pulls that player's real stats for that specific year from the NHL API and runs them through the same trained model (medium confidence); a career-total fallback only kicks in when no season-specific record can be found. Also grades hypothetical trades built from today's rosters using live NHL data.

## Why these design choices (and where they came from)

This project deliberately documents dead ends and negative results, not just what worked - that's more representative of real applied data science.

- **Scraping was ruled out.** Hockey-Reference/Sports-Reference's terms of use explicitly prohibit automated scraping and building tools from scraped data. Every data source here is either a public Kaggle dataset or the NHL's own API.
- **The value model excludes salary as an input on purpose.** If salary influenced what counts as "valuable," an expensive player would look better just for costing more - backwards for finding bargains. Salary is compared to the model's output afterward instead ("surplus value"), not blended into it.
- **Raw counting stats were replaced with per-game rates after a real bug.** An early version of the model rated shot-blocking as more valuable than goal-scoring - not because blocks matter more, but because the model had no way to account for ice time, so counting stats were silently acting as a proxy for playing time. Converting to per-game rates (and adding games played as its own feature) fixed this.
- **The draft model's added complexity didn't help, and that's reported honestly.** An XGBoost model using pick number + age + position scored *worse* (R²=0.096) than a simple log-linear baseline using pick number alone (R²=0.209) on truly unseen draft classes, even though it had a marginally better MAE. SHAP confirmed pick number dominates the signal; age and position add very little. The simpler model is what actually powers the pick-value curve.
- **Draft-class censoring is handled explicitly, not ignored.** A player drafted in 2019 hasn't had time to accumulate a career yet - training on his (currently low) career value would teach the model that recent picks are worth less than they really are. The model trains only on draft classes 10+ years old (2000-2012); more recent classes get predictions, not ground-truth labels.
- **The composite value model is scoped to skaters for one season (2016-17) plus live current players** - the richest public dataset with defensive/possession stats and salary together (for validating "surplus value") turned out to be a single season with no goalies. Older historical trades and goalies fall back to a cruder career-value estimate (or are explicitly flagged as ungraded) rather than silently producing a confident-looking number from data that doesn't support it.
- **Every valued asset in a trade carries a confidence label** (high/medium/low/none) reflecting which of these paths it came from, and unvalued assets are shown as unvalued, never treated as zero.
- **Player and pick values had to be put on the same timescale, after a real red flag, and the first fix over-corrected.** Picks are valued as an expected *career* total (the pick-value curve), but an early version of the trade grader valued players as a single *season* total - so a late 1st-round pick's whole-career expectation looked almost as valuable as one season of Connor McDavid. First fix: project a player's rate over a full career, using the population-average total games for their draft slot. That over-corrected - it's dragged down by early-bust picks who never had a real NHL career, so an already-proven, healthy 29-year-old McDavid (with real games already played) looked like his career was nearly over. Final fix: estimate remaining career games from age alone (an assumed retirement age), independent of draft slot - once a player has real performance data confirming they're an established NHLer, the bust-inclusive population average is the wrong reference class entirely. This is applied consistently to both the live/season-specific path and the 2016-17 precomputed table (which needed `age` and `draft_overall` added to its saved columns to support it).

## Validation

- The trained value model's "biggest bargains" list is dominated by young stars on entry-level contracts (e.g. Draisaitl, Matthews, Pastrnak) and its "biggest overpays" list matches contracts that were widely considered bad at the time (Kane, Weber, Subban, Parise, Suter) - a real-world sanity check the model wasn't explicitly trained to pass.
- The 2017 Martin Hanzal trade (Minnesota gave up a 1st, 2nd, and conditional 4th round pick for a rental) is a useful negative example, not a clean validation. An earlier, less-correct version of the value model happened to agree with public consensus that Minnesota overpaid - but only because it valued Hanzal on a single season, understating him relative to the picks' full-career values (see trade_grader.py's docstring). Once player and pick values were put on a consistent career-length scale, the grader flips to favoring Minnesota - which exposed a real, still-unsolved gap: the model has no way to distinguish a rental (a pending free agent who may only play a few months for the acquiring team) from a long-term asset, so it currently projects Hanzal's full remaining-career value as if he'd stay with Minnesota for years. Getting this specific case "right" would need contract/UFA-status data this project doesn't have - left as a known, honestly-reported limitation rather than tuned back to match the old answer.
- The draft pick-value curve is monotonically decreasing by construction (isotonic regression) and produces a realistic shape: ~103 expected career Point Shares for the 1st overall pick, decaying to single digits by the middle rounds.

## Project structure

```
src/
  etl/            data acquisition (Kaggle datasets, live NHL API, trade-text parsing)
  features/       feature engineering + the draft pick-value curve
  models/         value model training, positional scarcity, surplus value, draft model
  trade/          the trade grader that ties everything together
  dashboard/      the Streamlit app
tests/            unit tests for every pure-logic module
data/raw/         downloaded datasets (gitignored - regenerate with the etl scripts)
data/processed/   trained models and derived tables (gitignored - regenerate with the training scripts)
```

## Running it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Data pipeline
python -m src.etl.fetch_draft_data
python -m src.etl.fetch_trade_data
python -m src.etl.clean_trade_data

# Models
python -m src.models.train_value_model
python -m src.models.build_player_valuations
python -m src.models.train_draft_model
python -m src.features.pick_value_curve

# Dashboard
streamlit run src/dashboard/app.py
```

```bash
pytest
```

## Resume bullets (draft)

- Built an NHL player-valuation model (Ridge regression, per-game-rate features) from public data, validated against real contract outcomes (correctly surfaced known-bargain rookie contracts and known-bad veteran contracts without being trained to do so).
- Designed and trained a draft-outcome model handling right-censored labels (recent draft classes), using a time-based train/test split to honestly measure generalization to unseen draft classes; used SHAP to show a simpler baseline model was actually more robust than a gradient-boosted alternative, and shipped the simpler model.
- Built an end-to-end trade-grading tool combining two ML models with a rule-based confidence system, correctly reproducing public consensus on a real historical trade; extended it to value hypothetical trades using live data pulled from an undocumented public API.
- Parsed 15,000+ free-text trade records into structured data via a regex-based pipeline covering conditional draft picks, nested clauses, and real-world data-quality issues (typos, ambiguous names), backed by 38 unit tests.

## Known limitations

- Player-value coverage is strongest for the 2016-17 season (exact precomputed model output). Other seasons use that player's real season-specific stats pulled live from the NHL API through the same model (medium confidence) - the career-total fallback only applies when no season-specific NHL record exists for that player (very old players predating the league's digitized stats, or a name that doesn't resolve).
- **Draft Explorer coverage is 2000-2020 (Kaggle) plus 2021-2026 (live NHL API).** The Kaggle dataset's 2021 and especially 2022 rows turned out to be a pre-career snapshot - `point_shares`/`games_played` are NaN for all 225 2022 picks, and 2021's nonzero hit rate is roughly half what 2020's is - so those two years were dropped in favor of the same live-fetch approach used for 2023+. All of 2021-2026 now comes from the NHL's own draft-picks API (`api-web.nhle.com/v1/draft/picks/{year}/all`), which has no career-outcome stat of its own. Their "career Point Shares" column is a deliberate substitution: it's actually OUR composite value model's predicted career value (predicted per-game rate x games played, summed across every NHL regular season a player has appeared in so far), not the real Hockey-Reference stat - most of these players, especially 2023+, don't have a real one yet. Rows are 0 either because a pick hasn't played an NHL game yet (correct/expected for very recent picks) or because their name didn't resolve on the NHL API.
- Goalies aren't covered by the composite value model (the training dataset it's built on doesn't include them).
- Contract cost is only available for the 2016-17 training season - no clean multi-year public cap-hit dataset was found, so "surplus value" can't be computed historically outside that season.
- **The trade history dataset has a real gap: no browsable trade records past ~March 2025.** The source dataset (Kaggle, last updated December 2025) was frozen partway through the 2025-26 season, before that season's trade deadline - historically the highest-volume trade day of the year - and the rest of that season's transactions were never backfilled. Two alternative live sources were checked and ruled out: PuckPedia's terms of use explicitly prohibit scraping (same issue as Hockey-Reference), and NHL Trade Tracker - despite a permissive robots.txt - turned out to be equally stale (its most recent trade is also dated March 7, 2025; a misleading August 2026 sitemap timestamp turned out to be from a bulk site migration, not new content). No further live source was found. In the meantime, any real trade from this gap period can still be graded manually via the dashboard's "Build a hypothetical trade" mode, which pulls live current-season stats for any player name typed in - it just won't appear as a pre-loaded historical entry.
