# Hiver SDE Intern Assignment — AI Support Agent (SpotifyCares demo)

An AI customer-support agent built on Django + MySQL that: classifies intent,
drafts a grounded reply from historically similar resolved threads, and
decides whether to auto-handle or escalate to a human — with a reason.

This repo ships with a small synthetic dataset (`data/twcs_sample.csv`) shaped
exactly like the real Kaggle "Customer Support on Twitter" CSV, so the whole
pipeline runs in minutes with zero external downloads or API keys. Swap in
the real Kaggle CSV to get real results (see "Using the real dataset" below).

## Reproduce in under 15 minutes

```bash
git clone <this repo> && cd hiver-agent
pip install -r requirements.txt
python manage.py migrate                 # sqlite by default, no setup needed
python manage.py seed_intents
python manage.py ingest_tweets --csv data/twcs_sample.csv --brand SpotifyCares
python manage.py train_classifier --brand SpotifyCares
python manage.py load_golden --csv data/golden_set.csv
python manage.py run_eval --brand SpotifyCares      # -> eval_report.json
python manage.py run_pipeline --brand SpotifyCares --text "I can't log in, password reset broken"
python manage.py runserver                          # visit http://127.0.0.1:8000 for the live demo UI
```

Optional: set `OPENAI_API_KEY` (copy `.env.example` to `.env`) to get real
LLM-drafted replies and real LLM-judge scoring instead of the deterministic
fallbacks. The pipeline is fully functional either way.

## Using the real dataset

1. Download `twcs.csv` from Kaggle (`thoughtvector/customer-support-on-twitter`).
2. `python manage.py ingest_tweets --csv path/to/twcs.csv --brand <BrandName> --limit 50000`
   (a subsample is expected — see assignment rules).
3. Re-run `train_classifier`, then hand-label your own 150–250 `data/golden_set.csv`
   rows sampled from the real ingested data (this repo's `golden_set.csv` is a
   synthetic placeholder — replace it).
4. Re-run `run_eval`.

## How it works

```
CSV  ──ingest_tweets──▶  Tweet, Thread (MySQL/sqlite via Django ORM)
                              │
                     train_classifier (TF-IDF + LogisticRegression,
                     trained on keyword-weak-labels, NOT the golden set)
                              │
   new message ──▶ classify ──▶ retrieve_similar_threads (TF-IDF cosine sim
                              over this brand's resolved Threads)
                              │
                        draft_reply (LLM grounded on retrieved threads,
                        or deterministic template fallback)
                              │
                     decide_escalation (confidence + retrieval-strength +
                     keyword rules for legal/financial risk)
                              │
                     logged to PredictionLog ──▶ visible in /admin and the demo UI
```

`python manage.py run_eval` scores this whole flow against a hand-labelled
golden set, against two baselines (majority-class trivial, keyword-only
simple), and produces an LLM-judge reply-quality score.

## Project layout

- `supportagent/models.py` — Tweet, Thread, Intent, GoldenExample, PredictionLog
- `supportagent/agent.py` — all the AI logic (classify/retrieve/draft/escalate/judge)
- `supportagent/management/commands/` — CLI entry points (ingest, train, run, eval)
- `supportagent/views.py` + `templates/demo.html` — a one-page live demo UI
- `supportagent/admin.py` — browse all data/logs at `/admin`
- `report.md`, `decision_log.md` — assignment write-ups (fill in after running on real data)

## Deployment (Railway / Render, MySQL-backed)

1. Push this repo to GitHub.
2. Create a MySQL database (Railway/Render/PlanetScale all offer one free-tier).
3. In your host's environment variables, set:
   `USE_SQLITE=false`, `MYSQL_DATABASE`, `MYSQL_USER`, `MYSQL_PASSWORD`,
   `MYSQL_HOST`, `MYSQL_PORT`, `OPENAI_API_KEY`, `DJANGO_SECRET_KEY`,
   `DEBUG=false`.
4. The included `Procfile` runs `python manage.py migrate` on release and
   starts the app with `gunicorn config.wsgi`.
5. Railway: `railway up`. Render: connect the repo, it auto-detects the
   `Procfile`. Both give you a public URL for the `/` demo page and `/admin`.
6. After first deploy: `python manage.py createsuperuser` (via the platform's
   shell) to get admin access, then run the `ingest_tweets`/`train_classifier`
   commands against your real data through that same shell.

## What makes this a "trust" system, not just a reply generator

Every prediction is logged with its evidence: which past threads grounded the
reply, the classifier's confidence, and the exact reason for auto-handling or
escalating. Nothing is a black box — that evidence trail is what you'd show a
reviewer to argue the agent is safe to trust, and it's exactly what feeds the
"what's misleading about my headline number" analysis in `report.md`.
