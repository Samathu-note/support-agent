#!/usr/bin/env bash
# Exit on error
set -o errexit

# Install dependencies
pip install -r requirements.txt

# Collect static files
python manage.py collectstatic --no-input

# Apply database migrations
python manage.py migrate

# Seed sample data & train initial classifier if needed
python manage.py seed_intents
python manage.py ingest_tweets --csv data/twcs_sample.csv --brand SpotifyCares
python manage.py train_classifier --brand SpotifyCares
