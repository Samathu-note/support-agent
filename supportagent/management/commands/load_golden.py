"""
Loads a CSV of hand-labelled golden examples into the database.
Expected columns: brand,message_text,true_intent,should_escalate,escalation_reason,ideal_reply_notes,sampling_note

Usage: python manage.py load_golden --csv data/golden_set.csv
"""
import pandas as pd
from django.core.management.base import BaseCommand
from supportagent.models import GoldenExample, Intent


class Command(BaseCommand):
    help = "Load a hand-labelled golden evaluation CSV."

    def add_arguments(self, parser):
        parser.add_argument("--csv", required=True)

    def handle(self, *args, **opts):
        df = pd.read_csv(opts["csv"])
        count = 0
        for _, row in df.iterrows():
            intent, _ = Intent.objects.get_or_create(name=row["true_intent"])
            GoldenExample.objects.get_or_create(
                brand=row["brand"],
                message_text=row["message_text"],
                defaults=dict(
                    true_intent=intent,
                    should_escalate=bool(row["should_escalate"]),
                    escalation_reason=row.get("escalation_reason", ""),
                    ideal_reply_notes=row.get("ideal_reply_notes", ""),
                    sampling_note=row.get("sampling_note", ""),
                ),
            )
            count += 1
        self.stdout.write(self.style.SUCCESS(f"Loaded {count} golden examples."))
