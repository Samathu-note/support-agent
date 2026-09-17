"""
Loads the Kaggle 'Customer Support on Twitter' CSV, filters to one brand,
and reconstructs (customer_message -> company_reply) threads.

Usage:
    python manage.py ingest_tweets --csv data/twcs_sample.csv --brand SpotifyCares --limit 20000
"""
import pandas as pd
from django.core.management.base import BaseCommand
from supportagent.models import Tweet, Thread


class Command(BaseCommand):
    help = "Ingest the Kaggle Twitter customer-support CSV for one brand."

    def add_arguments(self, parser):
        parser.add_argument("--csv", required=True)
        parser.add_argument("--brand", required=True)
        parser.add_argument("--limit", type=int, default=20000)

    def handle(self, *args, **opts):
        df = pd.read_csv(opts["csv"], nrows=opts["limit"])
        brand = opts["brand"]

        def as_id(v):
            """Normalize an id-like value (which pandas may read as float
            because of NaNs in the column) to a clean string, or None."""
            if pd.isna(v):
                return None
            try:
                return str(int(float(v)))
            except (ValueError, TypeError):
                return str(v)

        df["tweet_id"] = df["tweet_id"].apply(as_id)
        df["in_response_to_tweet_id"] = df["in_response_to_tweet_id"].apply(as_id)

        # A row belongs to this brand's conversation if either it IS a reply
        # from the brand account, or it is the customer tweet that a brand
        # reply points back to via in_response_to_tweet_id.
        brand_rows = df[df["author_id"] == brand]
        brand_tweet_ids = set(brand_rows["tweet_id"])
        replied_to_ids = set(brand_rows["in_response_to_tweet_id"].dropna())
        relevant_ids = brand_tweet_ids | replied_to_ids
        subset = df[df["tweet_id"].isin(relevant_ids)]

        created = 0
        tweet_objs = {}
        for _, row in subset.iterrows():
            tw, _ = Tweet.objects.update_or_create(
                tweet_id=row["tweet_id"],
                defaults=dict(
                    author_id=str(row["author_id"]),
                    brand=brand,
                    text=str(row["text"]),
                    is_from_company=bool(row["author_id"] == brand),
                    in_response_to_tweet_id=row["in_response_to_tweet_id"],
                ),
            )
            tweet_objs[tw.tweet_id] = tw
            created += 1

        # Reconstruct threads: customer tweet -> the brand's direct reply to it.
        threads_created = 0
        for tid, tw in tweet_objs.items():
            if tw.is_from_company:
                continue
            reply = Tweet.objects.filter(in_response_to_tweet_id=tid, is_from_company=True).first()
            if reply:
                Thread.objects.get_or_create(
                    brand=brand,
                    customer_tweet=tw,
                    company_tweet=reply,
                    defaults=dict(customer_text=tw.text, company_text=reply.text),
                )
                threads_created += 1

        self.stdout.write(self.style.SUCCESS(
            f"Ingested {created} tweets for brand={brand!r}, reconstructed {threads_created} threads."
        ))
