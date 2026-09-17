"""
Trains the TF-IDF + Logistic Regression intent classifier using WEAK
LABELS derived from keyword rules over ingested customer tweets.
The golden set is deliberately excluded from training -- it exists only
to evaluate the pipeline honestly.

Usage: python manage.py train_classifier --brand SpotifyCares
"""
import pickle
from django.core.management.base import BaseCommand
from supportagent.models import Tweet
from supportagent.agent import IntentClassifier, weak_label


class Command(BaseCommand):
    help = "Train the intent classifier on weakly-labelled customer tweets."

    def add_arguments(self, parser):
        parser.add_argument("--brand", required=True)

    def handle(self, *args, **opts):
        brand = opts["brand"]
        tweets = list(Tweet.objects.filter(brand=brand, is_from_company=False).values_list("text", flat=True))
        if len(tweets) < 20:
            self.stdout.write(self.style.WARNING("Very few tweets ingested -- classifier will be weak. Ingest more first."))

        labels = [weak_label(t) for t in tweets]
        clf = IntentClassifier()
        clf.fit(tweets, labels)

        with open("intent_model.pkl", "wb") as f:
            pickle.dump(clf, f)

        from collections import Counter
        dist = Counter(labels)
        self.stdout.write(self.style.SUCCESS(f"Trained on {len(tweets)} tweets. Label distribution: {dict(dist)}"))
        self.stdout.write("Saved model to intent_model.pkl")
