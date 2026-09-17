"""
Runs the full agent pipeline on either a single --text message, or on every
Tweet for --brand that hasn't been logged yet. Logs everything to
PredictionLog so results are inspectable in the Django admin.

Usage:
    python manage.py run_pipeline --brand SpotifyCares --text "I can't log into my account!"
    python manage.py run_pipeline --brand SpotifyCares --batch 50
"""
import pickle
import os
from django.core.management.base import BaseCommand
from supportagent.models import Tweet, Thread, PredictionLog
from supportagent.agent import IntentClassifier, retrieve_similar_threads, draft_reply, decide_escalation


class Command(BaseCommand):
    help = "Run the classify -> retrieve -> draft -> escalate pipeline."

    def add_arguments(self, parser):
        parser.add_argument("--brand", required=True)
        parser.add_argument("--text", default=None)
        parser.add_argument("--batch", type=int, default=0)

    def handle(self, *args, **opts):
        brand = opts["brand"]
        clf = IntentClassifier()
        if os.path.exists("intent_model.pkl"):
            with open("intent_model.pkl", "rb") as f:
                clf = pickle.load(f)
        else:
            self.stdout.write(self.style.WARNING("No trained model found -- using keyword fallback. Run train_classifier first."))

        candidates = list(Thread.objects.filter(brand=brand).values("id", "customer_text", "company_text"))

        if opts["text"]:
            targets = [opts["text"]]
        else:
            targets = list(
                Tweet.objects.filter(brand=brand, is_from_company=False)
                .order_by("?")[: opts["batch"] or 10]
                .values_list("text", flat=True)
            )

        for text in targets:
            self._run_one(text, brand, clf, candidates)

    def _run_one(self, text, brand, clf, candidates):
        intent, confidence = clf.predict(text)
        grounding = retrieve_similar_threads(text, candidates, top_k=3)
        reply = draft_reply(text, grounding, brand)
        escalate, reason = decide_escalation(text, intent, confidence, grounding)

        PredictionLog.objects.create(
            brand=brand,
            input_text=text,
            predicted_intent=intent,
            intent_confidence=confidence,
            grounding_thread_ids=[g["thread"]["id"] for g in grounding],
            drafted_reply=reply,
            escalate=escalate,
            escalation_reason=reason,
        )
        self.stdout.write(f"\n--- {text[:80]!r}")
        self.stdout.write(f"Intent: {intent} ({confidence:.2f})  |  Escalate: {escalate} ({reason})")
        self.stdout.write(f"Reply: {reply}")
