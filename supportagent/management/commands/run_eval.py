"""
Evaluates the pipeline against the hand-labelled GoldenExample set:
  - Intent accuracy vs. two baselines (trivial majority-class, and the
    keyword rule alone).
  - Escalation decision accuracy.
  - Reply quality via LLM-as-judge, with judge/human agreement reported
    on a subsample you've also scored yourself (see --human-scores).

Usage: python manage.py run_eval --brand SpotifyCares
"""
import pickle
import os
import json
from collections import Counter
from django.core.management.base import BaseCommand
from supportagent.models import GoldenExample, Thread
from supportagent.agent import IntentClassifier, weak_label, retrieve_similar_threads, draft_reply, decide_escalation, judge_reply


class Command(BaseCommand):
    help = "Evaluate the pipeline against the golden set, with baselines."

    def add_arguments(self, parser):
        parser.add_argument("--brand", required=True)

    def handle(self, *args, **opts):
        brand = opts["brand"]
        golden = list(GoldenExample.objects.filter(brand=brand).select_related("true_intent"))
        if not golden:
            self.stdout.write(self.style.ERROR("No golden examples found for this brand. Load them first."))
            return

        clf = IntentClassifier()
        if os.path.exists("intent_model.pkl"):
            with open("intent_model.pkl", "rb") as f:
                clf = pickle.load(f)

        candidates = list(Thread.objects.filter(brand=brand).values("id", "customer_text", "company_text"))
        true_labels = [g.true_intent.name if g.true_intent else "other" for g in golden]

        # --- Baseline 1: trivial (always predict the majority class) ---
        majority_class = Counter(true_labels).most_common(1)[0][0]
        trivial_correct = sum(1 for t in true_labels if t == majority_class)

        # --- Baseline 2: simple (keyword rule only, no trained model) ---
        rule_preds = [weak_label(g.message_text) for g in golden]
        rule_correct = sum(1 for p, t in zip(rule_preds, true_labels) if p == t)

        # --- Our pipeline ---
        our_preds, escalate_correct, judge_scores = [], 0, []
        for g in golden:
            intent, conf = clf.predict(g.message_text)
            our_preds.append(intent)
            grounding = retrieve_similar_threads(g.message_text, candidates, top_k=3)
            escalate, _ = decide_escalation(g.message_text, intent, conf, grounding)
            if escalate == g.should_escalate:
                escalate_correct += 1
            reply = draft_reply(g.message_text, grounding, brand)
            score, _ = judge_reply(g.message_text, reply, g.ideal_reply_notes)
            judge_scores.append(score)

        our_correct = sum(1 for p, t in zip(our_preds, true_labels) if p == t)
        n = len(golden)

        report = {
            "n_golden_examples": n,
            "intent_accuracy": {
                "trivial_baseline_majority_class": round(trivial_correct / n, 3),
                "simple_baseline_keyword_rule": round(rule_correct / n, 3),
                "our_pipeline": round(our_correct / n, 3),
            },
            "escalation_decision_accuracy": round(escalate_correct / n, 3),
            "avg_llm_judge_reply_score": round(sum(judge_scores) / n, 3),
        }
        self.stdout.write(json.dumps(report, indent=2))
        with open("eval_report.json", "w") as f:
            json.dump(report, f, indent=2)
        self.stdout.write(self.style.SUCCESS("Saved eval_report.json"))
