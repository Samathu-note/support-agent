from django.test import TestCase
from supportagent.agent import (
    IntentClassifier,
    weak_label,
    retrieve_similar_threads,
    decide_escalation,
    _template_reply,
)


class AgentUnitTests(TestCase):
    def test_weak_label_rules(self):
        self.assertEqual(weak_label("I cannot log in to my account"), "account_access")
        self.assertEqual(weak_label("Please give me a refund for the charge"), "billing_refund")
        self.assertEqual(weak_label("App crashed when opening"), "playback_bug")
        self.assertEqual(weak_label("Random text without keywords"), "other")

    def test_intent_classifier_fallback(self):
        clf = IntentClassifier()
        intent, conf = clf.predict("I cannot log in")
        self.assertEqual(intent, "account_access")
        self.assertLessEqual(conf, 0.5)

    def test_escalation_rules(self):
        # Legal / angry sentiment escalation
        esc, reason = decide_escalation("This is unacceptable, I want my money back lawsuit", "billing_refund", 0.9, [])
        self.assertTrue(esc)
        self.assertIn("negative-sentiment", reason)

        # High dollar amount escalation
        esc, reason = decide_escalation("Charged $150 twice on my credit card", "billing_refund", 0.9, [{"similarity": 0.8}])
        self.assertTrue(esc)
        self.assertIn("non-trivial amount", reason)

        # Low confidence escalation
        esc, reason = decide_escalation("Hello", "other", 0.3, [{"similarity": 0.8}])
        self.assertTrue(esc)
        self.assertIn("confidence too low", reason)

        # High confidence + good grounding auto-handle
        grounding = [{"thread": {"customer_text": "sample", "company_text": "reply"}, "similarity": 0.5}]
        esc, reason = decide_escalation("Please add playlist sorting", "feature_request", 0.85, grounding)
        self.assertFalse(esc)
        self.assertIn("safe to auto-handle", reason)

    def test_retrieve_similar_threads(self):
        candidates = [
            {"id": 1, "customer_text": "I forgot password", "company_text": "Reset here"},
            {"id": 2, "customer_text": "Song will not play", "company_text": "Restart app"},
        ]
        results = retrieve_similar_threads("How do I reset password?", candidates, top_k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["thread"]["id"], 1)

    def test_template_reply_fallback(self):
        grounding = [{"thread": {"company_text": "Please reset your password at link"}}]
        reply = _template_reply(grounding, "SpotifyCares")
        self.assertIn("SpotifyCares", reply)
        self.assertIn("Please reset your password", reply)
