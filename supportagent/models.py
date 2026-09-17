from django.db import models


class Tweet(models.Model):
    """A single raw tweet from the Kaggle Customer Support on Twitter dataset."""
    tweet_id = models.CharField(max_length=32, unique=True)
    author_id = models.CharField(max_length=64)
    brand = models.CharField(max_length=64, db_index=True)
    text = models.TextField()
    is_from_company = models.BooleanField(default=False)
    in_response_to_tweet_id = models.CharField(max_length=32, null=True, blank=True)
    created_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        indexes = [models.Index(fields=["brand", "is_from_company"])]

    def __str__(self):
        return f"{self.tweet_id} ({'brand' if self.is_from_company else 'customer'})"


class Thread(models.Model):
    """A reconstructed (customer message -> brand's eventual reply) pair.
    This is the unit we retrieve against for grounding new replies."""
    brand = models.CharField(max_length=64, db_index=True)
    customer_tweet = models.ForeignKey(Tweet, related_name="as_customer_msg", on_delete=models.CASCADE)
    company_tweet = models.ForeignKey(Tweet, related_name="as_company_reply", on_delete=models.CASCADE)
    customer_text = models.TextField()
    company_text = models.TextField()
    embedding = models.JSONField(null=True, blank=True)

    def __str__(self):
        return f"Thread[{self.brand}] {self.customer_text[:40]!r}"


class Intent(models.Model):
    """One entry in the intent taxonomy you defined by reading the data."""
    name = models.CharField(max_length=64, unique=True)
    description = models.TextField(blank=True)

    def __str__(self):
        return self.name


class GoldenExample(models.Model):
    """A hand-labelled example for the evaluation set (150-250 required)."""
    brand = models.CharField(max_length=64, db_index=True)
    message_text = models.TextField()
    true_intent = models.ForeignKey(Intent, on_delete=models.SET_NULL, null=True)
    should_escalate = models.BooleanField()
    escalation_reason = models.TextField(blank=True)
    ideal_reply_notes = models.TextField(blank=True, help_text="What a good reply must contain/avoid.")
    sampling_note = models.CharField(max_length=255, blank=True, help_text="Why/how this example was chosen.")

    def __str__(self):
        return f"Golden[{self.brand}] {self.message_text[:40]!r}"


class PredictionLog(models.Model):
    """Every pipeline run gets logged here: prediction + evidence + decision."""
    brand = models.CharField(max_length=64, db_index=True)
    input_text = models.TextField()
    predicted_intent = models.CharField(max_length=64, blank=True)
    intent_confidence = models.FloatField(null=True, blank=True)
    grounding_thread_ids = models.JSONField(default=list, blank=True)
    drafted_reply = models.TextField(blank=True)
    escalate = models.BooleanField(default=False)
    escalation_reason = models.TextField(blank=True)
    judge_score = models.FloatField(null=True, blank=True)
    judge_rationale = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Log[{self.brand}] -> {self.predicted_intent} (escalate={self.escalate})"
