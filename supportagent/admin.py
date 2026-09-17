from django.contrib import admin
from .models import Tweet, Thread, Intent, GoldenExample, PredictionLog


@admin.register(Tweet)
class TweetAdmin(admin.ModelAdmin):
    list_display = ("tweet_id", "brand", "is_from_company", "created_at")
    list_filter = ("brand", "is_from_company")
    search_fields = ("text", "tweet_id")


@admin.register(Thread)
class ThreadAdmin(admin.ModelAdmin):
    list_display = ("id", "brand", "customer_text", "company_text")
    list_filter = ("brand",)
    search_fields = ("customer_text", "company_text")


@admin.register(Intent)
class IntentAdmin(admin.ModelAdmin):
    list_display = ("name", "description")


@admin.register(GoldenExample)
class GoldenExampleAdmin(admin.ModelAdmin):
    list_display = ("id", "brand", "message_text", "true_intent", "should_escalate")
    list_filter = ("brand", "true_intent", "should_escalate")
    search_fields = ("message_text",)


@admin.register(PredictionLog)
class PredictionLogAdmin(admin.ModelAdmin):
    list_display = ("id", "brand", "input_text_preview", "predicted_intent", "intent_confidence", "escalate", "created_at")
    list_filter = ("brand", "predicted_intent", "escalate", "created_at")
    search_fields = ("input_text", "drafted_reply", "escalation_reason")
    readonly_fields = ("created_at",)

    def input_text_preview(self, obj):
        return obj.input_text[:60] + ("..." if len(obj.input_text) > 60 else "")
    input_text_preview.short_description = "User Message"

