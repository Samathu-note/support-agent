import os
import json
import pickle
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .models import Thread, PredictionLog
from .agent import IntentClassifier, retrieve_similar_threads, draft_reply, decide_escalation

_cached_clf = None


def get_classifier():
    global _cached_clf
    if _cached_clf is None and os.path.exists("intent_model.pkl"):
        try:
            with open("intent_model.pkl", "rb") as f:
                _cached_clf = pickle.load(f)
        except Exception:
            _cached_clf = IntentClassifier()
    return _cached_clf or IntentClassifier()


@csrf_exempt
def demo(request):
    result = None
    error = None
    brand = request.POST.get("brand") or request.GET.get("brand", "SpotifyCares")
    
    is_ajax = request.headers.get("X-Requested-With") == "XMLHttpRequest" or request.content_type == "application/json"
    
    if request.method == "POST":
        if request.content_type == "application/json":
            try:
                body_data = json.loads(request.body)
                text = body_data.get("text", "").strip()
                brand = body_data.get("brand", brand)
            except Exception:
                text = ""
        else:
            text = request.POST.get("text", "").strip()
            brand = request.POST.get("brand", brand)

        if not text:
            error = "Please enter a customer message before submitting."
            if is_ajax:
                return JsonResponse({"error": error}, status=400)
            return render(request, "demo.html", {"error": error, "brand": brand})

        clf = get_classifier()
        candidates = list(Thread.objects.filter(brand=brand).values("id", "customer_text", "company_text"))
        intent, confidence = clf.predict(text)
        grounding = retrieve_similar_threads(text, candidates, top_k=3)
        reply = draft_reply(text, grounding, brand)
        escalate, reason = decide_escalation(text, intent, confidence, grounding)
        
        # Save to PredictionLog so it appears in /admin and audit logs
        try:
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
        except Exception:
            pass

        result = {
            "text": text,
            "intent": intent,
            "confidence": round(confidence, 2),
            "grounding": grounding,
            "reply": reply,
            "escalate": escalate,
            "reason": reason,
        }

        if is_ajax:
            return JsonResponse({"result": result})

    return render(request, "demo.html", {"result": result, "error": error, "brand": brand})


