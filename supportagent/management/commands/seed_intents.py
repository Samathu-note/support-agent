from django.core.management.base import BaseCommand
from supportagent.models import Intent
from supportagent.agent import INTENT_SEEDS

class Command(BaseCommand):
    help = "Create Intent rows from the taxonomy defined in agent.py"

    def handle(self, *args, **opts):
        for name in INTENT_SEEDS:
            Intent.objects.get_or_create(name=name)
        self.stdout.write(self.style.SUCCESS(f"Seeded {len(INTENT_SEEDS)} intents."))
