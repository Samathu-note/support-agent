# Decision log

1. **Brand: SpotifyCares (demo uses synthetic data shaped like it).** Chosen for a
   manageable, distinct intent set vs. a firehose brand like AmazonHelp. Swap the
   CSV/`--brand` flag for the real Kaggle brand once you download the full dataset.
2. **Weak supervision instead of hand-labelling training data.** Keyword rules
   auto-label a large pool of tweets to train the classifier; the golden set is
   reserved purely for evaluation so the eval number isn't inflated by testing
   on training data.
3. **TF-IDF + Logistic Regression over a heavier embedding model.** Faster to
   train, no model download needed (keeps the 15-minute reproduce promise),
   and is a defensible "simple baseline that actually works" story.
4. **Retrieval computed on the fly with TF-IDF cosine similarity**, not a
   precomputed vector DB. Fine at this scale (a subsample); documented as the
   first thing to swap for FAISS/pgvector at real scale.
5. **PyMySQL instead of mysqlclient.** Pure-Python driver, no native build step,
   easier to deploy on PaaS platforms with minimal system dependencies.
6. **LLM calls are optional, not required, to run the pipeline.** Both drafting
   and judging fall back to deterministic logic without `OPENAI_API_KEY`, so
   graders can `git clone && migrate && run_pipeline` with zero API cost/setup.
7. **Escalation is rule+confidence hybrid, not model-only.** A single
   uncalibrated confidence score is not enough to trust for financial/legal
   language; hard-coded overrides for legal threats and large dollar amounts.
8. **All predictions are logged to `PredictionLog`**, not just returned and
   discarded -- every decision is auditable in the Django admin.
9. **Golden set columns include `sampling_note`** on every row, forcing
   documentation of *why* each example was chosen, not just what it says.
10. **Judge is a separate LLM call from the drafting call**, not the same
    call self-grading its own output, to reduce the obvious bias of a model
    marking its own homework.
