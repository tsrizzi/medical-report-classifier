"""Sample medical texts used by benchmark/load-testing scripts.

The trained model (`models/triage_model.joblib`) is fit on the Medical
Abstracts TC Corpus, which is in English. Sample texts used to exercise the
API (latency probes, synthetic load, docs examples) must also be in English,
otherwise the TF-IDF vectorizer produces near-zero features and predictions
are meaningless.
"""

SAMPLE_MEDICAL_TEXTS = [
    "Acute abdominal pain required emergency surgical intervention in the cohort.",
    "Routine imaging study of digestive tract anatomy for teaching purposes.",
    "Endoscopy revealed moderate abnormal mucosal changes suggestive of recurrent inflammation.",
]
