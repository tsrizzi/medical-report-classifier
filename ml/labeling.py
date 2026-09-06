import re

URGENT_KEYWORDS = frozenset({
    "acute", "emergency", "critical", "severe", "arrest", "shock",
    "hemorrhage", "haemorrhage", "rupture", "ruptured", "fatal",
    "life-threatening", "sepsis", "septic", "infarction", "malignant",
    "respiratory failure", "cardiac arrest",
})

ATTENTION_KEYWORDS = frozenset({
    "moderate", "abnormal", "elevated", "suspicious", "recurrent",
    "progressive", "complication", "complications", "unstable",
    "chronic", "persistent", "worsening",
})


def _count_matches(text_lower: str, keywords: frozenset[str]) -> int:
    count = 0
    for keyword in keywords:
        pattern = r"\b" + re.escape(keyword) + r"\b"
        if re.search(pattern, text_lower):
            count += 1
    return count


def assign_urgency_label(text: str) -> str:
    text_lower = text.lower()
    if _count_matches(text_lower, URGENT_KEYWORDS) > 0:
        return "urgente"
    if _count_matches(text_lower, ATTENTION_KEYWORDS) > 0:
        return "atenção"
    return "normal"
