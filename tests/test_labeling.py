from ml.labeling import assign_urgency_label


def test_assign_urgency_label_urgent_keyword():
    text = "Patient presented with acute respiratory failure."
    assert assign_urgency_label(text) == "urgente"


def test_assign_urgency_label_attention_keyword():
    text = "Findings show a moderate abnormal growth pattern."
    assert assign_urgency_label(text) == "atenção"


def test_assign_urgency_label_normal_when_no_keywords():
    text = "Routine follow-up abstract discussing general anatomy."
    assert assign_urgency_label(text) == "normal"


def test_assign_urgency_label_urgent_takes_precedence_over_attention():
    text = "A chronic but ultimately fatal case of the disease."
    assert assign_urgency_label(text) == "urgente"
