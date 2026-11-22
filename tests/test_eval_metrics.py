from evals.run import classification_metrics


def test_metrics_penalize_false_positives_and_missed_cases():
    result = classification_metrics(["safety", "safety", "products"], ["safety", "products", "products"])
    assert result["accuracy"] == 2 / 3
    assert result["per_class"]["safety"]["recall"] == 0.5
    assert result["per_class"]["products"]["precision"] == 0.5


def test_empty_metrics_are_unmeasured():
    result = classification_metrics([], [])
    assert result["accuracy"] is None
    assert result["macro_f1"] is None
