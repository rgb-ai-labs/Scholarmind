def precision_at_k(retrieved_titles: list[str], expected_title: str, k: int) -> float:
    top_k = retrieved_titles[:k]
    denominator = min(k, len(retrieved_titles))
    if denominator == 0:
        return 0.0
    matches = sum(1 for title in top_k if title == expected_title)
    return matches / denominator


def recall_at_k(retrieved_titles: list[str], expected_title: str, k: int) -> float:
    top_k = retrieved_titles[:k]
    return 1.0 if expected_title in top_k else 0.0


def faithfulness(supported_flags: list[bool]) -> float:
    if not supported_flags:
        return 1.0
    return sum(1 for flag in supported_flags if flag) / len(supported_flags)
