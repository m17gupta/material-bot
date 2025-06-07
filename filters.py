import numpy as np
from utils import lab_distance

def apply_color_threshold(base_lab, candidates, threshold=5):
    """
    Filters candidates based on LAB Delta-E distance from base_lab.
    Each candidate must include a 'lab' key with [L, a, b].
    """
    results = []
    for item in candidates:
        dist = lab_distance(base_lab, item.get("lab", [0, 0, 0]))
        if dist <= threshold:
            item["delta_e"] = round(dist, 2)
            results.append(item)
    return results

def filter_by_exact_fields(candidates, filters):
    """
    Apply additional filtering on in-memory records after FAISS or DB subset.
    filters is a dict: { 'finish': [...], 'segment_types': [...], ... }
    """
    def match(item):
        for key, val in filters.items():
            if key in item:
                if isinstance(item[key], list):
                    if not any(v in item[key] for v in val):
                        return False
                elif item[key] not in val:
                    return False
        return True
    return [i for i in candidates if match(i)]
