# utils.py
import numpy as np



# utils/color_helpers.py
def is_white_color(hex_code: str) -> bool:
    return hex_code.lower() in ['#fff', '#ffffff']

def validate_hex_color(hex_code: str) -> bool:
    return bool(re.match(r'^#(?:[0-9a-fA-F]{3}){1,2}$', hex_code))

## ====== Dummy Checkup Start

DUMMY_TAGS = {"style1", "style2", "style3"}
DUMMY_COLOR = "#FFFFFF"
DUMMY_DESC = ["", "na", "none"]


def is_dummy_description(desc):
    return not desc or desc.strip().lower() in DUMMY_DESC

def is_dummy_color(hexcode):
    return hexcode.upper() == DUMMY_COLOR

def is_dummy_tags(tags):
    return set(tags).intersection(DUMMY_TAGS) == set(tags)

def calculate_profile_strength(row):
    valid_fields = [
        bool(row.get("title")),
        not is_dummy_description(row.get("description")),
        not is_dummy_color(row.get("color_hex")),
        not is_dummy_tags(row.get("tags", [])),
        bool(row.get("segment_types"))
    ]
    return round(sum(valid_fields) / len(valid_fields) * 100, 1)

def generate_hint(row):
    hints = []
    if is_dummy_description(row.get("description")): hints.append("Dummy Description")
    if is_dummy_color(row.get("color_hex")): hints.append("Default Color")
    if is_dummy_tags(row.get("tags", [])): hints.append("Generic Tags")
    if row.get("profile_strength", 100) < 60: hints.append("Low Strength")
    return ", ".join(hints)

## ---- Dummy Checkup End ====

def lab_distance(lab1, lab2):
    """
    Compute Delta-E (Euclidean) between two LAB colors.
    """
    l1 = np.array(lab1, dtype=np.float32)
    l2 = np.array(lab2, dtype=np.float32)
    return float(np.linalg.norm(l1 - l2))

def lrv_score(lrv):
    """
    Normalize LRV to a 0-1 scale for matching or boosting.
    """
    return min(max(lrv / 100, 0.0), 1.0)

def is_color_match(lab1, lab2, threshold=5.0):
    return lab_distance(lab1, lab2) <= threshold
