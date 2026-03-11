CHEXPERT_LABELS = [
    "no finding",
    "enlarged cardiomediastinum",
    "cardiomegaly",
    "lung opacity",
    "lung lesion",
    "edema",
    "consolidation",
    "pneumonia",
    "atelectasis",
    "pneumothorax",
    "pleural effusion",
    "pleural other",
    "fracture",
    "support devices",
]

LABEL_SYNONYMS = {
    "no finding": ["no acute disease", "no acute cardiopulmonary disease", "normal"],
    "enlarged cardiomediastinum": ["mediastinal widening", "enlarged mediastinum"],
    "cardiomegaly": ["enlarged heart", "cardiac enlargement"],
    "lung opacity": ["opacity", "infiltrate", "airspace disease"],
    "lung lesion": ["mass", "nodule"],
    "edema": ["pulmonary edema", "vascular congestion"],
    "consolidation": ["airspace consolidation"],
    "pneumonia": ["infection", "infectious process"],
    "atelectasis": ["collapse", "subsegmental atelectasis"],
    "pneumothorax": ["ptx"],
    "pleural effusion": ["effusion"],
    "pleural other": ["pleural thickening"],
    "fracture": ["rib fracture"],
    "support devices": ["line", "tube", "catheter", "pacemaker"],
}


def extract_concepts(text: str):
    t = text.lower()
    found = []
    for label in CHEXPERT_LABELS:
        if label in t:
            found.append(label)
            continue
        for syn in LABEL_SYNONYMS.get(label, []):
            if syn in t:
                found.append(label)
                break
    return sorted(set(found))

