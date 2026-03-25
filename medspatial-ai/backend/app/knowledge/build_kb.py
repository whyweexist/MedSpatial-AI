"""
MedSpatial AI — Medical Knowledge Base Builder
Builds a local SQLite database with FTS5 full-text search containing:
- Disease descriptions (ICD-10 mapped, 13 core pathologies)
- Anatomy descriptions (chest structures)
- Radiological finding descriptors
- Differential diagnosis decision trees
"""

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any

# ── Disease Knowledge ────────────────────────────────────────────
DISEASES = [
    {
        "icd10": "J18",
        "name": "Pneumonia",
        "description": (
            "Pneumonia is an acute respiratory infection affecting the lung parenchyma. "
            "On chest X-ray, it presents as airspace consolidation, lobar or segmental, "
            "often with air bronchograms. CT shows ground-glass opacity (GGO) or consolidation "
            "with possible pleural effusion. Most commonly caused by Streptococcus pneumoniae."
        ),
        "radiological_features": "Consolidation, air bronchograms, ground-glass opacity, lobar distribution",
        "hu_characteristics": "Consolidated regions: -100 to 50 HU (vs normal lung: -700 to -500 HU)",
        "differential": ["Tuberculosis", "Pulmonary edema", "Lung cancer", "Atelectasis"],
        "severity_markers": "Bilateral involvement, multilobar disease, cavitation",
        "clinical_action": "Antibiotic therapy, supportive care, hospital admission if severe",
    },
    {
        "icd10": "J93",
        "name": "Pneumothorax",
        "description": (
            "Pneumothorax is the presence of air in the pleural space, causing lung collapse. "
            "On imaging: thin visceral pleural line with absent lung markings peripherally. "
            "Tension pneumothorax is a life-threatening emergency with mediastinal shift."
        ),
        "radiological_features": "Pleural line, absent lung markings, deep sulcus sign on supine CXR",
        "hu_characteristics": "Pleural air space: -900 to -1000 HU",
        "differential": ["Bullous emphysema", "Skin fold artifact", "Pneumomediastinum"],
        "severity_markers": "Size >2cm, tension (mediastinal shift, cardiac compression)",
        "clinical_action": "Observation (small), needle aspiration or chest tube (large/tension)",
    },
    {
        "icd10": "J90",
        "name": "Pleural Effusion",
        "description": (
            "Pleural effusion is abnormal fluid accumulation in the pleural space. "
            "CXR shows blunting of costophrenic angles (>200mL), meniscus sign. "
            "CT quantifies accurately; fluid HU helps distinguish transudative vs exudative."
        ),
        "radiological_features": "Costophrenic blunting, meniscus sign, mediastinal shift if large",
        "hu_characteristics": "Transudative: 0-20 HU; Exudative/Hemorrhagic: 20-80 HU",
        "differential": ["Lung consolidation", "Elevated hemidiaphragm", "Diaphragm tumor"],
        "severity_markers": "Volume >500mL, bilateral effusions, loculated",
        "clinical_action": "Treat underlying cause; thoracentesis if diagnostic or symptomatic",
    },
    {
        "icd10": "D14.3",
        "name": "Lung Nodule/Mass",
        "description": (
            "A pulmonary nodule is a rounded opacity ≤3cm; >3cm = mass. "
            "Solid nodules, ground-glass nodules (GGN), and part-solid nodules. "
            "Key features: size, density, margins, growth rate (doubling time). "
            "Malignancy risk depends on Fleischner Society guidelines."
        ),
        "radiological_features": "Rounded opacity, spiculated vs smooth margins, satellite nodules",
        "hu_characteristics": "Solid: 20-80 HU; Ground-glass: -600 to -200 HU; Calcified: >200 HU",
        "differential": ["Primary lung cancer", "Metastasis", "Hamartoma", "Granuloma", "Carcinoid"],
        "severity_markers": "Spiculated margins, PET avidity, growth, size >8mm",
        "clinical_action": "Fleischner Society follow-up CT; PET-CT; tissue biopsy if suspicious",
    },
    {
        "icd10": "I51.7",
        "name": "Cardiomegaly",
        "description": (
            "Cardiomegaly on CXR: cardiothoracic ratio >0.5. "
            "CT provides precise measurement of cardiac chambers. "
            "Can indicate heart failure, cardiomyopathy, pericardial effusion, or valvular disease."
        ),
        "radiological_features": "CTR >0.5, vascular redistribution, Kerley B lines (if failure)",
        "hu_characteristics": "Cardiac muscle: 40-80 HU; Pericardial fluid: 0-30 HU",
        "differential": ["Pericardial effusion", "Obesity habitus", "Dilated cardiomyopathy"],
        "severity_markers": "CTR >0.6, pulmonary venous hypertension, pleural effusions",
        "clinical_action": "Echo, BNP, cardiology referral",
    },
    {
        "icd10": "J98.1",
        "name": "Atelectasis",
        "description": (
            "Atelectasis is partial or complete lung collapse due to obstruction or compression. "
            "Signs: opacity, volume loss, displacement of fissures, elevation of diaphragm, "
            "shift of mediastinum toward affected side."
        ),
        "radiological_features": "Volume loss, ipsilateral mediastinal shift, dense opacification",
        "hu_characteristics": "Collapsed lung: -500 to -200 HU (denser than normal aerated lung)",
        "differential": ["Pneumonia", "Pleural effusion", "Diaphragmatic pathology"],
        "severity_markers": "Complete lobar or total lung collapse",
        "clinical_action": "Chest physiotherapy, bronchoscopy if endobronchial obstruction",
    },
    {
        "icd10": "J18.0",
        "name": "Consolidation",
        "description": (
            "Pulmonary consolidation = replacement of air in alveoli by fluid, pus, blood, or tumor. "
            "Appears as homogeneous airspace opacification with air bronchograms."
        ),
        "radiological_features": "Homogeneous opacity, air bronchograms, preserved volume",
        "hu_characteristics": "Dense consolidation: -20 to 50 HU",
        "differential": ["Pneumonia", "Lung cancer", "Lymphoma", "Pulmonary edema"],
        "severity_markers": "Bilateral, multilobar, associated effusion",
        "clinical_action": "Identify underlying cause (infection, malignancy, inflammation)",
    },
    {
        "icd10": "J43",
        "name": "Emphysema",
        "description": (
            "Emphysema is permanent enlargement of air spaces distal to terminal bronchioles "
            "with destruction of alveolar walls. CT: low-attenuation areas without walls. "
            "Types: centrilobular, panlobular, paraseptal."
        ),
        "radiological_features": "Hyperinflation, flattened diaphragms, bullae, decreased vascularity",
        "hu_characteristics": "Emphysematous regions: -900 to -950 HU (more negative than normal lung)",
        "differential": ["Bullous lung disease", "Pneumothorax (bulla vs PTX)"],
        "severity_markers": "LAA <-950 HU on CT, FEV1/FVC ratio",
        "clinical_action": "Smoking cessation, bronchodilators, pulmonary rehabilitation",
    },
    {
        "icd10": "J84.1",
        "name": "Fibrosis",
        "description": (
            "Pulmonary fibrosis = scarring of lung tissue leading to stiffness and impaired gas exchange. "
            "CT: honeycombing, traction bronchiectasis, reticular pattern, basilar predominance."
        ),
        "radiological_features": "Honeycombing, traction bronchiectasis, reticular pattern, ground-glass",
        "hu_characteristics": "Fibrotic tissue: -200 to 0 HU (denser than normal lung)",
        "differential": ["NSIP", "HP", "Sarcoidosis", "Drug toxicity"],
        "severity_markers": "Honeycombing, extent >20% lung volume",
        "clinical_action": "Antifibrotic therapy (pirfenidone, nintedanib), lung transplant evaluation",
    },
    {
        "icd10": "S22",
        "name": "Fracture",
        "description": (
            "Rib/sternal/vertebral fractures visible on CT, may be subtle on CXR. "
            "Look for cortical discontinuity, step deformity, and associated soft tissue injury."
        ),
        "radiological_features": "Cortical break, periosteal reaction, step-off, fragment displacement",
        "hu_characteristics": "Bone: 300-1000 HU; Fresh fracture line may show hyperdense hematoma",
        "differential": ["Normal variant ribs", "Metastatic disease", "Bone island"],
        "severity_markers": "Multiple fractured ribs (flail chest), displacement",
        "clinical_action": "Analgesia, respiratory support, surgical stabilization if flail chest",
    },
    {
        "icd10": "A15",
        "name": "Tuberculosis",
        "description": (
            "TB classically presents with upper lobe cavitary disease, tree-in-bud pattern, "
            "and lymphadenopathy. Primary TB: lower/mid zone consolidation + hilar adenopathy. "
            "Miliary TB: diffuse 1-3mm nodules."
        ),
        "radiological_features": "Cavitation, tree-in-bud nodules, upper lobe consolidation, calcified nodes",
        "hu_characteristics": "Cavities: central air (-1000 HU), wall: -100 to 100 HU",
        "differential": ["Lung abscess", "Fungal infection", "Lung cancer with cavitation"],
        "severity_markers": "Bilateral involvement, miliary pattern, massive cavitation",
        "clinical_action": "RIPE therapy (Rifampicin, Isoniazid, Pyrazinamide, Ethambutol)",
    },
    {
        "icd10": "U07.1",
        "name": "COVID-19 patterns",
        "description": (
            "COVID-19 pneumonia: bilateral, peripheral, lower lobe predominant GGO. "
            "CT features: GGO with or without consolidation, crazy-paving pattern, "
            "reverse halo sign. CO-RADS reporting system standardizes findings."
        ),
        "radiological_features": "Bilateral GGO, peripheral, basal, crazy-paving, reverse halo sign",
        "hu_characteristics": "GGO: -600 to -300 HU; Consolidation: -100 to 50 HU",
        "differential": ["Other viral pneumonias", "Organizing pneumonia", "Atypical pneumonia"],
        "severity_markers": "Extent >50% lung, consolidation superimposed on GGO",
        "clinical_action": "Supportive care, anticoagulation, dexamethasone in severe cases",
    },
    {
        "icd10": "Z03.89",
        "name": "Normal/No Finding",
        "description": (
            "Normal chest imaging shows clear lung fields, no consolidation, "
            "normal cardiac silhouette (CTR <0.5), sharp costophrenic angles, "
            "and normal vascular markings."
        ),
        "radiological_features": "Clear lung fields, normal CTR, sharp CP angles, normal vascularity",
        "hu_characteristics": "Lung parenchyma: -700 to -500 HU; normal range",
        "differential": [],
        "severity_markers": "N/A",
        "clinical_action": "No action required; routine follow-up as appropriate",
    },
]

# ── Anatomy Knowledge ────────────────────────────────────────────
ANATOMY = [
    {
        "name": "Lungs",
        "description": "Paired respiratory organs occupying the thoracic cavity. Right lung has 3 lobes; left has 2. The lung parenchyma shows low attenuation (-700 to -500 HU) due to air content.",
        "normal_hu": {"min": -900, "max": -500},
        "adjacent_structures": ["Pleura", "Mediastinum", "Diaphragm", "Chest wall"],
    },
    {
        "name": "Pleura",
        "description": "Two-layered serous membrane surrounding each lung. Visceral pleura covers the lung; parietal pleura lines the chest wall. Normally not visible on imaging.",
        "normal_hu": {"min": 20, "max": 40},
        "adjacent_structures": ["Lung parenchyma", "Chest wall", "Diaphragm"],
    },
    {
        "name": "Heart",
        "description": "Muscular organ in the mediastinum. CTR should be <0.5 on PA CXR. CT shows cardiac chambers, myocardium (50-80 HU), pericardium.",
        "normal_hu": {"min": 40, "max": 80},
        "adjacent_structures": ["Pericardium", "Great vessels", "Left lung", "Diaphragm"],
    },
    {
        "name": "Mediastinum",
        "description": "Central thoracic compartment between the lungs. Contains heart, great vessels, trachea, esophagus, lymph nodes, thymus. Divided into anterior/middle/posterior.",
        "normal_hu": {"min": 30, "max": 60},
        "adjacent_structures": ["Lungs", "Sternum", "Vertebral column", "Diaphragm"],
    },
    {
        "name": "Ribs",
        "description": "12 pairs of curved bones forming the thoracic cage. CT clearly shows cortex (300-1000 HU). Rib fractures may be occult on CXR.",
        "normal_hu": {"min": 300, "max": 1000},
        "adjacent_structures": ["Intercostal muscles", "Pleura", "Vertebrae", "Sternum"],
    },
    {
        "name": "Diaphragm",
        "description": "Dome-shaped musculofibrous partition separating thorax and abdomen. Right dome is higher than left (liver effect). Normal position: right dome at level of anterior 6th rib.",
        "normal_hu": {"min": 40, "max": 60},
        "adjacent_structures": ["Lungs", "Liver", "Stomach", "Spleen"],
    },
    {
        "name": "Trachea",
        "description": "Midline airway from larynx to carina (~T4). Should be midline; tracheal shift indicates mediastinal pathology. Contains air (-1000 HU).",
        "normal_hu": {"min": -1000, "max": -950},
        "adjacent_structures": ["Thyroid", "Esophagus", "Great vessels"],
    },
]


def build_knowledge_base(output_path: str) -> None:
    """Create the SQLite knowledge base with FTS5 search."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if out.exists():
        out.unlink()

    conn = sqlite3.connect(str(out))
    cur = conn.cursor()

    # ── Main tables ──────────────────────────────────────────────
    cur.executescript("""
        CREATE TABLE IF NOT EXISTS diseases (
            id INTEGER PRIMARY KEY,
            icd10 TEXT,
            name TEXT NOT NULL,
            description TEXT,
            radiological_features TEXT,
            hu_characteristics TEXT,
            differential TEXT,
            severity_markers TEXT,
            clinical_action TEXT
        );

        CREATE TABLE IF NOT EXISTS anatomy (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT,
            normal_hu_min REAL,
            normal_hu_max REAL,
            adjacent_structures TEXT
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS disease_fts
            USING fts5(name, description, radiological_features, content='diseases', content_rowid='id');

        CREATE VIRTUAL TABLE IF NOT EXISTS anatomy_fts
            USING fts5(name, description, content='anatomy', content_rowid='id');
    """)

    # ── Insert diseases ──────────────────────────────────────────
    for d in DISEASES:
        cur.execute(
            """INSERT INTO diseases
               (icd10, name, description, radiological_features, hu_characteristics,
                differential, severity_markers, clinical_action)
               VALUES (?,?,?,?,?,?,?,?)""",
            (
                d["icd10"], d["name"], d["description"],
                d["radiological_features"], d["hu_characteristics"],
                json.dumps(d["differential"]),
                d["severity_markers"], d["clinical_action"],
            ),
        )

    # ── Insert anatomy ───────────────────────────────────────────
    for a in ANATOMY:
        cur.execute(
            """INSERT INTO anatomy
               (name, description, normal_hu_min, normal_hu_max, adjacent_structures)
               VALUES (?,?,?,?,?)""",
            (
                a["name"], a["description"],
                a["normal_hu"]["min"], a["normal_hu"]["max"],
                json.dumps(a["adjacent_structures"]),
            ),
        )

    # ── Populate FTS indexes ─────────────────────────────────────
    cur.executescript("""
        INSERT INTO disease_fts(rowid, name, description, radiological_features)
            SELECT id, name, description, radiological_features FROM diseases;

        INSERT INTO anatomy_fts(rowid, name, description)
            SELECT id, name, description FROM anatomy;
    """)

    conn.commit()
    conn.close()

    size_kb = out.stat().st_size / 1024
    print(f"✅ Knowledge base built: {out} ({size_kb:.0f} KB)")
    print(f"   {len(DISEASES)} diseases, {len(ANATOMY)} anatomy entries")


class MedicalKnowledgeBase:
    """Query interface for the SQLite medical knowledge base."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._conn: sqlite3.Connection | None = None

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def search_diseases(self, query: str, limit: int = 5) -> list[dict]:
        """Full-text search across disease entries."""
        conn = self._get_conn()
        rows = conn.execute(
            """SELECT d.* FROM disease_fts fts
               JOIN diseases d ON fts.rowid = d.id
               WHERE disease_fts MATCH ?
               ORDER BY rank LIMIT ?""",
            (query, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def search_anatomy(self, query: str, limit: int = 5) -> list[dict]:
        """Full-text search across anatomy entries."""
        conn = self._get_conn()
        rows = conn.execute(
            """SELECT a.* FROM anatomy_fts fts
               JOIN anatomy a ON fts.rowid = a.id
               WHERE anatomy_fts MATCH ?
               ORDER BY rank LIMIT ?""",
            (query, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_disease(self, name: str) -> dict | None:
        """Get a specific disease by exact name."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM diseases WHERE name LIKE ?", (f"%{name}%",)
        ).fetchone()
        return dict(row) if row else None

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None


def main():
    parser = argparse.ArgumentParser(description="Build MedSpatial AI knowledge base")
    parser.add_argument("--output", type=str, default="./data/medical_kb.sqlite")
    args = parser.parse_args()
    build_knowledge_base(args.output)


if __name__ == "__main__":
    main()
