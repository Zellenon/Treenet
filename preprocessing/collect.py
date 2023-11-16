import pandas as pd
from pathlib import Path
import re

spacefixer = re.compile(r"[ \n\r\t]{3,}")
emptyfieldfixer = re.compile(r"[a-zA-Z ]+:[ \t]+___\.?")

data_path = Path("MIMIC_FINAL")
patient_path = data_path / Path("patients")
patients = patient_path.iterdir()
patients_icd9 = {"train":[], "test":[]}
patients_icd10 = {"train":[], "test":[]}
icd9 = {"train":[], "test":[]}
icd9tree = {"train":[], "test":[]}
icd10 = {"train":[], "test":[]}
icd10tree = {"train":[], "test":[]}
texts_icd9 = {"train":[], "test":[]}
texts_icd10 = {"train":[], "test":[]}

patients = list(sorted(patients))
patient_data = {
        "train": patients[:int(len(patients)*(7/8))], 
        "test": patients[int(len(patients)*(7/8)):]
        }
for split in ["train", "test"]:
    for patient_dir in patient_data[split]:
        patient = patient_dir.name
        icd9file = patient_dir / Path("icd9.txt")
        icd10file = patient_dir / Path("icd10.txt")
        icd9treefile = patient_dir / Path("icd9-tree.txt")
        icd10treefile = patient_dir / Path("icd10-tree.txt")
        recordfile = patient_dir / Path("record.txt")
        if icd9file.exists():
            patients_icd9[split].append(patient)
            with open(icd9file) as f:
                text = f.read().strip().split(',')
                icd9[split].append(text)
            with open(recordfile) as f:
                texts_icd9[split].append(f.read().strip())
        if icd9treefile.exists():
            with open(icd9treefile) as f:
                text = f.read().strip().split(',')
                icd9tree[split].append(text)
        if icd10file.exists():
            patients_icd10[split].append(patient)
            with open(icd10file) as f:
                text = f.read().strip().split(',')
                icd10[split].append(text)
            with open(recordfile) as f:
                texts_icd10[split].append(f.read().strip())
        if icd10treefile.exists():
            with open(icd10treefile) as f:
                text = f.read().strip().split(',')
                icd10tree[split].append(text)

    print(f"{len(patients_icd9[split])}, {len(texts_icd9[split])}")
    with open(f"{split}_icd9_patients.txt", 'w') as f:
        f.write('\n'.join(patients_icd9[split]))
    with open(f"{split}_icd9_text.txt", 'w') as f:
        f.write('\n'.join(texts_icd9[split]))
    with open(f"{split}_icd9_codes.txt", 'w') as f:
        f.write('\n'.join([','.join(w) for w in icd9[split]]))
    with open(f"{split}_icd9_tree_codes.txt", 'w') as f:
        f.write('\n'.join([','.join(w) for w in icd9tree[split]]))
    print(f"{len(patients_icd10[split])}, {len(texts_icd10[split])}")
    with open(f"{split}_icd10_patients.txt", 'w') as f:
        f.write('\n'.join(patients_icd10[split]))
    with open(f"{split}_icd10_text.txt", 'w') as f:
        f.write('\n'.join(texts_icd10[split]))
    with open(f"{split}_icd10_codes.txt", 'w') as f:
        f.write('\n'.join([','.join(w) for w in icd10[split]]))
    with open(f"{split}_icd10_tree_codes.txt", 'w') as f:
        f.write('\n'.join([','.join(w) for w in icd10tree[split]]))
