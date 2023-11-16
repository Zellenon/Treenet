import pandas as pd
from pathlib import Path

data_path = Path("MIMIC_FINAL")
data_path.mkdir()
patient_path = data_path / Path("patients")
patient_path.mkdir()

patients = [w.strip() for w in open("mimic-iv-2.2/hosp/patients.txt").readlines()]
discharge = pd.read_csv("mimic-iv-note-deidentified-free-text-clinical-notes-2.2/discharge.csv")
# radiology = pd.read_csv("mimic-iv-note-deidentified-free-text-clinical-notes-2.2/radiology.csv")
print("Making patient dirs")
for patient in patients:
    patient_dir = patient_path / Path(patient)
    matching_records = discharge[discharge['subject_id'] == int(patient)]
    matching_records = matching_records['text']
    if len(matching_records) == 0: continue
    if not patient_dir.exists():
        patient_dir.mkdir()
        # (patient_dir / Path("records")).mkdir()
    # longest_record = max(matching_records, key=lambda x: len(x))
    # with open(patient_dir / Path("record.txt"), 'w') as f:
        # f.write(longest_record)





