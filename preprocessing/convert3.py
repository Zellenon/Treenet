import pandas as pd
from pathlib import Path
import re

spacefixer = re.compile(r"[ \n\r\t]{3,}")
emptyfieldfixer = re.compile(r"[a-zA-Z ]+:[ \t]+___\.?")

data_path = Path("MIMIC_FINAL")
patient_path = data_path / Path("patients")
patients = patient_path.iterdir()


discharge = pd.read_csv("mimic-iv-note-deidentified-free-text-clinical-notes-2.2/discharge.csv")
for patient_dir in sorted(patients):
    patient = patient_dir.name
    matching_records = discharge[discharge['subject_id'] == int(patient)]
    matching_records = matching_records['text']
    if len(matching_records) == 0: continue
    longest_record = max(matching_records, key=lambda x: len(x))
    longest_record = longest_record.replace('\n', ' ').strip()
    longest_record = re.sub(emptyfieldfixer, "", longest_record)
    longest_record = re.sub(spacefixer, "  ", longest_record)
    with open(patient_dir / Path("record.txt"), 'w') as f:
        f.write(longest_record)
    print(f"Wrote patient {patient} with \"{longest_record[:100]}\"")





