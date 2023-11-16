import pandas as pd
from pathlib import Path
from anytree import Node, RenderTree

data_path = Path("MIMIC_FINAL")
patient_path = data_path / Path("patients")
patients = patient_path.iterdir()
codes = {9: set(), 10: set()}
codes_tree = {9: set(), 10: set()}

icd_codes = pd.read_csv("mimic-iv-2.2/hosp/diagnoses_icd.csv")
for patient in sorted(patients):
    patient_codes = icd_codes[icd_codes['subject_id'] == int(patient.name)]
    for icd_version in {9,10}:
        patient_code_set = set()
        patient_code_set_tree = set()
        def do_with_subcodes(f, subcode):
            if len(subcode) < 1:
                return
            f(subcode)
            do_with_subcodes(f, subcode[:-1])
        patient_version_codes = patient_codes[patient_codes['icd_version'] == icd_version]
        if len(patient_version_codes) == 0: continue
        for i, row in patient_version_codes.iterrows():
            code = row['icd_code']
            patient_code_set.add(code)
            do_with_subcodes(patient_code_set_tree.add, code)
            codes[icd_version].add(code)
            do_with_subcodes(codes_tree[icd_version].add, code)
        with open(patient / Path(f"icd{icd_version}.txt"), 'w') as f:
            f.write(",".join(patient_code_set))
            # print(f"Wrote patient {patient.name} icd{icd_version}")
        with open(patient / Path(f"icd{icd_version}-tree.txt"), 'w') as f:
            f.write(",".join(patient_code_set_tree))
            # print(f"Wrote patient {patient.name} icd{icd_version} tree")



for icd_version in {9,10}:
    with open(data_path / Path(f"icd{icd_version}.txt"), 'w') as f:
        f.write(",".join(codes[icd_version]))
    with open(data_path / Path(f"icd{icd_version}-tree.txt"), 'w') as f:
        f.write(",".join(codes_tree[icd_version]))
