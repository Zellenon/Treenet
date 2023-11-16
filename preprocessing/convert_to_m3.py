import pandas as pd

################## NOTEEVENTS.CSV ####################

mimic4 = pd.read_csv("mimic-iv-note-deidentified-free-text-clinical-notes-2.2/discharge.csv")
mimic3 = pd.read_csv("../TransICD/mimicdata/NOTEEVENTS_header")

column_mapping = (
        ('ROW_ID','note_id'),
        ('SUBJECT_ID','subject_id'),
        ('HADM_ID','hadm_id'),
        ('CHARTTIME','charttime'),
        ('STORETIME','storetime'),
        ('TEXT','text'),
        )

for t3, t4 in column_mapping:
    mimic3[t3] = mimic4[t4]

mimic3['iserror'] = False

with open("../TransICD/mimicdata/NOTEEVENTS.csv", 'w') as f:
    f.write(mimic3.to_csv(index=False))


################## DIAGNOSES_ICD.CSV ####################
mimic4 = pd.read_csv("../MIMIC2/mimic-iv-2.2/hosp/diagnoses_icd.csv")
mimic3 = pd.read_csv("../TransICD/mimicdata/DIAGNOSES_ICD_header")

column_mapping = (
        # ('ROW_ID','note_id'),
        ('SUBJECT_ID','subject_id'),
        ('HADM_ID','hadm_id'),
        ('SEQ_NUM','seq_num'),
        ('ICD9_CODE','icd_code'),
        )

for t3, t4 in column_mapping:
    mimic3[t3] = mimic4[t4]

mimic3['ROW_ID'] = 0

with open("../TransICD/mimicdata/DIAGNOSES_ICD.csv", 'w') as f:
    f.write(mimic3.to_csv(index=False))


################## PROCEDURES_ICD.CSV ####################
mimic4 = pd.read_csv("../MIMIC2/mimic-iv-2.2/hosp/procedures_icd.csv")
mimic3 = pd.read_csv("../TransICD/mimicdata/PROCEDURES_ICD_header")

column_mapping = (
        # ('ROW_ID','note_id'),
        ('SUBJECT_ID','subject_id'),
        ('HADM_ID','hadm_id'),
        ('SEQ_NUM','seq_num'),
        ('ICD9_CODE','icd_code'),
        )

for t3, t4 in column_mapping:
    mimic3[t3] = mimic4[t4]

mimic3['ROW_ID'] = 0

with open("../TransICD/mimicdata/PROCEDURES_ICD.csv", 'w') as f:
    f.write(mimic3.to_csv(index=False))


################## D_ICD_DIAGNOSES.CSV ####################
mimic4 = pd.read_csv("../MIMIC2/mimic-iv-2.2/hosp/d_icd_diagnoses.csv")
mimic3 = pd.read_csv("../TransICD/mimicdata/D_ICD_DIAGNOSES_header")

column_mapping = (
        # ('ROW_ID','note_id'),
        ('SHORT_TITLE','long_title'),
        ('LONG_TITLE','long_title'),
        ('ICD9_CODE','icd_code'),
        # ('ICD9_CODE','icd_code'),
        )

for t3, t4 in column_mapping:
    mimic3[t3] = mimic4[t4]

mimic3['ROW_ID'] = 0

with open("../TransICD/mimicdata/D_ICD_DIAGNOSES.csv", 'w') as f:
    f.write(mimic3.to_csv(index=False))


################## D_ICD_PROCEDURES.CSV ####################
mimic4 = pd.read_csv("../MIMIC2/mimic-iv-2.2/hosp/d_icd_procedures.csv")
mimic3 = pd.read_csv("../TransICD/mimicdata/D_ICD_PROCEDURES_header")

column_mapping = (
        # ('ROW_ID','note_id'),
        ('SHORT_TITLE','long_title'),
        ('LONG_TITLE','long_title'),
        # ('SEQ_NUM','seq_num'),
        ('ICD9_CODE','icd_code'),
        )

for t3, t4 in column_mapping:
    mimic3[t3] = mimic4[t4]

mimic3['ROW_ID'] = 0

with open("../TransICD/mimicdata/D_ICD_PROCEDURES.csv", 'w') as f:
    f.write(mimic3.to_csv(index=False))
