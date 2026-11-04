-- Alternative for source data that stores either G20 or the literal
-- "ICD-10 G20" in icd10_code. Trim and ignore case, but do not match G20.1.
SELECT p.patient_id
FROM patients AS p
WHERE EXISTS (
    SELECT 1
    FROM visits AS v
    JOIN diagnoses AS d ON d.visit_id = v.visit_id
    WHERE v.patient_id = p.patient_id
      AND upper(btrim(d.icd10_code)) IN ('G20', 'ICD-10 G20')
)
AND NOT EXISTS (
    SELECT 1
    FROM medications AS m
    WHERE m.patient_id = p.patient_id
      AND m.drug_name ILIKE '%levodopa%'
)
ORDER BY p.patient_id;
