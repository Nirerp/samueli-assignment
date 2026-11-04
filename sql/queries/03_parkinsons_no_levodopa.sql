-- Treat G20 as an exact code, and any historical prescription as exposure.
SELECT p.patient_id
FROM patients AS p
WHERE EXISTS (
    SELECT 1
    FROM visits AS v
    JOIN diagnoses AS d ON d.visit_id = v.visit_id
    WHERE v.patient_id = p.patient_id
      AND d.icd10_code = 'G20'
)
AND NOT EXISTS (
    SELECT 1
    FROM medications AS m
    WHERE m.patient_id = p.patient_id
      AND m.drug_name ILIKE '%levodopa%'
)
ORDER BY p.patient_id;
