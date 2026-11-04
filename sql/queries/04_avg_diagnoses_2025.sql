-- Count diagnoses per visit first so visits with no diagnoses contribute zero.
WITH visit_counts AS (
    SELECT v.visit_id,
           lower(btrim(v.department)) AS department,
           COUNT(d.diagnosis_id) AS diagnosis_count
    FROM visits AS v
    LEFT JOIN diagnoses AS d ON d.visit_id = v.visit_id
    WHERE v.visit_date >= DATE '2025-01-01'
      AND v.visit_date < DATE '2026-01-01'
    GROUP BY v.visit_id, lower(btrim(v.department))
)
SELECT department, ROUND(AVG(diagnosis_count), 2) AS avg_diagnoses_per_visit
FROM visit_counts
GROUP BY department
ORDER BY department;
