-- One patient is counted once even after multiple qualifying visits.
-- Normalize case and surrounding spaces, but require the whole name.
SELECT COUNT(DISTINCT patient_id) AS patient_count
FROM visits
WHERE lower(btrim(department)) = 'neurology'
  AND visit_date >= DATE '2025-01-01'
  AND visit_date < DATE '2026-01-01';
