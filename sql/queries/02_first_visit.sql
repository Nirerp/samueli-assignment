-- For visits on the same first date, the lowest visit_id wins.
-- Patients without visits remain in the result with NULL visit fields.
SELECT p.patient_id, first_visit.visit_date, first_visit.department
FROM patients AS p
LEFT JOIN LATERAL (
    SELECT v.visit_date, v.department
    FROM visits AS v
    WHERE v.patient_id = p.patient_id
    ORDER BY v.visit_date ASC NULLS LAST, v.visit_id ASC
    LIMIT 1
) AS first_visit ON true
ORDER BY p.patient_id;
