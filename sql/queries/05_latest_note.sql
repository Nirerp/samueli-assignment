-- Keep visits without notes; note_id breaks created_at ties deterministically.
SELECT v.visit_id, latest.note_id, latest.note_text, latest.created_at
FROM visits AS v
LEFT JOIN LATERAL (
    SELECT n.note_id, n.note_text, n.created_at
    FROM notes AS n
    WHERE n.visit_id = v.visit_id
    ORDER BY n.created_at DESC NULLS LAST, n.note_id DESC
    LIMIT 1
) AS latest ON true
ORDER BY v.visit_id;
