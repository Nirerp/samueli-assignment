# Samueli NLP home assignment

Submission for the NLP Research Scientist home assignment. The [answer PDF](assignment_answers/Samueli%20Assignment.pdf) covers the written questions and prompt design. This repository also contains the [SQL answers](sql/queries/), the [Python pipeline](src/samueli_assignment/pipeline.py), and the [DSPy notebook](notebooks/part_2_2_dspy_prompt_development.ipynb).

The supplied assignment PDF and `Oncology.csv` are not included. The CSV may contain clinical text; keep it local and do not commit it. The notebook uses synthetic notes for prompt optimization. Its exported [prompt draft](output/dspy_pd/part_2_2_prompt_draft.md) and [optimization result](output/dspy_pd/dspy_result.json) are included for review. Synthetic scores are not evidence of clinical performance.

## Python pipeline (Part 3.2)

Install [uv](https://docs.astral.sh/uv/) and run:

```bash
uv sync
mkdir -p data
# Place the supplied Oncology.csv in data/ before running.
uv run samueli-assignment --backend fake
```

The script reads the `transcription` column, creates seeded random binary labels for testing evaluation code, and writes predictions to `Binary_PD` in memory. It prints a confusion matrix, PD precision, recall, F1, ROC-AUC, and counts of model or parsing failures. The fake backend produces arbitrary predictions; its metrics do not measure clinical accuracy. The input CSV is not changed. Use `--output output/predictions.csv` to save row-level results locally.

For a local Ollama model, start Ollama and pull a model first, then run:

```bash
uv run samueli-assignment --backend ollama --model qwen2.5-coder:7b --limit 1
```

Remove `--limit 1` to process the full file. This backend requests structured JSON and validates it against the Pydantic schema. The model name is an example, not a clinically validated choice.

For a hosted model supported by LiteLLM, set that provider's API key in your environment and pass its model identifier:

```bash
export GEMINI_API_KEY='your-key'
uv run samueli-assignment --backend litellm --model gemini/YOUR_MODEL --input path/to/approved_synthetic.csv
```

Only use approved synthetic or de-identified text with a hosted provider. Do not send sensitive clinical records outside the authorized environment. Keys, local data, and prediction outputs are Git-ignored.

## DSPy notebook (Part 2.2)

```bash
uv sync --group notebook
```

Open the notebook with the project `.venv` kernel. To rerun optimization, start Ollama with the model selected by `DSPY_OLLAMA_MODEL` (default `qwen2.5-coder:7b`) and set `GEMINI_API_KEY` in a local `.env` file or the shell. Set `DSPY_GEMINI_MODEL` to change the prompt model. The notebook uses Gemini to propose instructions and Ollama to evaluate them on synthetic notes; it does not send the supplied CSV to Gemini. Gemini quota or billing limits may stop a rerun. The prompt draft needs clinical review and evaluation on appropriate labeled notes before use.

## SQL answers (Part 3.1)

The five PostgreSQL queries are in [`sql/queries/`](sql/queries/). Query 3 has an additional variant for a literal `ICD-10 G20` code. Assumptions about department normalization, visit ties, diagnosis codes, and missing notes are documented in the SQL comments and answer PDF. The local database fixture used during development is not part of this submission.

## AI assistance

Codex helped create the DSPy notebook, write portions of the Python code, refine the initial SQL queries, and clarify assignment questions. The submission was reviewed and finalized by the candidate.
