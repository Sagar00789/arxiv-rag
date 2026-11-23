# Answer-quality results

Each row: retrieve top 5 -> answer with the LLM -> grade with an LLM judge.
Faithfulness and citation accuracy exclude refusals; correctness counts a refusal as wrong.
Per-question details: `eval/answers/*.jsonl`.

| date (UTC) | experiment | answer model | judge model | n | faithfulness | citation acc. | correctness | refusals | gold retrieved | correct when gold retrieved |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-11-23 20:50 | hybrid@0.8 | gemini-3.5-flash-lite | not recorded¹ | 25 | 1.0 | 1.0 | 0.96 | 0.04 | 0.8 | 1.0 |

¹ This first run predates logging the judge model; every later row records it.
