# Detailed project description

The project is organised as a pipeline. `agent.py` coordinates `agent_core`; `dsh.py` provides model loading and Python execution. This document grows as blocks are implemented.

## Block 1 — Schema profiler

```text
Task files → Schema profiler → Data card → Prompt builder
```

`build_data_card(objective, file_paths)` prepares compact file context for the LLM. It preserves `data_dictionary.csv` unchanged. For each data file, it reports the name, dimensions, separator, decimal mark, physical column names, observed missing-value tokens, and first three raw rows.

`_detect_separator` uses `csv.Sniffer`, with a header-frequency fallback. `_detect_decimal_format` uses regular expressions to identify decimal commas or points.

The profiler only describes files. It does not interpret the objective, select columns, clean data, count missing values, remove duplicates, or calculate an answer. Its output is passed to the prompt builder. It has been tested on all 115 file-based tasks in `dev.json`.
