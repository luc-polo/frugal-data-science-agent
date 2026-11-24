# Detailed project description

The project is organised as a pipeline. `agent.py` coordinates `agent_core`; `dsh.py` provides model loading and Python execution. This document grows as blocks are implemented.

## Block 1 — Schema profiler

```text
Task files → Schema profiler → Data card → Prompt builder
```

`build_data_card(objective, file_paths)` prepares compact file context for the LLM. It preserves `data_dictionary.csv` unchanged. For each data file, it reports the name, dimensions, separator, decimal mark, physical column names, observed missing-value tokens, and first three raw rows.

-We identify the the separator with a helper: `_detect_separator`. It uses `csv.Sniffer`, with a header-frequency fallback. 
-We detect decimal format with a helper :`_detect_decimal_format`. It uses regular expression matching (ex: re.compile(r"^[+-]?\d+,\d+(?:\s*[^\d].*)?$") to identify decimal commas or points.
-We identify missing values comparing every cell value with the hard coded list of potential Nan placeholders. We also scan the objective prompt for task-specific Nan numeric placeholders, such as `-1`. This process leverages regular expression matching.

Its output is passed to the prompt builder. 
