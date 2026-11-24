"""Build a compact description of the CSV files attached to a task."""

from __future__ import annotations

import csv
import re
from pathlib import Path


POSSIBLE_SEPARATORS = ",;\t|"
MISSING_TOKENS = {"", "?", "-", "na", "n/a", "null", "none", "missing"}


def _detect_separator(text: str) -> str:
    """Detect a CSV separator from the first few lines."""
    sample = "\n".join(text.splitlines()[:10])

    try:
        #csv.Sniffer examines the sample and looks for a repeated separator.
        #csv.Sniffer` is a prebuilt utility from Python’s standard csv module. It uses hand-written heuristics, not machine learning.
        return csv.Sniffer().sniff(sample, delimiters=POSSIBLE_SEPARATORS).delimiter 
    except csv.Error:
        # If automatic detection fails, use the separator seen most often in the header.
        header = sample.splitlines()[0] if sample else ""
        return max(POSSIBLE_SEPARATORS, key=header.count)


def _detect_decimal_format(values: list[str]) -> str:
    """Describe the decimal mark found in numeric-looking values."""

    #Creates a pattern matching values such as 12,50 or 12,50 €
    decimal_comma = re.compile(r"^[+-]?\d+,\d+(?:\s*[^\d].*)?$")
    decimal_point = re.compile(r"^[+-]?\d+\.\d+(?:\s*[^\d].*)?$")

    #Checks whether at least one complete value uses a decimal comma
    uses_comma = any(decimal_comma.fullmatch(value) for value in values)
    #Checks whether at least one uses a decimal point.
    uses_point = any(decimal_point.fullmatch(value) for value in values)

    if uses_comma and uses_point:
        return "mixed comma and point"
    if uses_comma:
        return "comma"
    if uses_point:
        return "point"
    return "not observed"


def build_data_card(objective: str, file_paths: list[str]) -> str:
    """Return a compact, human-readable description of a task's CSV files.

    The function follows seven steps:

    1. Find ``data_dictionary.csv`` and keep its content unchanged.
    2. Inspect every other attached CSV file.
    3. Detect each file's separator and decimal convention.
    4. Report its dimensions and physical column names.
    5. Include its first two or three raw data rows.
    6. Identify the missing-value placeholders used in the file (such as ?, -, n/a, or empty cells)
    7. Return all this information as one compact text card for the LLM.

    ``objective`` is included in the interface because later versions may use
    explicit parsing instructions from it. The current profiler only reports
    what it observes in the files; the prompt builder sends the objective to
    the LLM itself.
    """
    # Converts every path string into a Path object.
    paths = [Path(path) for path in file_paths]
    # Find the path of the dictionnary
    dictionary_path = next(
        (path for path in paths if path.name.lower() == "data_dictionary.csv"),
        None,
    )
    data_paths = [path for path in paths if path != dictionary_path]

    # Start constructing the card
    sections = ["DATA CARD"]

    # Iterate over all the data files in file_paths
    for path in data_paths:
        #Reads the complete file as text.
        text = path.read_text(encoding="utf-8-sig")
        #Calls our helper to detect ,, ;, tab, or |.
        separator = _detect_separator(text)

        # Parse the rows
        rows = list(csv.reader(text.splitlines(), delimiter=separator))
        # Removes completely empty rows.
        rows = [row for row in rows if any(cell.strip() for cell in row)]
        columns = rows[0] if rows else []
        data_rows = rows[1:]
        # Flattens all data cells into one list and removes surrounding spaces.
        values = [cell.strip() for row in data_rows for cell in row]

        
        missing_values = {
            "empty cell" if value == "" else value
            for value in values # Examines every cell value.
            if value.lower() in MISSING_TOKENS # Keeps only recognised placeholders such as ?, -, n/a, or missing
        }
        # Sorts and combines the missing values placeholders:
        missing_text = ", ".join(sorted(missing_values)) or "none observed"

        #Select sample rows
        raw_lines = [line for line in text.splitlines() if line.strip()]
        # Selects the first three data lines, excluding the header at position 0.
        sample_rows = raw_lines[1:4] 
        separator_text = "\\t" if separator == "\t" else separator

        #Add the file description
        sections.extend(
            [
                "",
                f"File: {path.name}",
                f"Shape: {len(data_rows)} rows × {len(columns)} columns",
                f'Separator: "{separator_text}"',
                f"Decimal format: {_detect_decimal_format(values)}",
                f"Columns: {', '.join(columns)}",
                f"Observed missing token: {missing_text}",
                "Sample rows:",
                #The * unpacks each sample row as a separate list item
                *(sample_rows or ["(no data rows)"]),
            ]
        )
    # Add the dictionary to the card
    if dictionary_path is not None:
        dictionary = dictionary_path.read_text(encoding="utf-8-sig").rstrip("\n")
        sections.extend(
            [
                "",
                "Column names, descriptions, and units for attached datafiles:",
                dictionary,
            ]
        )
    else:
        sections.extend(["", "Data dictionary: not provided"])

    return "\n".join(sections)
