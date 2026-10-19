import glob
import os

import numpy as np
import pandas as pd

RAW = "data/raw/westermo"
EXPECTED = ["Id", "Name", "Duration", "CalcPrio", "LastRun", "LastResults", "Verdict", "Cycle"]

# column meanings as given on the Zenodo page (record 8401356, Table 2)
MEANING = {
    "Id": "Unique numeric identifier of the test execution",
    "Name": "Unique numeric identifier of the test case",
    "Duration": "Approximated runtime of the test case",
    "CalcPrio": "Priority of the test case, calculated by the prioritization algorithm (output column, initially 0); dropped",
    "LastRun": "Previous last execution of the test case as date-time-string",
    "LastResults": "List of previous test results (Failed: 1, Passed: 0), ordered by ascending age",
    "Verdict": "Test verdict of this test execution (Failed: 1, Passed: 0)",
    "Cycle": "The number of the CI cycle this test execution belongs to",
}


def system_name(path):
    return os.path.basename(path).split("_")[1]


def first_cycle_checks(d):
    first = d[d.Cycle == d.Cycle.min()].sort_values("Id")
    t = pd.to_datetime(first.LastRun)
    gap = t.diff().dt.total_seconds()
    own = (gap - first.Duration).abs().mean()
    prev = (gap - first.Duration.shift()).abs().mean()
    lr = first.LastResults.str.strip("[]")
    # first-ever run of a test, yet LastResults already holds exactly the current verdict
    cur = ((lr.str.len() == 1) & (lr == first.Verdict.astype(str))).mean()
    return len(first), own, prev, cur


def main():
    files = sorted(glob.glob(f"{RAW}/*.csv"))
    lines = ["# Data schema", "", "Source: Enhanced Westermo dataset v1.0, Zenodo 10.5281/zenodo.8401356. CSV, `sep=\";\"`.", ""]
    lines += ["## Files", ""] + [f"- `{os.path.basename(f)}` -> system `{system_name(f)}`" for f in files] + [""]
    dtypes = {}
    evidence = []
    for f in files:
        d = pd.read_csv(f, sep=";", nrows=20000)
        assert list(d.columns) == EXPECTED, (f, list(d.columns))
        print("=====", system_name(f))
        print(d.dtypes)
        print(d.head(5).to_string())
        dtypes[system_name(f)] = {c: str(t) for c, t in d.dtypes.items()}
        evidence.append((system_name(f),) + first_cycle_checks(d))

    same = all(v == next(iter(dtypes.values())) for v in dtypes.values())
    assert same, "dtypes differ between systems"
    lines += ["## Columns (identical in all 9 files)", "", "| column | dtype as read by pandas | meaning (Zenodo) |", "|---|---|---|"]
    for c in EXPECTED:
        lines.append(f"| {c} | {next(iter(dtypes.values()))[c]} | {MEANING[c]} |")
    lines += ["", "`LastRun` and `LastResults` are read as strings. All other columns are integers. `Name` is already numeric.", ""]

    lines += [
        "## Observed vs documented",
        "",
        "In the first cycle of each system no earlier execution exists, yet every row has a `LastRun` and a non-empty `LastResults`.",
        "Within that cycle, rows sorted by Id have consecutive `LastRun` gaps that match the row's own `Duration`",
        "(mean abs error `own`) far better than the previous row's `Duration` (`prev`). `cur` is the share of first-cycle rows whose",
        "`LastResults` is a single entry equal to the row's own `Verdict`.",
        "For the 7 systems with 65 or more first-cycle rows, `LastRun` behaves like the timestamp of the current execution, and `LastResults`",
        "mostly holds the current verdict (cur below 1 is not explained here; rows with longer lists may be duplicate (Cycle, Name) runs).",
        "system11 and system21 have only 3 first-cycle rows, so they are inconclusive.",
        "This differs from the Zenodo wording above. Checked on the first 20000 rows of each file.",
        "",
        "| system | first-cycle rows | own (s) | prev (s) | cur |",
        "|---|---|---|---|---|",
    ]
    for s, n, own, prev, cur in evidence:
        lines.append(f"| {s} | {n} | {own:.3f} | {prev:.3f} | {cur:.3f} |")
    lines.append("")

    os.makedirs("results", exist_ok=True)
    with open("results/data_schema.md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


if __name__ == "__main__":
    main()
