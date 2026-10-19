# Data schema

Source: Enhanced Westermo dataset v1.0, Zenodo 10.5281/zenodo.8401356. CSV, `sep=";"`.

## Files

- `converted_system11_result.csv` -> system `system11`
- `converted_system13_result.csv` -> system `system13`
- `converted_system21_result.csv` -> system `system21`
- `converted_system28_result.csv` -> system `system28`
- `converted_system30_result.csv` -> system `system30`
- `converted_system33_result.csv` -> system `system33`
- `converted_system46_result.csv` -> system `system46`
- `converted_system64_result.csv` -> system `system64`
- `converted_system96_result.csv` -> system `system96`

## Columns (identical in all 9 files)

| column | dtype as read by pandas | meaning (Zenodo) |
|---|---|---|
| Id | int64 | Unique numeric identifier of the test execution |
| Name | int64 | Unique numeric identifier of the test case |
| Duration | int64 | Approximated runtime of the test case |
| CalcPrio | int64 | Priority of the test case, calculated by the prioritization algorithm (output column, initially 0); dropped |
| LastRun | str | Previous last execution of the test case as date-time-string |
| LastResults | str | List of previous test results (Failed: 1, Passed: 0), ordered by ascending age |
| Verdict | int64 | Test verdict of this test execution (Failed: 1, Passed: 0) |
| Cycle | int64 | The number of the CI cycle this test execution belongs to |

`LastRun` and `LastResults` are read as strings. All other columns are integers. `Name` is already numeric.

## Observed vs documented

In the first cycle of each system no earlier execution exists, yet every row has a `LastRun` and a non-empty `LastResults`.
Within that cycle, rows sorted by Id have consecutive `LastRun` gaps that match the row's own `Duration`
(mean abs error `own`) far better than the previous row's `Duration` (`prev`). `cur` is the share of first-cycle rows whose
`LastResults` is a single entry equal to the row's own `Verdict`.
For the 7 systems with 65 or more first-cycle rows, `LastRun` behaves like the timestamp of the current execution, and `LastResults`
mostly holds the current verdict (cur below 1 is not explained here; rows with longer lists may be duplicate (Cycle, Name) runs).
system11 and system21 have only 3 first-cycle rows, so they are inconclusive.
This differs from the Zenodo wording above. Checked on the first 20000 rows of each file.

| system | first-cycle rows | own (s) | prev (s) | cur |
|---|---|---|---|---|
| system11 | 3 | 183.000 | 233.500 | 1.000 |
| system13 | 65 | 2.219 | 120.156 | 0.969 |
| system21 | 3 | 210.500 | 279.000 | 1.000 |
| system28 | 140 | 4.827 | 54.799 | 0.729 |
| system30 | 183 | 2.615 | 49.626 | 0.781 |
| system33 | 172 | 2.509 | 52.632 | 0.773 |
| system46 | 185 | 2.103 | 51.103 | 0.741 |
| system64 | 134 | 2.376 | 72.406 | 0.761 |
| system96 | 195 | 2.031 | 41.747 | 0.759 |
