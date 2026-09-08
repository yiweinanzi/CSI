# SOTA obstacle audit

Existing evidence only. NON_CLAIM; no method change, new training, gate modification or SOTA claim.

Meters below are the original mean of bank median errors, not pooled medians.

| City | k | Endpoint | Alignment | Response | Full | Full minus Response |
| --- | --- | --- | --- | --- | --- | --- |
| target-boston | 0 | 12.215382 | 13.191927 | 11.594326 | 11.985595 | 0.391269 |
| target-boston | 8 | 15.176891 | 16.727959 | 15.334706 | 16.037143 | 0.702437 |
| target-boston | 32 | 14.612376 | 15.753129 | 13.704010 | 14.398876 | 0.694866 |
| target-boston | 128 | 13.343996 | 14.238239 | 12.647515 | 13.515084 | 0.867569 |
| target-seattle | 0 | 9.531428 | 10.080645 | 8.775194 | 9.151679 | 0.376485 |
| target-seattle | 8 | 12.860347 | 13.072464 | 11.414613 | 13.075028 | 1.660415 |
| target-seattle | 32 | 11.927112 | 12.449548 | 10.317519 | 11.388892 | 1.071373 |
| target-seattle | 128 | 10.964708 | 11.613438 | 10.235400 | 11.097603 | 0.862203 |

Full is strictly first among the four arms in 0/8 cells.
This ranking is descriptive and does not assert significant degradation in every cell.

The original assessed G4 conditions and G5 remain FAIL. Unassessed conditions remain unassessed.
Execution repair cannot change this independent localization table.

Training gradient summaries are retained in audit.json as hypotheses to investigate, not causal findings.
Method changes or main-model retraining require a separately authorized experiment and locked selection protocol.
Selection must not use target query results; repeated target exposure must be disclosed.

Source summary bytes, input hashes, identities and the live progress snapshot are retained alongside this report.
