# Triage Labels

The skills speak in terms of five canonical triage roles. This file maps those roles to the actual label strings used in this repo's issue tracker.

| Label in mattpocock/skills | Label in our tracker | Meaning                                  |
| -------------------------- | -------------------- | ---------------------------------------- |
| `needs-triage`             | `needs-triage`       | Maintainer needs to evaluate this issue  |
| `needs-info`               | `needs-info`         | Waiting on reporter for more information |
| `ready-for-agent`          | `ready-for-agent`    | Fully specified, ready for an AFK agent  |
| `ready-for-human`          | `ready-for-human`    | Requires human implementation            |
| `wontfix`                  | `wontfix`            | Will not be actioned                     |
| `cleanup`                  | `cleanup`            | Code quality / dead code / ergonomics    |

## Cleanup sprint (#73-#78)

The original cleanup sprint is complete except for the deliberately deferred experiment
pipeline consolidation. That work now belongs to the method experiment in #55.

| # | Title | Status |
|---|---|---|
| #73 | Remove dead entry point and one-shot scripts | Done |
| #74 | Extract shared graph constants from builder.py/adapter.py | Done |
| #75 | Collapse duplicate config factory methods | Done |
| #76 | Split metrics.py monolith into focused modules | Done via PR #83 |
| #77 | Deduplicate Phi computation | Done |
| #78 | Unify experiment pipeline paths | Superseded by #55 |

When a skill mentions a role (e.g. "apply the AFK-ready triage label"), use the corresponding label string from this table.

Edit the right-hand column to match whatever vocabulary you actually use.
