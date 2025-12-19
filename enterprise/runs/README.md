# ⚠️ CRITICAL: DO NOT DELETE OR MODIFY

This directory serves as the system's **"Black Box" (Audit Log)** and **"Source of Truth."**
Any modification or deletion of files in this directory will result in the permanent loss of the operational timeline and evidence of intervention (Diffs).

## Files
- **`timeline.jsonl`**: The immutable, append-only event log. This is the primary database.
- **`incidents.json`**: Snapshots of the system state (if enabled).

> "To do is to be... but without a record, the deed never happened."
