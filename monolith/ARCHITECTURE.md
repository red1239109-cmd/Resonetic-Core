Resonetics v2_core is a lightweight event ledger engine based on local files, aimed at achieving tamper resistance (Integrity), execution safety (Safety), and recovery capability (Resync/Repair).

The core consists of the following four elements:

1. **Ledger (Append-only Frames)**: Sequential recording based on length framing (`len:json\n`).

2. **HEAD (Anchor Pointer)**: A header file for quickly referencing the last verified chain tip.

3. **OATH (File Identity Monitoring)**: Tracking and verifying at the OS level whether "the file is the same file."

4. **Lock Constitution (No IO-State Lock)**: Rules to prevent deadlock and contamination between IO, UI, and STATE locks.

---

### 2. Data Model

#### 2.1 Ledger Frame
Each ledger entry is recorded in the following structure.

Storage format: `"{length}:{payload_json}\n"`

Verification unit:
- `sig`: HMAC-SHA256(secret, canonical_json(fields))
- `hash`: SHA256(canonical_json + "|sig:" + sig)
- `prev_hash`: Uses the previous entry's hash as a chain link

Thus, integrity is provided through HMAC (sig) + chain linking (hash/prev_hash).

#### 2.2 HEAD File
The HEAD stores `(global_step, chain_tip_hash)` to provide an upper bound for "how far it matches" during quick boot.

It offers optimization hints for audits to avoid reading the entire ledger every time.

The HEAD is updated via atomic replacement (`.tmp → os.replace`).

#### 2.3 SNAP (Snapshot)
The snapshot is a cache for "state restoration," and `snapshot_loaded=True` is set to True only when a verified snapshot is successfully loaded.

The snapshot is linked to the ledger's anchor/tip, allowing verification of whether it is a "matching snapshot."

---

### 3. Security/Safety Design

#### 3.1 Base Directory Boundary
All file operations cannot escape the BASE_DIR boundary.

Boundary checks are performed OS-specifically via `within_base()`.

- POSIX: Uses resolve + `is_relative_to` (or parts comparison)
- Windows: Applies casefold processing and path prefix checks

#### 3.2 OATH (File Identity Monitoring)
OATH tracks "whether this file is the same file."

- POSIX: Uses `(st_dev, st_ino)` as identity
- Windows: Extracts final path based on handle (`GetFinalPathNameByHandleW`) and canonicalizes

Core files (KEY/HEAD/LEDGER/SNAP) are mandatory OATH targets by default.

If file identity shifts (e.g., reparse point or symbolic replacement), the system is immediately isolated via QUARANTINE.

#### 3.3 Lock Constitution (IO ↔ STATE Separation)
The engine prohibits acquiring the STATE lock during IO.

IO sections are marked with `io_context()`.

`TrackedRLock(io=False)` immediately triggers Fatal if acquired during an IO section.

`_guard_no_state_lock()` prohibits "calling IO while holding the STATE lock."

Streamlit UI interactions are performed outside the STATE lock.

This discipline structurally blocks:
- Deadlock
- UI freezing
- IO-state contamination (partial state updates)

---

### 4. Execution Flow

#### 4.1 Boot Sequence
1. Perform base directory safety check (register core file identities)
2. Attempt snapshot load (set `snapshot_loaded=True` only on successful verification)
3. Perform ledger verify (apply BOOT policy, deep scan)
4. Finalize `gs` (chain step), `hc` (chain tip hash), `ledger_offset`, and log ONLINE

#### 4.2 Tick (Pulse Commit)
Commits a PULSE event at regular intervals (COMMIT_INTERVAL).

Commit acquires HEAD/LEDGER locks in sorted order to prevent deadlock.

If HEAD changes due to external factors (race), `ResoneticsHeadStale` occurs → recovered via rebuild.

#### 4.3 Audit (Fast/Deep)
Audit controls the following based on policy:
- `strict_stream`: Parsing error handling method
- `lock`: Whether to use locks during verification
- `budget`: Scan time budget (partial verification YIELD possible)
- `allow_extra`: Whether to allow schema extensions

Audit results are recorded for forensic purposes:
- `last_audit_frames` (vc/vf)
- `resume_off` / `next_prev_hash`, etc., in state

---

### 5. Failure/Recovery Strategy

#### 5.1 Resync
If the ledger is damaged or broken in the middle,
`_resync()` searches for "next frame start candidate" to continue verification.

If skip limit (`MAX_SKIPS_ALLOWED`) is exceeded, it aborts with Fatal.

#### 5.2 Quarantine
Critical events such as security violations or identity drift immediately set:
- `quarantine=True`
- `running=False`

and record `root_cause`.

---

### 6. Design Philosophy (Summary)
Resonetics is optimized not for "perfect trust," but for:
- Verifiable minimal trust
- Immediate isolation on contamination
- Partial verification and recovery possibility.
