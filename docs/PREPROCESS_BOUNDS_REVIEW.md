# Preprocessing Bounds and Source Identity

Base commit: 8bdebcdda1bf8bbb3e4a122d13398a9ddd28abd4.
Review branch: fix/preprocess-bounds-audit.

## Changes

The next preprocessing DMA previously always loaded eight rows. For an AIV
partition with nine rows, the second load started at row nine but requested
eight rows. It could read past the partition and, on the last worker, past the
input allocation. The guarded prefetch now requests min(8, remaining rows).
The first load, consumer counts, queue depth, and DMA overload are retained.

The shared calc/reduce buffers were sized only for four hIn rows. With D=16,
that provided 64 floats, while preprocessing n=6 needs 96 floats, and invRms
conversion for 65 assigned rows needs 65. Both allocations now use
max(tileElements, kHinBlockRows * hInTile). For D >= 128 their size is unchanged.
This also preserves the larger hIn scratch requirement for D > 128.

The kernel change does not reorder arithmetic, add barriers, alter M64 tiling,
or change the MIX 1:2 flags, direct MatmulImpl, queue depths, or workspace order.
The CPU formula model now reflects the existing folded invRms * alpha scale
and stream-0-first accumulation. This is an audit correction, not a new device
arithmetic optimization.

## Verification

Run from the repository root inside the Codespace:

~~~sh
python -B audit_group_vec.py
python -B -m unittest -v test_offline_audit.py
git -c core.whitespace=cr-at-eol diff --check
~~~

The diff check accepts the existing CRLF line endings without normalizing source.

The audit passes 931 distinct row intervals, including empty workers, local
counts 0..129, and actual 24/48-core partitions. It sweeps n in {4,6,8} and
D=16..16384 in steps of 16, with invRms counts around 64 and 512. Source checks
bind the model to the actual constants, all 16 InitBuffer calls, and the
prefetch branch. They deliberately reject unfamiliar source changes so the
model must be reviewed together with a new implementation.

The maximum modeled AIV allocation is 148,864 bytes, including gamma and each
allocation rounded to 32 bytes, below 192 KiB. This is the explicit buffer
budget for the audited 24-AIC/48-AIV setup, not a measurement of runtime UB use
on every possible platform configuration.

Regression tests reject the old prefetch, each old scratch allocation,
parameter drift, and an enabled timing probe. Version tests cover dirty files,
ignored source files, rename/delete, line-ending conversion, and symlinks.

No CANN build, simulator, NPU correctness run, or latency benchmark was run.
The Codespace PATH has no ccec, bisheng, or npu-smi. The CPU formula check is a
sanity model with rounding helpers, not a bit-exact device oracle. In particular,
SDK ReduceSum temporary-space requirements, physical DMA behavior, synchronization,
and official device arithmetic equivalence still require the target toolchain.

## Version Binding

After committing and with no concurrent edits, capture a source record outside
the repository so the output does not make the worktree dirty:

~~~sh
python -B source_record.py --require-head > /tmp/mhcpre-source-record.json
~~~

Check the command exit status. Exit 0 requires a clean Git worktree and exact
code/ file-byte equality with the recorded HEAD. Exit 1 still emits valid JSON
but rejects the strict binding; exit 2 means capture failed. Without
--require-head, dirty worktrees are recorded explicitly, never labeled as HEAD.

The JSON includes HEAD/tree IDs, actual host/kernel constants, sorted file
paths/sizes/SHA-256 values for all ordinary files under code/ (including ignored
files), and hashes of the audit and record scripts. The source digest is SHA-256
of the files list encoded by json.dumps with ensure_ascii=True, sort_keys=True,
separators=(",", ":"), then ASCII bytes. CRLF and LF are different source bytes.
A clean git status alone is insufficient; matches_head compares committed blob
bytes with actual files. Symlinks and unsupported tree entries are rejected.

Git queries disable optional index writes. Capture repeats source/tool hashes,
HEAD and status checks to detect concurrent changes; it is not an atomic
filesystem snapshot. Capture an idle worktree. The JSON does not include its
own hash or pretend to contain the future commit that will store it.

For each later device run, keep the source JSON alongside an independent run
record containing the build command and CANN version, binary/package SHA-256,
submission/task ID, exact B/S/n/D, dtype, gamma setting, correctness outcome,
raw latency samples and measurement settings. Capture before and after build;
retain both records and investigate any changed code digest. Preserve raw logs.
The source JSON alone cannot prove that a binary was built from those bytes.
Its CANN/device verification fields are intentionally null.

For a non-Git competition directory, retain an immutable copy/archive plus a
sorted per-file byte hash manifest, archive hash and task/result record. A later
Git import must retain that manifest as provenance. This script requires Git;
it fails outside a repository rather than inventing a commit association.

## Next Experiments

First obtain device correctness and a repeated, shape-bound baseline for this
commit. The existing 18-21 us reports cannot be assigned to this change and do
not establish a specific cause for the performance gap.

- A: eight rows and hIn tile 1024 exceed the current UB design; the double input
  queue alone would use 256 KiB, before the fp32 cast buffer.
- C: preserve the padded eight-slot pre/post row stride. activeRows*n is not a
  contiguous valid prefix for n=4/6. Any small-shape threshold needs measurement.
- B: merging hIn loads to rows*n DMA blocks is a separate candidate. Validate
  stride, alignment and tail addresses, then compare the same shapes and binary.
- D: moving Warm work across the flag is a separate scheduling experiment.
  Removing its first producer requires a replacement before the first DeQue;
  bias/ones initialization and the cross-core protocol must still be satisfied.

Keep one experimental change per measured commit. Prioritize waits on the actual
critical path using evidence; fewer source instructions or DMA calls alone do
not establish a speedup. The existing bias/ones fixes (proposal E) were already
present in the base snapshot.

## Historical Artifacts

code/ is the source reviewed here. s1_package/, S1_submission.zip, and the old
SHA256_MANIFEST.txt are historical artifacts and were not regenerated. They
must not be submitted as though they contain this fix. The old snapshot hashes
in REVIEW_START_HERE.md identify only the 2026-09-09 baseline.
