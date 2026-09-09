"""Generate the S1 submission package for the group_vec candidate.

Run AFTER the cloud gate passes (build OK + cloud_runner E3 all cases
0.00% mismatch).  It:

1. verifies the frozen-file hashes (platform originals must be untouched);
2. copies the submission files into ./s1_package/;
3. writes s1_package/SHA256S1.txt (the manifest the S1 form asks for).

Usage (from anywhere):
    python make_s1_package.py
"""

from __future__ import annotations

import hashlib
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CODE = ROOT / "code"

# Files that must stay byte-identical to the platform originals
# (MhcPre_direct_matmul_batch_v1 / official sources).
FROZEN = [
    "code/op_kernel/mhc_pre_tiling.h",
    "code/op_kernel/tiling_key_mhc_pre.h",
]

# Files that make up the submission.
SUBMIT = [
    "code/op_host/mhc_pre.cpp",
    "code/op_host/CMakeLists.txt",
    "code/op_kernel/mhc_pre.cpp",
    "code/op_kernel/mhc_pre_tiling.h",
    "code/op_kernel/tiling_key_mhc_pre.h",
    "code/op_kernel/CMakeLists.txt",
    "code/CMakeLists.txt",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest().upper()


def fail(msg: str) -> None:
    sys.exit("FAIL: " + msg)


def main() -> None:
    sys.stdout.reconfigure(errors="replace")
    baseline = ROOT.parent / "MhcPre_direct_matmul_batch_v1"

    # 1. Frozen files unchanged vs the batch_v1 baseline (which itself was
    #    verified against the platform originals at copy time).
    for rel in FROZEN:
        cand = ROOT / rel
        base = baseline / rel
        if not cand.exists():
            fail(f"missing {rel}")
        if base.exists() and sha256(cand) != sha256(base):
            fail(f"frozen file changed vs baseline: {rel}")
        print(f"  frozen OK   {sha256(cand)[:16]}...  {rel}")

    # 2. Copy the submission files.
    pkg = ROOT / "s1_package"
    if pkg.exists():
        shutil.rmtree(pkg)
    for rel in SUBMIT:
        src = ROOT / rel
        if not src.exists():
            fail(f"missing {rel}")
        dst = pkg / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    # 3. Manifest.
    lines = []
    for rel in SUBMIT:
        digest = sha256(pkg / rel)
        lines.append(f"{digest}  {rel}")
    manifest = pkg / "SHA256S1.txt"
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for line in lines:
        print("  " + line)
    print(f"\nS1 package ready: {pkg}")
    print("Submit the files listed in SHA256S1.txt; include the manifest as-is.")


if __name__ == "__main__":
    main()