"""Build the S1 submission package for MhcPre_group_vec_v1.

Steps:
1. Run the offline audit (must PASS).
2. Verify the files that must stay byte-identical to the passed baseline
   (CMakeLists x3 + tiling_key_mhc_pre.h).
3. Compute SHA-256 over the four canonical sources.
4. Zip code/ into S1_submission.zip and emit SHA256_MANIFEST.txt.

Usage:
    python make_submission.py [--ref <passed baseline code dir>]
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODE = HERE / "code"

CANONICAL = [
    "op_host/mhc_pre.cpp",
    "op_kernel/mhc_pre.cpp",
    "op_kernel/mhc_pre_tiling.h",
    "op_kernel/tiling_key_mhc_pre.h",
]
BYTE_IDENTICAL = [
    "CMakeLists.txt",
    "op_host/CMakeLists.txt",
    "op_kernel/CMakeLists.txt",
    "op_kernel/tiling_key_mhc_pre.h",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    args = sys.argv[1:]
    ref = Path(args[args.index("--ref") + 1]) if "--ref" in args else \
        HERE.parent / "MhcPre_direct_matmul_batch_v1" / "code"

    print("== [1/4] offline audit ==")
    r = subprocess.run([sys.executable, str(HERE / "audit_group_vec.py"),
                        "--code", str(CODE)], capture_output=True, text=True)
    sys.stdout.write(r.stdout)
    sys.stderr.write(r.stderr)
    if r.returncode != 0:
        sys.exit("FAIL: audit did not pass; refusing to package")

    print("== [2/4] byte-identical files vs baseline ==")
    if not ref.is_dir():
        sys.exit(f"FAIL: baseline dir missing: {ref}")
    for rel in BYTE_IDENTICAL:
        cand, base = CODE / rel, ref / rel
        if not base.exists():
            print(f"  WARN baseline missing {rel}, skipped")
            continue
        same = cand.read_bytes() == base.read_bytes()
        print(f"  {'OK ' if same else 'DIFF'} {rel}")
        if not same:
            sys.exit(f"FAIL: {rel} differs from the passed baseline")

    print("== [3/4] SHA-256 manifest ==")
    lines = ["# MhcPre_group_vec_v1 S1 manifest", ""]
    for rel in CANONICAL:
        digest = sha256(CODE / rel)
        lines.append(f"{digest}  {rel}")
        print(f"  {digest}  {rel}")
    manifest = HERE / "SHA256_MANIFEST.txt"
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("== [4/4] package ==")
    zpath = HERE / "S1_submission.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(CODE.rglob("*")):
            if p.is_file():
                z.write(p, Path("code") / p.relative_to(CODE))
        z.write(manifest, "SHA256_MANIFEST.txt")
        z.write(HERE / "audit_group_vec.py", "audit_group_vec.py")
    print(f"  wrote {zpath} ({zpath.stat().st_size} bytes)")
    print("S1 package ready (upload gate: cloud E1/E2/E3 must be green first).")


if __name__ == "__main__":
    main()
