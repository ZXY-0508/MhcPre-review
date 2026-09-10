"""Print a byte-level code/ identity as JSON; never write a record into code/."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from audit_group_vec import _strip_block_comments


CONSTANTS = {
    "op_host/mhc_pre.cpp": ("kVectorTileElements", "M64_TILING_M"),
    "op_kernel/mhc_pre.cpp": ("kQueueDepth", "kPreBlockRows", "kPostGroupRows",
                               "kHinBlockRows", "kHinTileElements", "kSumBatchRows",
                               "kTimingProbeIters"),
}


def git(repo: Path, *args: str) -> bytes:
    return subprocess.run(["git", "--no-optional-locks", "-C", str(repo), *args], check=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def entry(path: str, data: bytes) -> dict:
    return {"path": path, "size_bytes": len(data), "sha256": digest(data)}


def source_files(repo: Path) -> list[dict]:
    code = repo / "code"
    if not code.is_dir() or code.is_symlink():
        raise ValueError("code/ must be a real directory")
    result = []
    def scan_error(error):
        raise error

    for directory, dirs, files in os.walk(code, onerror=scan_error, followlinks=False):
        for name in dirs + files:
            path = Path(directory) / name
            if path.is_symlink():
                raise ValueError(f"symlinks are not supported: {path}")
            if path.is_dir():
                continue
            if not path.is_file():
                raise ValueError(f"not a regular source file: {path}")
            result.append(entry(path.relative_to(repo).as_posix(), path.read_bytes()))
    if not result:
        raise ValueError("code/ is empty")
    return sorted(result, key=lambda item: item["path"])


def head_files(repo: Path, head: str) -> list[dict]:
    result = []
    for row in git(repo, "ls-tree", "-rz", "--full-tree", head, "--", "code").split(b"\0"):
        if not row:
            continue
        meta, name = row.split(b"\t", 1)
        mode, kind, object_id = meta.split()
        if kind != b"blob" or mode not in (b"100644", b"100755"):
            raise ValueError("HEAD code/ must contain only regular files")
        result.append(entry(name.decode("utf-8"), git(repo, "cat-file", "blob", object_id.decode())))
    return sorted(result, key=lambda item: item["path"])


def read_constants(repo: Path) -> dict:
    values = {}
    for path, names in CONSTANTS.items():
        text = _strip_block_comments((repo / "code" / path).read_text(encoding="utf-8"))
        values[path] = {}
        for name in names:
            found = re.findall(r"constexpr\s+uint\d+_t\s+" + name +
                               r"\s*=\s*(\d+)[uUlL]*\s*;", text)
            if len(found) != 1:
                raise ValueError(f"expected one literal declaration: {path}: {name}")
            values[path][name] = int(found[0])
    return values


def create_record(repo: Path) -> dict:
    repo = repo.resolve()
    if Path(git(repo, "rev-parse", "--show-toplevel").decode().strip()).resolve() != repo:
        raise ValueError("--repo must name the repository root")
    head = git(repo, "rev-parse", "HEAD").decode().strip()
    status = git(repo, "status", "--porcelain=v1", "-z", "--untracked-files=all")
    files = source_files(repo)
    constants = read_constants(repo)
    committed_files = head_files(repo, head)
    # Pin the audit/record implementation separately from the submitted code/.
    tools = [entry(name, (repo / name).read_bytes())
             for name in ("audit_group_vec.py", "source_record.py")]
    canonical = json.dumps(files, ensure_ascii=True, sort_keys=True,
                           separators=(",", ":")).encode("ascii")
    record = {
        "schema_version": 1,
        "git": {"head": head, "tree": git(repo, "rev-parse", head + "^{tree}").decode().strip(),
                "dirty": bool(status),
                "status": [s.decode("utf-8") for s in status.split(b"\0") if s]},
        "source": {"scope": "all regular files under code/, including ignored files",
                   "files": files, "sha256": digest(canonical),
                   "matches_head": files == committed_files},
        "constants": constants,
        "tools": tools,
        "verification": {"cann_build": None, "device_correctness": None,
                         "device_latency_us": None},
    }
    if (head != git(repo, "rev-parse", "HEAD").decode().strip()
            or status != git(repo, "status", "--porcelain=v1", "-z", "--untracked-files=all")
            or files != source_files(repo)
            or tools != [entry(item["path"], (repo / item["path"]).read_bytes()) for item in tools]):
        raise ValueError("repository changed during capture; retry with an idle worktree")
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--require-head", action="store_true",
                        help="exit 1 unless Git is clean and code/ bytes match HEAD")
    args = parser.parse_args()
    try:
        record = create_record(args.repo)
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"source record failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(record, ensure_ascii=True, sort_keys=True, indent=2))
    if args.require_head and (record["git"]["dirty"] or not record["source"]["matches_head"]):
        print("source record is not a clean HEAD snapshot", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
