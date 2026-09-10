"""Offline regressions; temporary repositories are created by unittest only."""
import contextlib
import io
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import audit_group_vec as audit
import source_record

ROOT = Path(__file__).resolve().parent


class BoundsAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kernel = audit.load_text(ROOT / "code", audit.KERNEL)
        cls.host = audit.load_text(ROOT / "code", audit.HOST)

    def test_current_source_contract(self):
        audit.check_source_model(self.kernel, self.host)

    def test_reject_old_prefetch_and_each_old_allocation(self):
        mutations = [("xParams.blockCount = static_cast<uint16_t>(nextRows);",
                      "xParams.blockCount = kPreBlockRows;")]
        mutations += [(f"InitBuffer({name}, scratchElements * sizeof(float))",
                       f"InitBuffer({name}, kHinBlockRows * hInTile_ * sizeof(float))")
                      for name in ("calcBuf_", "reduceWorkBuf_")]
        for old, new in mutations:
            with self.subTest(mutation=old), self.assertRaises(SystemExit):
                audit.check_source_model(self.kernel.replace(old, new), self.host)

    def test_reject_stale_constants_and_enabled_probe(self):
        for name, old, new in (("kHinTileElements", 256, 1024),
                               ("kTimingProbeIters", 0, 1500),
                               ("kQueueDepth", 2, 3)):
            with self.subTest(name=name), self.assertRaises(SystemExit):
                audit.check_source_model(self.kernel.replace(f"{name} = {old};", f"{name} = {new};"), self.host)
        with self.assertRaises(SystemExit):
            audit.check_source_model(self.kernel, self.host.replace("kVectorTileElements = 512;", "kVectorTileElements = 2048;"))

    def test_tail_schedule_and_scratch(self):
        with contextlib.redirect_stdout(io.StringIO()):
            audit.check_preprocess_bounds()
        self.assertEqual(audit.preprocess_loads(100, 9), [(100, 8), (108, 1)])
        self.assertEqual(audit.preprocess_loads(100, 16), [(100, 8), (108, 8)])
        self.assertEqual(audit.preprocess_loads(100, 0), [])
        self.assertEqual(audit.preprocess_loads(100, 9, clamp=False)[-1], (108, 8))

    def test_ub_alignment_and_gamma(self):
        self.assertEqual(audit.ub_bytes_for(8, 16384, 2731, True), 148864)
        self.assertEqual(audit.ub_bytes_for(8, 16, 2731, True) -
                         audit.ub_bytes_for(8, 16, 2731, False), 512 * 4)
        for d in (16, 64, 80, 112, 128, 272):
            self.assertLess(audit.ub_bytes_for(8, d, 2731, True), 192 * 1024)


class SourceRecordTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.repo = Path(self.tmp.name)
        shutil.copytree(ROOT / "code", self.repo / "code")
        for name in ("audit_group_vec.py", "source_record.py"):
            shutil.copyfile(ROOT / name, self.repo / name)
        self.git("init", "-q")
        self.git("config", "core.autocrlf", "false")
        self.git("add", ".")
        self.git("-c", "user.name=Offline Audit", "-c", "user.email=audit@example.invalid",
                 "commit", "-qm", "fixture")
        self.kernel = self.repo / "code" / audit.KERNEL

    def git(self, *args):
        return source_record.git(self.repo, *args)

    def record(self):
        return source_record.create_record(self.repo)

    def test_clean_repeat_and_cli(self):
        first = self.record()
        self.assertEqual(first, self.record())
        self.assertFalse(first["git"]["dirty"])
        self.assertTrue(first["source"]["matches_head"])
        self.assertTrue(all(value is None for value in first["verification"].values()))
        result = subprocess.run(["python", "-B", str(ROOT / "source_record.py"),
                                 "--repo", str(self.repo), "--require-head"],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), first)

    def test_dirty_source_and_strict_exit(self):
        first = self.record()
        self.kernel.write_bytes(self.kernel.read_bytes() + b"\n// changed\n")
        second = self.record()
        self.assertEqual(first["git"]["head"], second["git"]["head"])
        self.assertNotEqual(first["source"]["sha256"], second["source"]["sha256"])
        self.assertFalse(second["source"]["matches_head"])
        result = subprocess.run(["python", "-B", str(ROOT / "source_record.py"),
                                 "--repo", str(self.repo), "--require-head"],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        self.assertTrue(json.loads(result.stdout)["git"]["dirty"])

    def test_ignored_source_add_rename_remove(self):
        (self.repo / ".gitignore").write_text("code/ignored*\n")
        self.git("add", ".gitignore")
        self.git("-c", "user.name=Offline Audit", "-c", "user.email=audit@example.invalid",
                 "commit", "-qm", "ignore fixture")
        first = self.record()
        extra = self.repo / "code" / "ignored.h"
        extra.write_bytes(b"ignored but compiled input\n")
        second = self.record()
        self.assertFalse(second["git"]["dirty"])
        self.assertFalse(second["source"]["matches_head"])
        extra = extra.rename(extra.with_name("ignored-renamed.h"))
        third = self.record()
        self.assertNotEqual(second["source"]["sha256"], third["source"]["sha256"])
        extra.unlink()
        self.assertEqual(first["source"], self.record()["source"])
        (self.repo / "code" / "CMakeLists.txt").unlink()
        self.assertFalse(self.record()["source"]["matches_head"])

    def test_clean_git_with_different_line_endings(self):
        self.git("config", "core.autocrlf", "true")
        self.kernel.write_bytes(self.kernel.read_bytes().replace(b"\r\n", b"\n"))
        self.git("add", "--renormalize", "code")
        self.git("-c", "user.name=Offline Audit", "-c", "user.email=audit@example.invalid",
                 "commit", "--allow-empty", "-qm", "normalize fixture")
        self.kernel.write_bytes(self.kernel.read_bytes().replace(b"\n", b"\r\n"))
        self.git("add", "code/op_kernel/mhc_pre.cpp")
        record = self.record()
        self.assertFalse(record["git"]["dirty"], record["git"]["status"])
        self.assertFalse(record["source"]["matches_head"])

    def test_symlink_and_missing_constant_rejected(self):
        link = self.repo / "code" / "linked.h"
        link.symlink_to(self.kernel)
        with self.assertRaises(ValueError):
            self.record()
        link.unlink()
        self.kernel.write_bytes(self.kernel.read_bytes().replace(b"kTimingProbeIters = 0;", b"kTimingProbeIters = (0);"))
        with self.assertRaises(ValueError):
            self.record()


if __name__ == "__main__":
    unittest.main()
