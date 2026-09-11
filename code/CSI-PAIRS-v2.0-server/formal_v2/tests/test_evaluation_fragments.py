from __future__ import annotations

from formal_v2.tests.platform_support import symlink_or_skip

import gzip
import hashlib
import os
import sys
import tempfile
import unittest
from pathlib import Path

from formal_v2.formal_evaluation_fragments import (
    CsvFragmentError,
    inspect_csv_fragment,
    merge_csv_fragments,
    visit_csv_fragment_rows,
    write_csv_fragment,
    write_csv_fragment_stream,
)
from formal_v2.formal_evidence import bind_rows
from formal_v2.formal_io import write_csv


class EvaluationFragmentTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_fragment_decompresses_to_exact_legacy_csv_bytes_and_is_deterministic(self):
        rows = [
            {"b": 1, "a": "comma,value"},
            {"c": "line\r\nbreak", "b": 2},
        ]
        evidence = {
            "dataset_sha256": "d" * 64,
            "runtime_provenance": {"not": "a CSV scalar"},
            "ignored_list": [1, 2],
            "fixture": False,
        }
        bound = bind_rows(rows, evidence)
        fields = self._legacy_fieldnames(bound)
        reference = self.root / "reference.csv"
        write_csv(reference, bound)

        first = self.root / "first.csv.gz"
        second = self.root / "second.csv.gz"
        first_receipt = write_csv_fragment(
            first,
            (row for row in rows),
            fieldnames=fields,
            evidence=evidence,
        )
        second_receipt = write_csv_fragment(
            second,
            iter(rows),
            fieldnames=fields,
            evidence=evidence,
        )
        self.assertEqual(gzip.decompress(first.read_bytes()), reference.read_bytes())
        self.assertEqual(first.read_bytes(), second.read_bytes())
        self.assertEqual(first.read_bytes()[4:8], b"\x00\x00\x00\x00")
        self.assertEqual(first.read_bytes()[8:10], b"\x02\xff")
        self.assertEqual(first_receipt.sha256, second_receipt.sha256)
        self.assertEqual(first_receipt.bytes, first.stat().st_size)
        self.assertEqual(first_receipt.row_count, len(rows))
        self.assertEqual(first_receipt.fieldnames, tuple(fields))
        inspected = inspect_csv_fragment(first, fieldnames=fields)
        self.assertEqual(inspected, first_receipt)

    def test_stream_writer_matches_atomic_fragment_bytes(self):
        rows = [{"seed": 1, "value": "a,b"}, {"seed": 2, "value": "line\nvalue"}]
        evidence = {"fixture": False, "nested": {"ignored": True}}
        fields = ["seed", "value", "fixture"]
        reference = self.root / "reference.csv.gz"
        write_csv_fragment(
            reference,
            rows,
            fieldnames=fields,
            evidence=evidence,
        )
        streamed = self.root / "streamed.csv.gz"
        with streamed.open("wb") as raw:
            count = write_csv_fragment_stream(
                raw,
                iter(rows),
                fieldnames=fields,
                evidence=evidence,
            )
        self.assertEqual(count, 2)
        self.assertEqual(streamed.read_bytes(), reference.read_bytes())

        invalid = self.root / "invalid-stream.csv.gz"
        with invalid.open("wb") as raw:
            with self.assertRaisesRegex(CsvFragmentError, "outside the canonical"):
                write_csv_fragment_stream(
                    raw,
                    [{"seed": 1, "extra": True}],
                    fieldnames=["seed"],
                )

    def test_row_visitor_receives_dict_rows_and_returns_receipt(self):
        rows = [
            {"seed": 1, "value": "a,b"},
            {"seed": 2, "value": "line\nvalue"},
        ]
        fields = ["seed", "value"]
        fragment = self.root / "visited.csv.gz"
        written = write_csv_fragment(fragment, rows, fieldnames=fields)
        visited = []

        receipt = visit_csv_fragment_rows(
            fragment,
            fieldnames=fields,
            visitor=visited.append,
        )

        self.assertEqual(
            visited,
            [
                {"seed": "1", "value": "a,b"},
                {"seed": "2", "value": "line\nvalue"},
            ],
        )
        self.assertTrue(all(type(row) is dict for row in visited))
        self.assertEqual(receipt, written)

    def test_row_visitor_accepts_an_empty_fragment(self):
        fragment = self.root / "visited-empty.csv.gz"
        written = write_csv_fragment(fragment, (), fieldnames=["seed", "score"])
        visited = []

        receipt = visit_csv_fragment_rows(
            fragment,
            fieldnames=["seed", "score"],
            visitor=visited.append,
        )

        self.assertEqual(visited, [])
        self.assertEqual(receipt, written)

    def test_row_visitor_rejects_invalid_fragment_streams(self):
        corrupt = self.root / "visited-corrupt.csv.gz"
        corrupt.write_bytes(b"not-gzip")
        with self.assertRaises(CsvFragmentError):
            visit_csv_fragment_rows(
                corrupt,
                fieldnames=["a"],
                visitor=lambda _row: None,
            )

        wrong_width = self.root / "visited-wrong-width.csv.gz"
        self._write_raw_gzip(wrong_width, b"a,b\r\n1\r\n")
        with self.assertRaisesRegex(CsvFragmentError, "row width"):
            visit_csv_fragment_rows(
                wrong_width,
                fieldnames=["a", "b"],
                visitor=lambda _row: None,
            )

        invalid_utf8 = self.root / "visited-invalid-utf8.csv.gz"
        self._write_raw_gzip(invalid_utf8, b"a\r\n\xff\r\n")
        with self.assertRaisesRegex(CsvFragmentError, "corrupt or truncated"):
            visit_csv_fragment_rows(
                invalid_utf8,
                fieldnames=["a"],
                visitor=lambda _row: None,
            )

        first = self.root / "visited-first.csv.gz"
        second = self.root / "visited-second.csv.gz"
        write_csv_fragment(first, [{"a": 1}], fieldnames=["a"])
        write_csv_fragment(second, [{"a": 2}], fieldnames=["a"])
        concatenated = self.root / "visited-concatenated.csv.gz"
        concatenated.write_bytes(first.read_bytes() + second.read_bytes())
        with self.assertRaisesRegex(CsvFragmentError, "concatenated gzip"):
            visit_csv_fragment_rows(
                concatenated,
                fieldnames=["a"],
                visitor=lambda _row: None,
            )

    def test_row_visitor_propagates_callback_exceptions_and_closes_the_stream(self):
        fragment = self.root / "visitor-failure.csv.gz"
        write_csv_fragment(
            fragment,
            ({"a": value} for value in range(3)),
            fieldnames=["a"],
        )
        failure = OSError("visitor stopped")
        visited = []

        def fail_on_second_row(row):
            visited.append(row["a"])
            if row["a"] == "1":
                raise failure

        with self.assertRaises(OSError) as raised:
            visit_csv_fragment_rows(
                fragment,
                fieldnames=["a"],
                visitor=fail_on_second_row,
            )

        self.assertIs(raised.exception, failure)
        self.assertEqual(visited, ["0", "1"])
        self.assertEqual(
            inspect_csv_fragment(fragment, fieldnames=["a"]).row_count,
            3,
        )

    def test_row_visitor_keeps_memory_bounded_for_one_million_rows(self):
        row_total = 1_000_000
        fragment = self.root / "million.csv.gz"
        with fragment.open("wb") as raw:
            with gzip.GzipFile(
                filename="",
                mode="wb",
                compresslevel=9,
                fileobj=raw,
                mtime=0,
            ) as compressed:
                compressed.write(b"value\r\n")
                block = b"x\r\n" * 10_000
                for _ in range(row_total // 10_000):
                    compressed.write(block)
        visited = 0
        baseline_blocks = sys.getallocatedblocks()
        maximum_block_growth = 0

        def count_row(_row):
            nonlocal maximum_block_growth, visited
            visited += 1
            if visited % 100_000 == 0:
                maximum_block_growth = max(
                    maximum_block_growth,
                    sys.getallocatedblocks() - baseline_blocks,
                )

        receipt = visit_csv_fragment_rows(
            fragment,
            fieldnames=["value"],
            visitor=count_row,
        )

        self.assertEqual(visited, row_total)
        self.assertEqual(receipt.row_count, row_total)
        self.assertEqual(receipt.bytes, fragment.stat().st_size)
        self.assertEqual(receipt.sha256, hashlib.sha256(fragment.read_bytes()).hexdigest())
        self.assertLess(maximum_block_growth, 100_000)

    def test_empty_table_matches_legacy_zero_byte_output(self):
        fields = ["seed", "score"]
        reference = self.root / "empty-reference.csv"
        write_csv(reference, bind_rows([], {"fixture": False}))
        self.assertEqual(reference.read_bytes(), b"")

        fragment = self.root / "empty.csv.gz"
        receipt = write_csv_fragment(fragment, iter(()), fieldnames=fields)
        self.assertGreater(receipt.bytes, 0)
        self.assertEqual(receipt.row_count, 0)
        self.assertEqual(gzip.decompress(fragment.read_bytes()), b"")
        self.assertEqual(
            inspect_csv_fragment(fragment, fieldnames=fields).row_count,
            0,
        )

        merged = self.root / "empty-merged.csv"
        merged_receipt = merge_csv_fragments(
            [fragment], merged, fieldnames=fields
        )
        self.assertEqual(merged.read_bytes(), reference.read_bytes())
        self.assertEqual(merged_receipt.bytes, 0)
        self.assertEqual(merged_receipt.row_count, 0)
        self.assertEqual(
            merged_receipt.sha256,
            hashlib.sha256(b"").hexdigest(),
        )

    def test_unicode_uses_the_same_utf8_csv_contract_as_legacy_writer(self):
        rows = [
            {"city": "caf\u00e9", "note": "\u4f60\u597d,world"},
            {"city": "M\u00fcnchen", "note": "line\n\u4e8c"},
        ]
        evidence = {"label": "\u0394", "nested": {"ignored": True}}
        bound = bind_rows(rows, evidence)
        fields = self._legacy_fieldnames(bound)
        reference = self.root / "unicode-reference.csv"
        write_csv(reference, bound)
        fragment = self.root / "unicode.csv.gz"
        write_csv_fragment(
            fragment,
            rows,
            fieldnames=fields,
            evidence=evidence,
        )
        self.assertEqual(gzip.decompress(fragment.read_bytes()), reference.read_bytes())

    def test_non_divisible_fragments_merge_in_canonical_order_with_one_header(self):
        rows = [
            {"seed": 7, "position": index, "score": index / 10.0}
            for index in range(7)
        ]
        evidence = {"fixture": False, "metadata": {"ignored": True}}
        bound = bind_rows(rows, evidence)
        fields = self._legacy_fieldnames(bound)
        reference = self.root / "all-reference.csv"
        write_csv(reference, bound)

        fragments = []
        for index, (start, stop) in enumerate(((0, 3), (3, 6), (6, 7))):
            fragment = self.root / f"fragment-{index}.csv.gz"
            write_csv_fragment(
                fragment,
                (rows[position] for position in range(start, stop)),
                fieldnames=fields,
                evidence=evidence,
            )
            fragments.append(fragment)

        output = self.root / "merged.csv"
        receipt = merge_csv_fragments(fragments, output, fieldnames=fields)
        self.assertEqual(output.read_bytes(), reference.read_bytes())
        self.assertEqual(receipt.row_count, 7)
        self.assertEqual(receipt.bytes, len(reference.read_bytes()))
        self.assertEqual(
            receipt.sha256,
            hashlib.sha256(reference.read_bytes()).hexdigest(),
        )
        expected_header = (",".join(fields) + "\r\n").encode("utf-8")
        self.assertEqual(output.read_bytes().count(expected_header), 1)

    def test_header_mismatch_and_unknown_row_fields_are_rejected(self):
        fragment = self.root / "fields.csv.gz"
        write_csv_fragment(
            fragment,
            [{"a": 1, "b": 2}],
            fieldnames=["a", "b"],
        )
        with self.assertRaisesRegex(CsvFragmentError, "header differs"):
            inspect_csv_fragment(fragment, fieldnames=["b", "a"])
        with self.assertRaisesRegex(CsvFragmentError, "header differs"):
            merge_csv_fragments(
                [fragment],
                self.root / "wrong-schema.csv",
                fieldnames=["a", "c"],
            )

        invalid = self.root / "unknown.csv.gz"
        with self.assertRaisesRegex(CsvFragmentError, "outside the canonical"):
            write_csv_fragment(
                invalid,
                [{"a": 1, "unknown": 2}],
                fieldnames=["a"],
            )
        self.assertFalse(invalid.exists())
        self.assertEqual(self._temporary_files(), [])

        for invalid_fields in ([], [""], ["a", "a"], ["a", 1]):
            with self.subTest(fieldnames=invalid_fields):
                with self.assertRaises(CsvFragmentError):
                    write_csv_fragment(
                        self.root / "invalid-fields.csv.gz",
                        [{"a": 1}],
                        fieldnames=invalid_fields,
                    )

    def test_corrupt_truncated_physical_empty_and_header_only_fragments_are_rejected(self):
        corrupt = self.root / "corrupt.csv.gz"
        corrupt.write_bytes(b"not-gzip")
        with self.assertRaises(CsvFragmentError):
            inspect_csv_fragment(corrupt, fieldnames=["a"])

        physical_empty = self.root / "physical-empty.csv.gz"
        physical_empty.write_bytes(b"")
        with self.assertRaisesRegex(CsvFragmentError, "physically empty"):
            inspect_csv_fragment(physical_empty, fieldnames=["a"])

        valid = self.root / "valid.csv.gz"
        write_csv_fragment(valid, [{"a": "value"}], fieldnames=["a"])
        truncated = self.root / "truncated.csv.gz"
        payload = valid.read_bytes()
        truncated.write_bytes(payload[:-4])
        output = self.root / "existing.csv"
        output.write_bytes(b"preserve-me")
        with self.assertRaisesRegex(CsvFragmentError, "corrupt or truncated"):
            merge_csv_fragments([truncated], output, fieldnames=["a"])
        self.assertEqual(output.read_bytes(), b"preserve-me")

        header_only = self.root / "header-only.csv.gz"
        self._write_raw_gzip(header_only, b"a\r\n")
        with self.assertRaisesRegex(CsvFragmentError, "header but no rows"):
            inspect_csv_fragment(header_only, fieldnames=["a"])

        wrong_level = self.root / "wrong-level.csv.gz"
        self._write_raw_gzip(wrong_level, b"a\r\n1\r\n", compresslevel=1)
        with self.assertRaisesRegex(CsvFragmentError, "level-9"):
            inspect_csv_fragment(wrong_level, fieldnames=["a"])

    def test_symlinks_and_half_written_temporary_fragments_are_rejected(self):
        fragment = self.root / "real.csv.gz"
        write_csv_fragment(fragment, [{"a": 1}], fieldnames=["a"])
        linked = self.root / "linked.csv.gz"
        symlink_or_skip(linked, fragment)
        with self.assertRaisesRegex(CsvFragmentError, "regular file"):
            inspect_csv_fragment(linked, fieldnames=["a"])

        victim = self.root / "victim.csv"
        victim.write_bytes(b"do-not-replace")
        linked_output = self.root / "linked-output.csv"
        symlink_or_skip(linked_output, victim)
        with self.assertRaisesRegex(CsvFragmentError, "regular file"):
            merge_csv_fragments(
                [fragment], linked_output, fieldnames=["a"]
            )
        self.assertEqual(victim.read_bytes(), b"do-not-replace")

        incomplete = self.root / "incomplete.csv.gz"
        temporary = self.root / f".{incomplete.name}.worker.tmp"
        temporary.write_bytes(b"partial")
        with self.assertRaisesRegex(CsvFragmentError, "incomplete temporary"):
            inspect_csv_fragment(incomplete, fieldnames=["a"])

    def test_evidence_conflicts_and_duplicate_fragment_paths_are_rejected(self):
        fragment = self.root / "conflict.csv.gz"
        with self.assertRaisesRegex(CsvFragmentError, "override evidence"):
            write_csv_fragment(
                fragment,
                [{"a": 1, "fixture": True}],
                fieldnames=["a", "fixture"],
                evidence={"fixture": False},
            )
        self.assertFalse(fragment.exists())

        write_csv_fragment(fragment, [{"a": 1}], fieldnames=["a"])
        with self.assertRaisesRegex(CsvFragmentError, "duplicate paths"):
            merge_csv_fragments(
                [fragment, fragment],
                self.root / "duplicates.csv",
                fieldnames=["a"],
            )

    def test_concatenated_gzip_members_are_rejected(self):
        first = self.root / "first.csv.gz"
        second = self.root / "second.csv.gz"
        write_csv_fragment(first, [{"a": 1}], fieldnames=["a"])
        write_csv_fragment(second, [{"a": 2}], fieldnames=["a"])
        concatenated = self.root / "concatenated.csv.gz"
        concatenated.write_bytes(first.read_bytes() + second.read_bytes())
        with self.assertRaisesRegex(CsvFragmentError, "concatenated gzip"):
            inspect_csv_fragment(concatenated, fieldnames=["a"])

    def test_unicode_encoding_failure_is_atomic_and_uses_fragment_error(self):
        target = self.root / "invalid-unicode.csv.gz"
        with self.assertRaisesRegex(CsvFragmentError, "failed to write"):
            write_csv_fragment(
                target,
                [{"value": "\ud800"}],
                fieldnames=["value"],
            )
        self.assertFalse(target.exists())
        self.assertEqual(self._temporary_files(), [])

    @staticmethod
    def _legacy_fieldnames(rows):
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
        return fields

    @staticmethod
    def _write_raw_gzip(path, payload, *, compresslevel=9):
        with path.open("wb") as raw:
            with gzip.GzipFile(
                filename="",
                mode="wb",
                compresslevel=compresslevel,
                fileobj=raw,
                mtime=0,
            ) as compressed:
                compressed.write(payload)

    def _temporary_files(self):
        return sorted(
            path
            for path in self.root.rglob("*")
            if path.is_file() and path.name.endswith(".tmp")
        )


if __name__ == "__main__":
    unittest.main()
