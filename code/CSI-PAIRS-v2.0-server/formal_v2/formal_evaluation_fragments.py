from __future__ import annotations

import csv
import gzip
import hashlib
import io
import os
import tempfile
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Callable, Iterable, Mapping


GZIP_COMPRESSION_LEVEL = 9


class CsvFragmentError(RuntimeError):
    """Raised when a fragment or merged CSV violates the streaming contract."""


@dataclass(frozen=True)
class CsvArtifactReceipt:
    path: Path
    bytes: int
    sha256: str
    row_count: int
    fieldnames: tuple[str, ...]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _lexists(path: Path) -> bool:
    return os.path.lexists(os.fspath(path))


def _temporary_files(path: Path) -> list[Path]:
    if not path.parent.is_dir():
        return []
    return sorted(path.parent.glob(f".{path.name}.*.tmp"))


def _fieldnames(values: Iterable[str]) -> tuple[str, ...]:
    try:
        result = tuple(values)
    except TypeError as error:
        raise CsvFragmentError("CSV fieldnames must be an iterable of strings") from error
    if not result or any(type(value) is not str or not value for value in result):
        raise CsvFragmentError("CSV fieldnames must be nonempty strings")
    if len(result) != len(set(result)):
        raise CsvFragmentError("CSV fieldnames must be unique")
    return result


def _prepare_atomic_target(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise CsvFragmentError(f"CSV parent must be a regular directory: {path.parent}")
    if _lexists(path) and (path.is_symlink() or not path.is_file()):
        raise CsvFragmentError(f"CSV target must be a regular file: {path}")
    if _temporary_files(path):
        raise CsvFragmentError(f"CSV target has an incomplete temporary file: {path}")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish(
    temporary: Path,
    target: Path,
    *,
    row_count: int,
    fieldnames: tuple[str, ...],
) -> CsvArtifactReceipt:
    byte_count = temporary.stat().st_size
    digest = _sha256_file(temporary)
    os.replace(temporary, target)
    _fsync_directory(target.parent)
    return CsvArtifactReceipt(
        path=target,
        bytes=byte_count,
        sha256=digest,
        row_count=row_count,
        fieldnames=fieldnames,
    )


def _bind_evidence(row: object, evidence: Mapping[str, object]) -> dict:
    try:
        bound = dict(row)
    except (TypeError, ValueError) as error:
        raise CsvFragmentError("CSV rows must be mapping-compatible") from error
    for key, value in evidence.items():
        if isinstance(value, (dict, list)):
            continue
        if key in bound and bound[key] != value:
            raise CsvFragmentError(
                f"CSV row attempts to override evidence field {key!r}"
            )
        bound[key] = value
    return bound


def write_csv_fragment_stream(
    raw: BinaryIO,
    rows: Iterable[object],
    *,
    fieldnames: Iterable[str],
    evidence: Mapping[str, object] | None = None,
) -> int:
    """Write one deterministic gzip CSV fragment to an already-open binary stream."""

    fields = _fieldnames(fieldnames)
    if evidence is None:
        evidence = {}
    if not isinstance(evidence, Mapping):
        raise CsvFragmentError("CSV evidence must be a mapping")
    if not callable(getattr(raw, "write", None)):
        raise CsvFragmentError("CSV fragment stream must be writable binary output")
    row_count = 0
    try:
        with gzip.GzipFile(
            filename="",
            mode="wb",
            compresslevel=GZIP_COMPRESSION_LEVEL,
            fileobj=raw,
            mtime=0,
        ) as compressed:
            with io.TextIOWrapper(
                compressed,
                encoding="utf-8",
                errors="strict",
                newline="",
            ) as text:
                writer = csv.DictWriter(text, fieldnames=fields, dialect="excel")
                header_written = False
                for source in rows:
                    bound = _bind_evidence(source, evidence)
                    extras = set(bound).difference(fields)
                    if extras:
                        raise CsvFragmentError(
                            "CSV row has fields outside the canonical header: "
                            f"{sorted(map(str, extras))}"
                        )
                    if not header_written:
                        writer.writeheader()
                        header_written = True
                    writer.writerow(bound)
                    row_count += 1
    except CsvFragmentError:
        raise
    except (csv.Error, OSError, TypeError, UnicodeError, ValueError) as error:
        raise CsvFragmentError("failed to write CSV fragment stream") from error
    return row_count


def write_csv_fragment(
    path: str | Path,
    rows: Iterable[object],
    *,
    fieldnames: Iterable[str],
    evidence: Mapping[str, object] | None = None,
) -> CsvArtifactReceipt:
    """Stream rows into a deterministic, atomically published gzip CSV fragment."""

    target = Path(path)
    fields = _fieldnames(fieldnames)
    if evidence is None:
        evidence = {}
    if not isinstance(evidence, Mapping):
        raise CsvFragmentError("CSV evidence must be a mapping")
    _prepare_atomic_target(target)
    temporary: Path | None = None
    row_count = 0
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=target.parent,
            delete=False,
        ) as raw:
            temporary = Path(raw.name)
            row_count = write_csv_fragment_stream(
                raw,
                rows,
                fieldnames=fields,
                evidence=evidence,
            )
            raw.flush()
            os.fsync(raw.fileno())
        receipt = _publish(
            temporary,
            target,
            row_count=row_count,
            fieldnames=fields,
        )
        temporary = None
        return receipt
    except CsvFragmentError:
        raise
    except (csv.Error, OSError, TypeError, UnicodeError, ValueError) as error:
        raise CsvFragmentError(f"failed to write CSV fragment: {target}") from error
    finally:
        if temporary is not None and _lexists(temporary):
            temporary.unlink()


def _validate_fragment_path(path: Path) -> None:
    temporary = _temporary_files(path)
    if temporary:
        raise CsvFragmentError(f"CSV fragment has an incomplete temporary file: {path}")
    if path.is_symlink() or not path.is_file():
        raise CsvFragmentError(f"CSV fragment must be a regular file: {path}")
    if path.stat().st_size == 0:
        raise CsvFragmentError(f"CSV fragment is physically empty: {path}")


class _ValidatedGzipReader(io.RawIOBase):
    """Expose one gzip member while hashing its compressed bytes."""

    _READ_SIZE = 64 * 1024

    def __init__(self, raw: BinaryIO, path: Path):
        super().__init__()
        self._raw = raw
        self._path = path
        self._decoder = zlib.decompressobj(wbits=16 + zlib.MAX_WBITS)
        self._digest = hashlib.sha256()
        self._byte_count = 0
        self._complete = False
        self._pending = self._read_compressed(10)
        header = self._pending
        if (
            len(header) != 10
            or header[:3] != b"\x1f\x8b\x08"
            or header[3] != 0
            or header[4:8] != b"\x00\x00\x00\x00"
            or header[8:10] != b"\x02\xff"
        ):
            raise CsvFragmentError(
                "CSV fragment is not a deterministic blank-name mtime=0 level-9 "
                f"gzip stream: {path}"
            )

    def readable(self) -> bool:
        return True

    def readinto(self, buffer) -> int:
        if self._complete:
            return 0
        view = memoryview(buffer).cast("B")
        if not view:
            return 0
        try:
            while True:
                if self._decoder.eof:
                    self._finish_member()
                    return 0
                if not self._pending:
                    self._pending = self._read_compressed(self._READ_SIZE)
                    if not self._pending:
                        raise CsvFragmentError(
                            f"CSV fragment is corrupt or truncated: {self._path}"
                        )
                decoded = self._decoder.decompress(self._pending, len(view))
                self._pending = self._decoder.unconsumed_tail
                if self._decoder.unused_data:
                    raise CsvFragmentError(
                        "CSV fragment contains trailing or concatenated gzip data: "
                        f"{self._path}"
                    )
                if decoded:
                    view[: len(decoded)] = decoded
                    return len(decoded)
        except zlib.error as error:
            raise CsvFragmentError(
                f"CSV fragment is corrupt or truncated: {self._path}"
            ) from error

    @property
    def byte_count(self) -> int:
        return self._byte_count

    @property
    def sha256(self) -> str:
        return self._digest.hexdigest()

    def _read_compressed(self, size: int) -> bytes:
        chunk = self._raw.read(size)
        if chunk:
            self._digest.update(chunk)
            self._byte_count += len(chunk)
        return chunk

    def _finish_member(self) -> None:
        if self._pending or self._decoder.unused_data:
            raise CsvFragmentError(
                "CSV fragment contains trailing or concatenated gzip data: "
                f"{self._path}"
            )
        trailing = self._read_compressed(self._READ_SIZE)
        if trailing:
            raise CsvFragmentError(
                "CSV fragment contains trailing or concatenated gzip data: "
                f"{self._path}"
            )
        self._complete = True


def _consume_fragment(
    path: Path,
    fields: tuple[str, ...],
    emit,
) -> CsvArtifactReceipt:
    _validate_fragment_path(path)
    count = 0
    has_header = False
    callback_error: BaseException | None = None
    try:
        with path.open("rb") as raw:
            validated = _ValidatedGzipReader(raw, path)
            with io.BufferedReader(validated) as decompressed:
                with io.TextIOWrapper(
                    decompressed,
                    encoding="utf-8",
                    errors="strict",
                    newline="",
                ) as text:
                    reader = csv.reader(text, dialect="excel", strict=True)
                    try:
                        header = next(reader)
                    except StopIteration:
                        pass
                    else:
                        has_header = True
                        if tuple(header) != fields:
                            raise CsvFragmentError(
                                "CSV fragment header differs from canonical schema: "
                                f"{path}"
                            )
                        for row in reader:
                            if len(row) != len(fields):
                                raise CsvFragmentError(
                                    "CSV fragment row width differs from its header: "
                                    f"{path}"
                                )
                            try:
                                emit(row)
                            except BaseException as error:
                                callback_error = error
                                raise
                            count += 1
    except CsvFragmentError:
        raise
    except (csv.Error, EOFError, gzip.BadGzipFile, OSError, UnicodeError) as error:
        if error is callback_error:
            raise
        raise CsvFragmentError(f"CSV fragment is corrupt or truncated: {path}") from error
    if has_header and count == 0:
        raise CsvFragmentError(f"CSV fragment has a header but no rows: {path}")
    return CsvArtifactReceipt(
        path=path,
        bytes=validated.byte_count,
        sha256=validated.sha256,
        row_count=count,
        fieldnames=fields,
    )


def visit_csv_fragment_rows(
    path: str | Path,
    *,
    fieldnames: Iterable[str],
    visitor: Callable[[dict[str, str]], object],
) -> CsvArtifactReceipt:
    """Visit validated Dict rows without retaining the fragment in memory."""

    source = Path(path)
    fields = _fieldnames(fieldnames)
    if not callable(visitor):
        raise CsvFragmentError("CSV fragment visitor must be callable")

    def emit(row: list[str]) -> None:
        visitor(dict(zip(fields, row)))

    return _consume_fragment(source, fields, emit)


def inspect_csv_fragment(
    path: str | Path,
    *,
    fieldnames: Iterable[str],
) -> CsvArtifactReceipt:
    """Strictly validate a fragment without mutating it."""

    source = Path(path)
    fields = _fieldnames(fieldnames)
    return _consume_fragment(source, fields, lambda _row: None)


def merge_csv_fragments(
    fragment_paths: Iterable[str | Path],
    output_path: str | Path,
    *,
    fieldnames: Iterable[str],
) -> CsvArtifactReceipt:
    """Merge fragments in caller-supplied canonical order and publish one CSV."""

    fields = _fieldnames(fieldnames)
    fragments = tuple(Path(path) for path in fragment_paths)
    identities = [os.path.abspath(os.fspath(path)) for path in fragments]
    if len(identities) != len(set(identities)):
        raise CsvFragmentError("canonical fragment order contains duplicate paths")
    target = Path(output_path)
    target_identity = os.path.abspath(os.fspath(target))
    if target_identity in identities:
        raise CsvFragmentError("merged output cannot overwrite an input fragment")
    _prepare_atomic_target(target)
    temporary: Path | None = None
    row_count = 0
    header_written = False
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            errors="strict",
            newline="",
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=target.parent,
            delete=False,
        ) as output:
            temporary = Path(output.name)
            writer = csv.writer(output, dialect="excel")

            def emit(row):
                nonlocal header_written, row_count
                if not header_written:
                    writer.writerow(fields)
                    header_written = True
                writer.writerow(row)
                row_count += 1

            for fragment in fragments:
                _consume_fragment(fragment, fields, emit)
            output.flush()
            os.fsync(output.fileno())
        receipt = _publish(
            temporary,
            target,
            row_count=row_count,
            fieldnames=fields,
        )
        temporary = None
        return receipt
    except CsvFragmentError:
        raise
    except (csv.Error, OSError, TypeError, UnicodeError, ValueError) as error:
        raise CsvFragmentError(f"failed to merge CSV fragments into: {target}") from error
    finally:
        if temporary is not None and _lexists(temporary):
            temporary.unlink()
