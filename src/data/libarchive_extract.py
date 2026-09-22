"""RAR/ZIP extraction through libarchive via ctypes.

Avoids depending on an external ``unrar`` binary or on network installs.
Falls back with a clear error if libarchive is not present on the system.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import List, Tuple

ARCHIVE_OK, ARCHIVE_EOF = 0, 1
_CANDIDATES = ["libarchive.so.13", "libarchive.so", "libarchive.13.dylib", "libarchive.dylib"]


def _load_lib():
    last = None
    for name in _CANDIDATES:
        try:
            return ctypes.CDLL(name)
        except OSError as exc:  # pragma: no cover
            last = exc
    raise OSError(f"libarchive not found (tried {_CANDIDATES}): {last}")


def _bind(la):
    la.archive_read_new.restype = ctypes.c_void_p
    la.archive_read_support_filter_all.argtypes = [ctypes.c_void_p]
    la.archive_read_support_format_all.argtypes = [ctypes.c_void_p]
    la.archive_read_open_filename.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t]
    la.archive_read_next_header.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
    la.archive_entry_pathname.argtypes = [ctypes.c_void_p]
    la.archive_entry_pathname.restype = ctypes.c_char_p
    la.archive_entry_pathname_utf8.argtypes = [ctypes.c_void_p]
    la.archive_entry_pathname_utf8.restype = ctypes.c_char_p
    la.archive_entry_size.argtypes = [ctypes.c_void_p]
    la.archive_entry_size.restype = ctypes.c_longlong
    la.archive_entry_filetype.argtypes = [ctypes.c_void_p]
    la.archive_entry_filetype.restype = ctypes.c_uint
    la.archive_read_data.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    la.archive_read_data.restype = ctypes.c_ssize_t
    la.archive_error_string.argtypes = [ctypes.c_void_p]
    la.archive_error_string.restype = ctypes.c_char_p
    la.archive_read_free.argtypes = [ctypes.c_void_p]
    return la


def list_archive(src: Path) -> List[Tuple[str, int]]:
    return _walk(src, None)


def extract_with_libarchive(src: Path, dest: Path) -> List[Tuple[str, int]]:
    dest.mkdir(parents=True, exist_ok=True)
    return _walk(src, dest)


def _walk(src: Path, dest) -> List[Tuple[str, int]]:
    la = _bind(_load_lib())
    a = la.archive_read_new()
    la.archive_read_support_filter_all(a)
    la.archive_read_support_format_all(a)
    if la.archive_read_open_filename(a, str(src).encode(), 1 << 20) != ARCHIVE_OK:
        raise RuntimeError(la.archive_error_string(a))
    entry = ctypes.c_void_p()
    buf = ctypes.create_string_buffer(1 << 20)
    out: List[Tuple[str, int]] = []
    try:
        while True:
            rc = la.archive_read_next_header(a, ctypes.byref(entry))
            if rc == ARCHIVE_EOF:
                break
            if rc != ARCHIVE_OK:
                raise RuntimeError(la.archive_error_string(a))
            name_b = la.archive_entry_pathname_utf8(entry) or la.archive_entry_pathname(entry)
            name = name_b.decode("utf-8", "replace")
            size = int(la.archive_entry_size(entry))
            ftype = la.archive_entry_filetype(entry)
            out.append((name, size))
            if dest is None:
                continue
            safe = os.path.normpath(name.replace("\\", "/")).lstrip("./")
            target = Path(dest) / safe
            if ftype == 0o040000:
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with open(target, "wb") as fh:
                while True:
                    n = la.archive_read_data(a, buf, len(buf))
                    if n == 0:
                        break
                    if n < 0:
                        raise RuntimeError(la.archive_error_string(a))
                    fh.write(buf.raw[:n])
    finally:
        la.archive_read_free(a)
    return out
