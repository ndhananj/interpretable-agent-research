from __future__ import annotations

from pathlib import Path


def safe_path(root: Path, rel_path: str) -> Path:
    target = (root / rel_path).resolve()
    root_resolved = root.resolve()
    if root_resolved not in target.parents and target != root_resolved:
        raise ValueError(f"Path escapes work dir: {rel_path}")
    return target


def normalize_edit_path(root: Path, path: str) -> str:
    raw = Path(path)
    root_resolved = root.resolve()
    if raw.is_absolute():
        raw_resolved = raw.resolve()
        if root_resolved not in raw_resolved.parents and raw_resolved != root_resolved:
            raise ValueError(f"Path escapes work dir: {path}")
        return raw_resolved.relative_to(root_resolved).as_posix()

    raw_posix = raw.as_posix()
    for prefix in work_dir_prefixes(root):
        prefix_posix = prefix.as_posix().rstrip("/")
        if raw_posix == prefix_posix:
            return "."
        if raw_posix.startswith(f"{prefix_posix}/"):
            return raw_posix[len(prefix_posix) + 1 :]
    return raw_posix


def work_dir_prefixes(root: Path) -> list[Path]:
    prefixes = [root, root.resolve()]
    try:
        prefixes.append(root.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        pass
    return prefixes
