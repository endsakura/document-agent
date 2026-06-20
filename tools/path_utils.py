"""文件路径解析 — 兼容相对路径、MCP 子进程不同 cwd。"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def resolve_file_path(path: str) -> Path:
    """
    解析为存在的绝对路径。
    依次尝试：原路径、cwd 相对、项目根相对。
    """
    if not path or not str(path).strip():
        raise FileNotFoundError("路径为空")

    raw = Path(path.strip())
    candidates = []

    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.extend([
            Path.cwd() / raw,
            PROJECT_ROOT / raw,
            raw,
        ])

    seen = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        if resolved.is_file():
            return resolved

    raise FileNotFoundError(f"文件不存在: {path}（cwd={Path.cwd()}, root={PROJECT_ROOT}）")
