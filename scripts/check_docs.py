"""Check every Markdown file against the documentation style rules.

The rules are mechanical, so they are checked mechanically rather than by
reading: forbidden characters, forbidden turns of phrase, emphasis outside
headings and tables, first person in technical text, and broken links.

Code fences and table rows are skipped where a rule would damage them.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".venv", "node_modules", "_context", ".git", ".pytest_cache"}
SKIP_PATHS = {Path("reports/before")}

FORBIDDEN_CHARS = {
    "—": "длинное тире",
    "–": "короткое тире",
    "→": "стрелка",
    "←": "стрелка",
    "⇒": "стрелка",
    "·": "символ-разделитель",
    "•": "символ-разделитель",
    "…": "многоточие как разделитель",
    "−": "минус-символ вместо дефиса",
    "↔": "стрелка",
    "«": "кавычка-ёлочка",
    "»": "кавычка-ёлочка",
    "“": "типографская кавычка",
    "”": "типографская кавычка",
    "‘": "типографская кавычка",
    "’": "типографская кавычка",
    "✓": "галочка",
    "✔": "галочка",
    "✗": "крестик",
    "✘": "крестик",
}
EMOJI = re.compile("[\U0001f000-\U0001faff\U00002600-\U000027bf\U0001f1e6-\U0001f1ff️⭐❗❓]")

FORBIDDEN_WORDS = [
    "ключев",
    "комплексн",
    "важно отметить",
    "стоит подчеркнуть",
    "в рамках",
    r"\bполностью\b",
    r"\bмаксимально\b",
    "при этом",
    "таким образом",
    r"\bсущественно\b",
    r"\bзначительно\b",
    r"\bзаметно\b",
    r"\bсознательно\b",
    r"\bпренебрежимо\b",
    "честный ответ",
    "красивый результат",
]
HISTORY = [
    r"\bбыло\b",
    r"\bстало\b",
    "доработк",
    r"\bранее\b",
    r"\bтеперь\b",
    r"\bраньше\b",
    "после внедрения",
    "прежней версии",
    "предыдущей версии",
    "исходн\\w* прототип",
    r"\bпомогло\b",
    "не давшие эффекта",
    "не бесплатно",
    "что изменилось",
]
COLLOQUIAL = [
    r"\bсмотрит\b",
    r"\bвидит\b",
    r"\bумеет\b",
    r"\bнаучилась\b",
    r"\bнаучился\b",
    r"\bвидят\b",
    r"\bсмотрят\b",
]
FIRST_PERSON = [r"\bмы\b", r"\bнам\b", r"\bнаш[аеиоуы]?[йемх]?\b", r"\bя\b", r"\bмне\b", r"\bнас\b"]
PHRASES = [
    (r"не\s+[^,.;]{1,40},\s+а\s+", 'конструкция "не X, а Y"'),
    (r"это\s+не\s+просто\s+", 'конструкция "это не просто X"'),
    (r"именно\s+поэтому", 'оборот "именно поэтому"'),
]
QUESTION_HEADING = re.compile(r"^#{1,6}\s+.*\?\s*$")


EMPHASIS = re.compile(r"\*\*[^*]+\*\*")


def markdown_files() -> list[Path]:
    out = []
    for path in sorted(ROOT.rglob("*.md")):
        relative = path.relative_to(ROOT)
        if SKIP_DIRS & set(relative.parts):
            continue
        if any(skip in relative.parents for skip in SKIP_PATHS):
            continue
        out.append(path)
    return out


def check(path: Path) -> list[str]:
    problems = []
    in_fence = False
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        where = f"{path.relative_to(ROOT)}:{number}"
        for char, name in FORBIDDEN_CHARS.items():
            if char in line:
                problems.append(f"{where}: {name} ({char})")
        if EMOJI.search(line):
            problems.append(f"{where}: эмодзи")
        lowered = line.lower()
        for word in FORBIDDEN_WORDS:
            if re.search(word, lowered):
                problems.append(f'{where}: слово "{word}"')
        for pattern in HISTORY:
            if re.search(pattern, lowered):
                problems.append(f'{where}: история разработки "{pattern}"')
        for pattern in COLLOQUIAL:
            if re.search(pattern, lowered):
                problems.append(f"{where}: разговорный оборот {pattern}")
        for pattern in FIRST_PERSON:
            if re.search(pattern, lowered):
                problems.append(f"{where}: первое лицо {pattern}")
        for pattern, name in PHRASES:
            if re.search(pattern, lowered):
                problems.append(f"{where}: {name}")
        is_table = stripped.startswith("|")
        is_heading = stripped.startswith("#")
        if is_heading and QUESTION_HEADING.match(stripped):
            problems.append(f"{where}: заголовок-вопрос")
        if not is_table and not is_heading and EMPHASIS.search(line):
            problems.append(f"{where}: жирный шрифт вне заголовка и таблицы")
    return problems


def links(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    heads = {_anchor(h) for h in re.findall(r"^#{1,6}\s+(.*)$", text, re.M)}
    problems = []
    for label, link in re.findall(r"\[([^\]]+)\]\(([^)]+)\)", text):
        if link.startswith(("http://", "https://", "mailto:")):
            continue
        target, _, fragment = link.partition("#")
        if not target:
            if fragment and fragment not in heads:
                problems.append(f"{path.relative_to(ROOT)}: битый якорь #{fragment} ({label})")
            continue
        resolved = (path.parent / target).resolve()
        if not resolved.exists():
            problems.append(f"{path.relative_to(ROOT)}: нет файла {link} ({label})")
            continue
        if fragment and resolved.suffix == ".md":
            other = {
                _anchor(h)
                for h in re.findall(r"^#{1,6}\s+(.*)$", resolved.read_text(encoding="utf-8"), re.M)
            }
            if fragment not in other:
                problems.append(f"{path.relative_to(ROOT)}: битый якорь {link} ({label})")
    return problems


def _anchor(heading: str) -> str:
    return re.sub(r"[^\w\s-]", "", heading.strip().lower()).replace(" ", "-")


def main(quiet: bool = False) -> int:
    style, broken = [], []
    files = markdown_files()
    for path in files:
        style += check(path)
        broken += links(path)
    if not quiet:
        for line in style + broken:
            print(line)
    print(f"файлов: {len(files)}, нарушений стиля: {len(style)}, битых ссылок: {len(broken)}")
    return 1 if style or broken else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="проверка стилистики и ссылок во всех md")
    ap.add_argument("--quiet", action="store_true")
    sys.exit(main(**vars(ap.parse_args())))
