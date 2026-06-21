#!/usr/bin/env python3
"""Local similarity audit for the thesis.

This is not a replacement for JSA. It is a conservative local check that looks
for exact shared word n-grams between thesis LaTeX files and source materials.
Use it to find passages that may need rewriting or clearer citation.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import re
import shutil
import subprocess
from collections import Counter, defaultdict
from pathlib import Path


TEXT_EXTENSIONS = {".txt", ".md", ".tex"}
DEFAULT_THESIS_GLOBS = ["chapters/*.tex"]
DEFAULT_SOURCE_GLOBS = ["papers/*.txt", "papers/*.md", "papers/*.tex", "papers/*.pdf"]


def strip_latex(text: str) -> str:
    text = re.sub(r"(?<!\\)%.*", " ", text)
    text = re.sub(r"\\(cite|citep|citet|ref|label|pageref)\*?(\[[^\]]*\])?\{[^{}]*\}", " ", text)
    text = re.sub(r"\\(chapter|section|subsection|subsubsection)\*?(\[[^\]]*\])?\{([^{}]*)\}", r" \3 ", text)
    text = re.sub(r"\\(texttt|emph|textit|textbf)\{([^{}]*)\}", r" \2 ", text)
    text = re.sub(r"\\[A-Za-z]+\*?(\[[^\]]*\])?(\{[^{}]*\})?", " ", text)
    text = re.sub(r"[{}$^_~&#]", " ", text)
    return text


def tokenize(text: str) -> list[str]:
    return re.findall(r"[^\W_]+(?:[-'][^\W_]+)?", text.lower(), flags=re.UNICODE)


def read_text_file(path: Path) -> str:
    for encoding in ("utf-8", "utf-8-sig", "cp1250", "latin-1"):
        try:
            return path.read_text(encoding=encoding, errors="strict")
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="ignore")


def extract_pdf_text(path: Path) -> tuple[str | None, str | None]:
    try:
        import pypdf  # type: ignore

        reader = pypdf.PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages), "pypdf"
    except Exception:
        pass

    try:
        import PyPDF2  # type: ignore

        reader = PyPDF2.PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages), "PyPDF2"
    except Exception:
        pass

    pdftotext = shutil.which("pdftotext")
    if pdftotext:
        try:
            result = subprocess.run(
                [pdftotext, "-layout", str(path), "-"],
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="ignore",
            )
            return result.stdout, "pdftotext"
        except Exception:
            pass

    return None, None


def file_tokens(path: Path, *, is_latex: bool) -> tuple[list[tuple[str, int]], str | None]:
    if path.suffix.lower() == ".pdf":
        text, method = extract_pdf_text(path)
        if text is None:
            return [], "pdf text extraction unavailable"
        lines = text.splitlines()
        source_note = f"pdf extracted with {method}"
    else:
        lines = read_text_file(path).splitlines()
        source_note = None

    out: list[tuple[str, int]] = []
    for line_no, line in enumerate(lines, 1):
        clean = strip_latex(line) if is_latex else line
        out.extend((token, line_no) for token in tokenize(clean))
    return out, source_note


def expand_patterns(root: Path, patterns: list[str]) -> list[Path]:
    files: list[Path] = []
    for pattern in patterns:
        files.extend(path for path in root.glob(pattern) if path.is_file())
    return sorted(set(files))


def build_source_index(
    source_files: list[Path], ngram: int, max_sources_per_phrase: int
) -> tuple[dict[str, list[tuple[Path, int]]], list[str], Counter[str]]:
    index: dict[str, list[tuple[Path, int]]] = defaultdict(list)
    notes: list[str] = []
    counts: Counter[str] = Counter()

    for path in source_files:
        tokens, note = file_tokens(path, is_latex=path.suffix.lower() == ".tex")
        if note:
            notes.append(f"- {path}: {note}")
        if not tokens:
            continue
        words = [token for token, _line in tokens]
        lines = [line for _token, line in tokens]
        for i in range(0, max(0, len(words) - ngram + 1)):
            phrase = " ".join(words[i : i + ngram])
            counts[str(path)] += 1
            if len(index[phrase]) < max_sources_per_phrase:
                index[phrase].append((path, lines[i]))
    return index, notes, counts


def find_matches(
    thesis_files: list[Path], source_index: dict[str, list[tuple[Path, int]]], ngram: int, limit: int
) -> list[dict[str, object]]:
    seen: set[tuple[str, int, str]] = set()
    matches: list[dict[str, object]] = []

    for path in thesis_files:
        tokens, _note = file_tokens(path, is_latex=True)
        words = [token for token, _line in tokens]
        lines = [line for _token, line in tokens]
        for i in range(0, max(0, len(words) - ngram + 1)):
            phrase = " ".join(words[i : i + ngram])
            if phrase not in source_index:
                continue
            key = (str(path), lines[i], phrase)
            if key in seen:
                continue
            seen.add(key)
            matches.append(
                {
                    "thesis_file": path,
                    "thesis_line": lines[i],
                    "phrase": phrase,
                    "sources": source_index[phrase],
                }
            )
            if len(matches) >= limit:
                return matches
    return matches


def write_report(
    output: Path,
    thesis_files: list[Path],
    source_files: list[Path],
    matches: list[dict[str, object]],
    notes: list[str],
    ngram: int,
) -> None:
    lines: list[str] = []
    lines.append("# Thesis Similarity Audit")
    lines.append("")
    lines.append(f"Generated: {_dt.datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"N-gram size: {ngram} words")
    lines.append("")
    lines.append("This is a local exact n-gram check, not a JSA-equivalent result.")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append(f"Thesis files checked: {len(thesis_files)}")
    for path in thesis_files:
        lines.append(f"- `{path}`")
    lines.append("")
    lines.append(f"Source files checked: {len(source_files)}")
    for path in source_files:
        lines.append(f"- `{path}`")
    lines.append("")
    if notes:
        lines.append("## Extraction Notes")
        lines.append("")
        lines.extend(notes)
        lines.append("")
    lines.append("## Exact Overlap Findings")
    lines.append("")
    if not matches:
        lines.append("No exact shared n-grams were found at the configured threshold.")
    else:
        lines.append(f"Found {len(matches)} exact shared n-gram occurrence(s) shown below.")
        lines.append("")
        for idx, match in enumerate(matches, 1):
            thesis_file = match["thesis_file"]
            thesis_line = match["thesis_line"]
            phrase = match["phrase"]
            lines.append(f"### Match {idx}")
            lines.append("")
            lines.append(f"Thesis: `{thesis_file}` line {thesis_line}")
            lines.append(f"Phrase: `{phrase}`")
            lines.append("Sources:")
            for src_path, src_line in match["sources"]:  # type: ignore[index]
                lines.append(f"- `{src_path}` line {src_line}")
            lines.append("")
    lines.append("## Recommended Use")
    lines.append("")
    lines.append("- Inspect each match manually; technical phrases and titles may be harmless.")
    lines.append("- Rewrite passages that reuse source wording too closely.")
    lines.append("- Add or verify citations for literature-derived claims.")
    lines.append("- Run again with a smaller `--ngram` value for a stricter check if needed.")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="Workspace root")
    parser.add_argument("--thesis-glob", action="append", default=None, help="Glob for thesis files")
    parser.add_argument("--source-glob", action="append", default=None, help="Glob for source files")
    parser.add_argument("--ngram", type=int, default=10, help="Exact word n-gram length")
    parser.add_argument("--limit", type=int, default=200, help="Maximum matches to report")
    parser.add_argument(
        "--output",
        default="master_thesis/similarity_audit.md",
        help="Markdown report path",
    )
    parser.add_argument("--max-sources-per-phrase", type=int, default=5)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    thesis_patterns = args.thesis_glob or DEFAULT_THESIS_GLOBS
    source_patterns = args.source_glob or DEFAULT_SOURCE_GLOBS
    thesis_files = expand_patterns(root / "master_thesis", thesis_patterns)
    source_files = expand_patterns(root, source_patterns)

    source_index, notes, _counts = build_source_index(
        source_files, args.ngram, args.max_sources_per_phrase
    )
    matches = find_matches(thesis_files, source_index, args.ngram, args.limit)
    output = (root / args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    write_report(output, thesis_files, source_files, matches, notes, args.ngram)
    print(f"Wrote {output}")
    print(f"Matches: {len(matches)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
