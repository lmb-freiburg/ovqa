"""
Same logic as md-toc package except skipping h1 and forcing unix newlines.
"""

from pathlib import Path

import md_toc


def main():
    file = Path("README.md")
    toc = md_toc.build_toc(file.as_posix(), keep_header_levels=3)
    content = file.read_text().replace("\r\n", "\n")
    marker = "<!--TOC-->"
    marker_splits = content.split(marker)
    if len(marker_splits) == 2:
        before, after = marker_splits
    elif len(marker_splits) == 3:
        before, _old_toc, after = marker_splits
    else:
        raise ValueError(
            f"Marker {marker} found {len(marker_splits) - 1} times in {file}, expected 1 or 2."
        )
    toc = toc.replace("\r\n", "\n")
    new_toc = []
    for line in toc.splitlines():
        if line.startswith("- "):
            continue
        elif line.startswith("  "):
            line = line[2:]
        new_toc.append(line)
    toc = "\n".join(new_toc)
    print(toc)
    file.write_text(
        before + marker + "\n\n" + toc + "\n\n" + marker + after, encoding="utf-8", newline="\n"
    )


if __name__ == "__main__":
    main()
