from __future__ import annotations

import re
import xml.etree.ElementTree as ET

from bs4 import BeautifulSoup


def parse_summary_html(
    html_text: str,
) -> list[tuple[str, list[str]]]:
    """
    Parse the HTML embedded inside a MedlinePlus <full-summary>.

    Returns:
        [
            ("full_summary", [...blocks...]),
            ("What are the symptoms?", [...blocks...]),
            ...
        ]

    Only headings actually present in the source are used.
    """

    soup = BeautifulSoup(
        html_text,
        "html.parser",
    )

    sections: list[tuple[str, list[str]]] = []

    current_section = "full_summary"
    current_blocks: list[str] = []

    def flush_section() -> None:
        nonlocal current_blocks

        if current_blocks:
            sections.append(
                (
                    current_section,
                    current_blocks,
                )
            )

            current_blocks = []

    for node in soup.find_all(
        ["h2", "h3", "h4", "p", "li"]
    ):

        # --------------------------
        # Real section heading
        # --------------------------

        if node.name in {"h2", "h3", "h4"}:

            heading = clean_text(
                node.get_text(
                    " ",
                    strip=True,
                )
            )

            if not heading:
                continue

            flush_section()

            current_section = heading

            continue

        # --------------------------
        # Avoid double-counting
        # nested list items
        # --------------------------

        if (
            node.name == "li"
            and node.find_parent("li") is not None
        ):
            continue

        text = clean_text(
            node.get_text(
                " ",
                strip=True,
            )
        )

        if not text:
            continue

        if node.name == "li":
            text = f"- {text}"

        current_blocks.append(text)

    flush_section()

    # Defensive fallback
    if not sections:

        text = clean_text(
            soup.get_text(
                " ",
                strip=True,
            )
        )

        if text:
            sections.append(
                (
                    "full_summary",
                    [text],
                )
            )

    return sections


_WHITESPACE_RE = re.compile(r"\s+")


def clean_text(text: str | None) -> str:
    """
    Collapse repeated whitespace without otherwise modifying content.
    """
    if not text:
        return ""

    return _WHITESPACE_RE.sub(" ", text).strip()


def element_text(element: ET.Element | None) -> str:
    """
    Extract all human-readable text recursively from an XML element.

    This removes XML markup while preserving the actual words inside
    nested tags such as <a>, <strong>, etc.
    """
    if element is None:
        return ""

    return clean_text(" ".join(element.itertext()))


def local_name(tag: str) -> str:
    """
    Strip an XML namespace if one exists.

    Example:
        {namespace}term-group -> term-group
    """
    return tag.rsplit("}", 1)[-1]


def child_text(parent: ET.Element, child_name: str) -> str:
    """
    Return the cleaned text of a direct child by local tag name.
    """
    for child in parent:
        if local_name(child.tag) == child_name:
            return element_text(child)

    return ""


def child_texts(parent: ET.Element, child_name: str) -> list[str]:
    """
    Return all non-empty direct-child values with a given tag.
    """
    values: list[str] = []

    for child in parent:
        if local_name(child.tag) == child_name:
            value = element_text(child)

            if value:
                values.append(value)

    return values


def extract_summary_blocks(summary: ET.Element | None) -> list[str]:
    """
    Convert a MedlinePlus <full-summary> into paragraph-aware blocks.

    The full-summary contains HTML-like markup such as:
        <p>...</p>
        <ul><li>...</li></ul>

    We preserve those logical boundaries rather than flattening the
    entire summary into one long string.
    """
    if summary is None:
        return []

    blocks: list[str] = []

    # Occasionally an XML node may contain text before its first child.
    leading_text = clean_text(summary.text)
    if leading_text:
        blocks.append(leading_text)

    for child in summary:

        tag = local_name(child.tag)

        if tag in {"p", "div"}:
            text = element_text(child)

            if text:
                blocks.append(text)

        elif tag in {"ul", "ol"}:
            list_items: list[str] = []

            for item in child:
                if local_name(item.tag) != "li":
                    continue

                text = element_text(item)

                if text:
                    list_items.append(text)

            if list_items:
                block = "\n".join(
                    f"- {item}"
                    for item in list_items
                )
                blocks.append(block)

        elif tag == "li":
            text = element_text(child)

            if text:
                blocks.append(f"- {text}")

        else:
            # Be conservative: retain readable text even if MedlinePlus
            # introduces another markup tag in the future.
            text = element_text(child)

            if text:
                blocks.append(text)

        tail = clean_text(child.tail)

        if tail:
            blocks.append(tail)

    # Fallback in case the summary contains no child markup.
    if not blocks:
        text = element_text(summary)

        if text:
            blocks.append(text)

    return blocks

def clean_glossary_text(text: str | None) -> str:
    """
    MedlinePlus Definitions XML currently contains a stray
    literal '>' at the beginning of term/definition values.
    """
    value = clean_text(text)

    if value.startswith(">"):
        value = value[1:].lstrip()

    return value