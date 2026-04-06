"""Post-process Mermaid SVG files for Word compatibility.

Word may fail to render <foreignObject> content in SVG files. This tool replaces
foreignObject label payloads with native SVG <text>/<tspan> elements.
"""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"
FONT_FAMILY = "Times New Roman, 宋体, SimSun"
FONT_SIZE = "12pt"
TEXT_FILL = "#333"
DEFAULT_SINGLE_LINE_DY = "0.35em"
TERMINATOR_DY = "0.45em"
TERMINATOR_LABELS = {"开始", "结束"}

ET.register_namespace("", SVG_NS)
ET.register_namespace("xlink", XLINK_NS)


def _tag_local_name(tag: str) -> str:
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def _extract_text(node: ET.Element) -> str:
    parts: list[str] = []
    for elem in node.iter():
        local = _tag_local_name(elem.tag)
        if elem.text:
            parts.append(elem.text)
        if local == "br":
            parts.append("\n")
        if elem.tail:
            parts.append(elem.tail)
    text = "".join(parts).replace("\r", "")
    lines = [line.strip() for line in text.split("\n")]
    lines = [" ".join(line.split()) for line in lines if line.strip()]
    return "\n".join(lines)


def _to_float(value: str | None, default: float) -> float:
    if not value:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _apply_text_style(node: ET.Element) -> None:
    """Force explicit text style so Word does not rely on CSS inheritance."""
    node.set("font-family", FONT_FAMILY)
    node.set("font-size", FONT_SIZE)
    if not node.get("fill"):
        node.set("fill", TEXT_FILL)
    if node.get("dominant-baseline"):
        node.attrib.pop("dominant-baseline", None)


def convert_svg(path: Path) -> int:
    tree = ET.parse(path)
    root = tree.getroot()
    parent_map = {child: parent for parent in root.iter() for child in parent}

    replaced = 0
    for elem in list(root.iter()):
        if _tag_local_name(elem.tag) != "foreignObject":
            continue

        parent = parent_map.get(elem)
        if parent is None:
            continue

        text_content = _extract_text(elem)
        idx = list(parent).index(elem)
        parent.remove(elem)

        if not text_content:
            replaced += 1
            continue

        width = _to_float(elem.get("width"), 0.0)
        height = _to_float(elem.get("height"), 0.0)
        text_el = ET.Element(f"{{{SVG_NS}}}text")
        text_el.set("x", f"{width / 2:.3f}")
        text_el.set("y", f"{height / 2:.3f}")
        text_el.set("text-anchor", "middle")
        _apply_text_style(text_el)

        lines = text_content.split("\n")
        if len(lines) == 1:
            # Word does not reliably support dominant-baseline for SVG text.
            single_line = lines[0].strip()
            text_el.set("dy", TERMINATOR_DY if single_line in TERMINATOR_LABELS else DEFAULT_SINGLE_LINE_DY)
            text_el.text = lines[0]
        else:
            y_offset = (-0.6 * (len(lines) - 1)) + 0.35
            for i, line in enumerate(lines):
                tspan = ET.SubElement(text_el, f"{{{SVG_NS}}}tspan")
                tspan.set("x", f"{width / 2:.3f}")
                tspan.set("dy", "1.2em" if i > 0 else f"{y_offset:.2f}em")
                _apply_text_style(tspan)
                tspan.text = line

        parent.insert(idx, text_el)
        replaced += 1

    # Enforce uniform font/size on existing Mermaid text nodes too.
    for elem in root.iter():
        local = _tag_local_name(elem.tag)
        if local in {"text", "tspan"}:
            _apply_text_style(elem)
        if local == "text":
            has_tspan_children = any(_tag_local_name(child.tag) == "tspan" for child in list(elem))
            text_value = (elem.text or "").strip()
            if (not has_tspan_children) and text_value in TERMINATOR_LABELS:
                elem.set("dy", TERMINATOR_DY)
                continue
            # For single-line text, use baseline shift that Word handles reliably.
            if (not has_tspan_children) and text_value and (not elem.get("dy")):
                elem.set("dy", DEFAULT_SINGLE_LINE_DY)

    tree.write(path, encoding="utf-8", xml_declaration=True)
    return replaced


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Mermaid SVG foreignObject labels to SVG text.")
    parser.add_argument("files", nargs="+", help="SVG files to process")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    results: list[tuple[str, int]] = []
    for file_arg in args.files:
        path = Path(file_arg)
        converted = convert_svg(path)
        results.append((str(path), converted))
    print({"converted": results})
