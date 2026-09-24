import re
import json

_ORDINALS = {
    "BİRİNCİ": 1, "İKİNCİ": 2, "ÜÇÜNCÜ": 3,
    "DÖRDÜNCÜ": 4, "BEŞİNCİ": 5, "ALTINCI": 6, "YEDİNCİ": 7,
}

_ROMAN = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}

_KISIM_RE = re.compile(r"(?<=\n)([A-ZÇŞİĞÜÖ]+)\s*KISIM\s*\n")
_BOLUM_RE = re.compile(r"(?<=\n)([A-ZÇŞİĞÜÖ]+)\s*BÖLÜM\s*\n")
_MADDE_RE = re.compile(r"(?<!Geçici )Madde\s+(\d+)\s*[–\-]")
_SECTION_RE = re.compile(r"(?<=\n)([IVXLCDM]+)\.\s+(.+?)(?=\s+Madde\s+\d+\s*[–\-]|\s*$|\n)")
_APPENDIX_MARKER = "İŞLENEMEYEN HÜKÜMLER"

# Roman numeral sections that are structural (not letter subsections)
# Letter subsections like "C.", "D." are excluded
_STRUCTURAL_SINGLE = {"I", "V", "X"}


def _ordinal_to_int(ordinal: str) -> int:
    return _ORDINALS.get(ordinal, 0)


def _roman_to_int(roman: str) -> int:
    result = 0
    prev = 0
    for c in reversed(roman):
        curr = _ROMAN.get(c, 0)
        if curr < prev:
            result -= curr
        else:
            result += curr
        prev = curr
    return result


def _is_structural_section(roman_text: str) -> bool:
    if len(roman_text) >= 2:
        return True
    return roman_text in _STRUCTURAL_SINGLE


def _normalize_label(raw: str) -> str:
    raw = raw.strip()
    if raw.endswith("KISIM") and not raw.endswith(" KISIM"):
        return raw[:-5] + " KISIM"
    if raw.endswith("BÖLÜM") and not raw.endswith(" BÖLÜM"):
        return raw[:-5] + " BÖLÜM"
    return raw


def _get_next_line(text: str, pos: int) -> str:
    end = text.find("\n", pos)
    if end < 0:
        end = len(text)
    return text[pos:end].strip()


def _clean_title(title: str) -> str:
    title = title.strip()
    title = re.sub(r"\s+", " ", title)
    title = re.sub(r"\d+$", "", title).strip()
    return title


def extract_toc(text: str) -> dict:
    appendix_pos = text.find(_APPENDIX_MARKER)
    if appendix_pos < 0:
        appendix_pos = len(text)

    text = text[:appendix_pos]

    headers = []

    for m in _KISIM_RE.finditer(text):
        ordinal_text = m.group(1)
        ordinal = _ordinal_to_int(ordinal_text)
        if ordinal == 0:
            continue
        label = _normalize_label(m.group(0).strip())
        title = _get_next_line(text, m.end())
        headers.append((m.start(), "part", {
            "ordinal": ordinal,
            "label": label,
            "title": _clean_title(title),
        }))

    for m in _BOLUM_RE.finditer(text):
        ordinal_text = m.group(1)
        ordinal = _ordinal_to_int(ordinal_text)
        if ordinal == 0:
            continue
        label = _normalize_label(m.group(0).strip())
        title = _get_next_line(text, m.end())
        headers.append((m.start(), "chapter", {
            "ordinal": ordinal,
            "label": label,
            "title": _clean_title(title),
        }))

    for m in _SECTION_RE.finditer(text):
        roman_text = m.group(1)
        if not _is_structural_section(roman_text):
            continue
        section_num = _roman_to_int(roman_text)
        raw_title = m.group(2).strip()
        cleaned = _clean_title(raw_title)
        if len(cleaned) > 80:
            continue
        if re.search(r'\b(Bölüm|Kısım|Madde)\b', cleaned):
            continue
        headers.append((m.start(), "section", {
            "ordinal": section_num,
            "label": f"{roman_text}. {cleaned}",
            "title": cleaned,
        }))

    for m in _MADDE_RE.finditer(text):
        art_num = int(m.group(1))
        headers.append((m.start(), "article", {
            "number": art_num,
        }))

    headers.sort(key=lambda x: x[0])

    parts = []
    stack = []

    for pos, htype, data in headers:
        if htype == "part":
            data["type"] = "part"
            data["article_start"] = None
            data["article_end"] = None
            data["children"] = []
            parts.append(data)
            stack = [data]

        elif htype == "chapter" and stack:
            data["type"] = "chapter"
            data["article_start"] = None
            data["article_end"] = None
            data["children"] = []
            stack[0]["children"].append(data)
            stack = [stack[0], data]

        elif htype == "section" and stack:
            data["type"] = "section"
            data["article_start"] = None
            data["article_end"] = None
            parent = stack[-1]
            existing = [c for c in parent.get("children", [])
                        if c.get("type") == "section" and c.get("ordinal") == data["ordinal"]]
            if existing:
                continue
            while len(stack) > 1 and stack[-1].get("type") == "section":
                stack.pop()
            parent = stack[-1]
            parent.setdefault("children", []).append(data)
            stack.append(data)

        elif htype == "article" and stack:
            art_num = data["number"]
            for node in reversed(stack):
                if node.get("article_start") is None or art_num < node["article_start"]:
                    node["article_start"] = art_num
                if node.get("article_end") is None or art_num > node["article_end"]:
                    node["article_end"] = art_num

    return {"parts": parts}


def build_article_path_lookup(toc: dict) -> dict:
    lookup = {}
    for part in toc["parts"]:
        _walk_part(part, lookup)
    return lookup


def _walk_part(node: dict, lookup: dict, path: dict = None):
    if path is None:
        path = {}

    node_type = node.get("type")

    if node_type == "part":
        path = {"part": node["label"], "chapter": None, "section": None}
    elif node_type == "chapter":
        path = dict(path)
        path["chapter"] = node["label"]
        path["section"] = None
    elif node_type == "section":
        path = dict(path)
        path["section"] = node["label"]

    for child in node.get("children", []):
        _walk_part(child, lookup, dict(path))

    if node.get("article_start") is not None and node.get("article_end") is not None:
        for n in range(node["article_start"], node["article_end"] + 1):
            if n not in lookup:
                lookup[n] = dict(path)

    return lookup
