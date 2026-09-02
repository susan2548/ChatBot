import re


_RETRIEVAL_STOP_TERMS = {
    "อะไร", "อย่างไร", "ยังไง", "ภาษา", "เอาไว้", "ทำอะไร", "ขอ", "ช่วย",
    "what", "how", "does", "the", "and", "with", "from", "คือ", "ใช้",
    "กับ", "ต่างกัน", "แตกต่างกัน",
}


_COMPARISON_MARKERS = re.compile(
    r"ต่าง|แตกต่าง|เปรียบเทียบ|เทียบ(?:กับ)?|\bvs\.?\b|\bversus\b|\bdifference\b",
    re.IGNORECASE,
)

_COMPARISON_CONCEPTS = {
    "while": {
        "query": re.compile(r"(?<![A-Za-z0-9_])while(?:\s*loop)?(?![A-Za-z0-9_])", re.IGNORECASE),
        "content": re.compile(
            r"\bwhile\s+loop\b|\bwhile\s*\(|ลูป\s*while|while\s*ลูป",
            re.IGNORECASE,
        ),
    },
    "for": {
        "query": re.compile(r"(?<![A-Za-z0-9_])for(?:\s*loop)?(?![A-Za-z0-9_])", re.IGNORECASE),
        "content": re.compile(
            r"\bfor\s+loop\b|\bfor\s*\(|ลูป\s*for|for\s*ลูป",
            re.IGNORECASE,
        ),
    },
    "do_while": {
        "query": re.compile(r"(?<![A-Za-z0-9_])do\s*[-/]?\s*while(?![A-Za-z0-9_])", re.IGNORECASE),
        "content": re.compile(
            r"\bdo\s*[-/]?\s*while\b|ลูป\s*do\s*[-/]?\s*while",
            re.IGNORECASE,
        ),
    },
}


def normalize_retrieval_query(query):
    normalized = re.sub(r"\s+", " ", str(query or "").strip())
    replacements = {
        r"(?<![A-Za-z0-9_])if\s*else(?![A-Za-z0-9_])": " if else conditional statement เงื่อนไข ",
        r"(?<![A-Za-z0-9_])for\s*loop(?![A-Za-z0-9_])": " for loop ลูป for วนซ้ำ ",
        r"(?<![A-Za-z0-9_])while\s*loop(?![A-Za-z0-9_])": " while loop ลูป while วนซ้ำ ",
        r"\bswitch\s*case\b": "switch case เงื่อนไข",
        r"ประกาศ\s*ตัวแปร|สร้าง\s*ตัวแปร": "ประกาศตัวแปร ตัวแปร การประกาศ variable variables declaration declare data type ชนิดข้อมูล ชื่อตัวแปร initialization",
        r"กำหนด\s*ค่า\s*ตัวแปร": "กำหนดค่าตัวแปร variable assignment initialization",
        r"printf\s*คืออะไร|ใช้\s*printf": "printf stdio.h output format string แสดงผล",
        r"รับ\s*ค่า.*แป้นพิมพ์|scanf\s*คืออะไร": "scanf stdio.h input format specifier รับข้อมูล",
    }
    for pattern, replacement in replacements.items():
        normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", normalized).strip()


def comparison_retrieval_anchors(query):
    """Return concepts that must both be represented for a comparison question."""
    text = str(query or "")
    if not _COMPARISON_MARKERS.search(text):
        return ()
    # Detect do-while first and remove it before detecting a standalone while.
    masked = text
    anchors = []
    do_while = _COMPARISON_CONCEPTS["do_while"]["query"]
    if do_while.search(masked):
        anchors.append("do_while")
        masked = do_while.sub(" ", masked)
    for anchor in ("while", "for"):
        if _COMPARISON_CONCEPTS[anchor]["query"].search(masked):
            anchors.append(anchor)
    return tuple(anchors) if len(anchors) >= 2 else ()


def retrieval_anchor_strength(content, anchor):
    """Return 2 for a heading match, 1 for a body match, otherwise 0."""
    concept = _COMPARISON_CONCEPTS.get(anchor)
    if concept is None:
        return 0
    text = str(content or "")
    topic_text = text.split("|", 1)[0]
    if concept["content"].search(topic_text):
        return 2
    return 1 if concept["content"].search(text) else 0


def extract_retrieval_terms(query):
    """Extract useful English/code tokens and Thai phrases without external tokenizers."""
    raw_terms = re.findall(r"[A-Za-z_][A-Za-z0-9_+#.-]*|[ก-๙]{2,}", query.lower())
    return {
        term for term in raw_terms
        if term not in _RETRIEVAL_STOP_TERMS and (len(term) >= 2 or term == "c")
    }


def lexical_relevance(content, terms):
    """Return a bounded boost and term coverage, favouring matches in the topic label."""
    lowered = content.lower()
    topic_text = lowered.split("|", 1)[0]
    useful_terms = {term for term in terms if term != "c"}
    if not useful_terms:
        return 0.0, 0.0
    matched = {term for term in useful_terms if term in lowered}
    if not matched:
        return 0.0, 0.0
    coverage = len(matched) / len(useful_terms)
    topic_matches = sum(1 for term in matched if term in topic_text)
    boost = min(0.22, coverage * 0.14 + topic_matches * 0.04)
    return boost, coverage


def retrieval_topic_key(content):
    """Group chunks from the same labelled topic so one topic cannot crowd out all evidence."""
    first_section = re.sub(r"\s+", " ", content.lower()).split("|", 1)[0].strip()
    return first_section[:180]
