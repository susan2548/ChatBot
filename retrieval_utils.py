import re


_RETRIEVAL_STOP_TERMS = {
    "อะไร", "อย่างไร", "ยังไง", "ภาษา", "เอาไว้", "ทำอะไร", "ขอ", "ช่วย",
    "what", "how", "does", "the", "and", "with", "from", "คือ", "ใช้",
}


def normalize_retrieval_query(query):
    normalized = re.sub(r"\s+", " ", str(query or "").strip())
    replacements = {
        r"\bif\s*else\b": "if else conditional statement เงื่อนไข",
        r"\bfor\s*loop\b": "for loop วนซ้ำ",
        r"\bwhile\s*loop\b": "while loop วนซ้ำ",
        r"\bswitch\s*case\b": "switch case เงื่อนไข",
        r"ประกาศ\s*ตัวแปร|สร้าง\s*ตัวแปร": "ประกาศตัวแปร ตัวแปร การประกาศ variable variables declaration declare data type ชนิดข้อมูล ชื่อตัวแปร initialization",
        r"กำหนด\s*ค่า\s*ตัวแปร": "กำหนดค่าตัวแปร variable assignment initialization",
        r"printf\s*คืออะไร|ใช้\s*printf": "printf stdio.h output format string แสดงผล",
        r"รับ\s*ค่า.*แป้นพิมพ์|scanf\s*คืออะไร": "scanf stdio.h input format specifier รับข้อมูล",
    }
    for pattern, replacement in replacements.items():
        normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)
    return normalized


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
