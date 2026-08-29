import unittest

try:
    from retrieval_utils import (
        extract_retrieval_terms,
        lexical_relevance,
        normalize_retrieval_query,
        retrieval_topic_key,
    )
except ModuleNotFoundError as exc:
    extract_retrieval_terms = None
    MISSING_DEPENDENCY = str(exc)
else:
    MISSING_DEPENDENCY = ""


@unittest.skipIf(
    extract_retrieval_terms is None,
    f"optional deployment dependency missing: {MISSING_DEPENDENCY}",
)
class RetrievalHelperTests(unittest.TestCase):
    def test_thai_variable_query_expands_to_declaration_terms(self):
        normalized = normalize_retrieval_query("ประกาศตัวแปร C อย่างไร")
        terms = extract_retrieval_terms(normalized)
        self.assertIn("ประกาศตัวแปร", terms)
        self.assertIn("variable", terms)
        self.assertIn("declaration", terms)

    def test_printf_query_expands_to_header_and_output_terms(self):
        normalized = normalize_retrieval_query("printfคืออะไรเอาไว้ทำอะไร")
        terms = extract_retrieval_terms(normalized)
        self.assertIn("printf", terms)
        self.assertIn("stdio.h", terms)
        self.assertIn("output", terms)

    def test_topic_match_outranks_unrelated_content(self):
        terms = extract_retrieval_terms(
            normalize_retrieval_query("ประกาศตัวแปร C อย่างไร")
        )
        relevant = "หัวข้อ: การประกาศตัวแปร | เนื้อหา: variable declaration ใช้ data type และชื่อตัวแปร"
        unrelated = "หัวข้อ: การเปิดไฟล์ | เนื้อหา: ใช้ FILE และ fopen เพื่ออ่านข้อมูล"
        relevant_boost, relevant_coverage = lexical_relevance(relevant, terms)
        unrelated_boost, unrelated_coverage = lexical_relevance(unrelated, terms)
        self.assertGreater(relevant_boost, unrelated_boost)
        self.assertGreater(relevant_coverage, unrelated_coverage)

    def test_variable_declaration_heading_beats_generic_data_type(self):
        terms = extract_retrieval_terms(
            normalize_retrieval_query("ประกาศตัวแปร C อย่างไร")
        )
        declaration = (
            "หัวข้อ: ตัวแปรและการประกาศ (Variables) | "
            "เนื้อหา: ตัวแปรมีชนิดข้อมูล การประกาศใช้ type name = value และ initialization"
        )
        generic = (
            "หัวข้อ: ชนิดข้อมูล (Data Types) | "
            "เนื้อหา: data types define values that variables can store"
        )
        declaration_boost, _ = lexical_relevance(declaration, terms)
        generic_boost, _ = lexical_relevance(generic, terms)
        self.assertGreater(declaration_boost, generic_boost)

    def test_topic_key_groups_chunks_with_same_heading(self):
        first = "หัวข้อ: ตัวแปร (Variables) | เนื้อหา: ส่วนแรก"
        second = "หัวข้อ: ตัวแปร (Variables) | เนื้อหา: ส่วนที่สอง"
        self.assertEqual(retrieval_topic_key(first), retrieval_topic_key(second))


if __name__ == "__main__":
    unittest.main()
