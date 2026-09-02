"""ทดสอบ REBEL decoder output parser: <trip>/<subj>/<obj> และ edge case ต่าง ๆ"""

from triple_extraction.extractor.rebel import parse_rebel_output


def test_parses_single_triplet_in_real_rebel_order():
    # rebel-large จริง: <triplet> head <subj> tail <obj> relation (relation อยู่ท้าย)
    raw = "<triplet> Marie Curie <subj> Warsaw <obj> place of birth"

    parsed = parse_rebel_output(raw)

    assert parsed.triples == [
        {"head": "Marie Curie", "relation": "place of birth", "tail": "Warsaw"}
    ]
    assert parsed.unparsed == []


def test_accepts_trip_tag_as_triplet_alias():
    raw = "<trip> Marie Curie <subj> Warsaw <obj> place of birth"

    parsed = parse_rebel_output(raw)

    assert parsed.triples == [
        {"head": "Marie Curie", "relation": "place of birth", "tail": "Warsaw"}
    ]


def test_parses_multiple_triples():
    raw = (
        "<triplet> Paris <subj> France <obj> capital of"
        "<triplet> Eiffel Tower <subj> Paris <obj> located in"
    )

    parsed = parse_rebel_output(raw)

    assert parsed.triples == [
        {"head": "Paris", "relation": "capital of", "tail": "France"},
        {"head": "Eiffel Tower", "relation": "located in", "tail": "Paris"},
    ]


def test_accepts_spec_sep_token_as_separator():
    # spec เขียนรูปแบบ <trip> h <sep> r <sep> t — parser ต้องรองรับด้วย
    raw = "<trip> head <sep> relation <sep> tail"

    parsed = parse_rebel_output(raw)

    assert parsed.triples == [
        {"head": "head", "relation": "relation", "tail": "tail"}
    ]


def test_wrong_column_count_goes_to_unparsed():
    raw = "<triplet> only-head-and-tail <subj> b<triplet> a <subj> b <obj> r"

    parsed = parse_rebel_output(raw)

    assert parsed.triples == [{"head": "a", "relation": "r", "tail": "b"}]
    assert parsed.unparsed == ["only-head-and-tail <subj> b"]


def test_text_without_trip_tag_is_unparsed_raw():
    raw = "just plain decoder noise"

    parsed = parse_rebel_output(raw)

    assert parsed.triples == []
    assert parsed.unparsed == ["just plain decoder noise"]


def test_empty_input_gives_empty_result():
    parsed = parse_rebel_output("")

    assert parsed.triples == []
    assert parsed.unparsed == []


def test_triple_with_empty_field_goes_to_unparsed():
    raw = "<triplet>  <subj> Warsaw <obj> place of birth"

    parsed = parse_rebel_output(raw)

    assert parsed.triples == []
    assert parsed.unparsed == ["<subj> Warsaw <obj> place of birth"]


def test_fields_are_trimmed():
    raw = "<triplet>  Head  <subj>  Tail  <obj>  rel "

    parsed = parse_rebel_output(raw)

    assert parsed.triples == [{"head": "Head", "relation": "rel", "tail": "Tail"}]


def test_ignores_sequence_special_tokens_around_output():
    raw = "<s><pad><triplet> A <subj> B <obj> r</s>"

    parsed = parse_rebel_output(raw)

    assert parsed.triples == [{"head": "A", "relation": "r", "tail": "B"}]
    assert parsed.unparsed == []
