"""ทดสอบ REBEL decoder output parser: <trip>/<subj>/<obj> และ edge case ต่าง ๆ"""

from triple_extraction.extractor.rebel import parse_rebel_output


def test_parses_single_trip_tag_sequence():
    raw = "<trip> Marie Curie <subj> born in <obj> Warsaw"

    parsed = parse_rebel_output(raw)

    assert parsed.triples == [
        {"head": "Marie Curie", "relation": "born in", "tail": "Warsaw"}
    ]
    assert parsed.unparsed == []


def test_parses_multiple_triples():
    raw = (
        "<trip> Paris <subj> capital of <obj> France"
        "<trip> Eiffel Tower <subj> located in <obj> Paris"
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
    raw = "<trip> only-head-and-relation <subj> rel<trip> a <subj> r <obj> b"

    parsed = parse_rebel_output(raw)

    assert parsed.triples == [{"head": "a", "relation": "r", "tail": "b"}]
    assert parsed.unparsed == ["only-head-and-relation <subj> rel"]


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
    raw = "<trip>  <subj> rel <obj> tail"

    parsed = parse_rebel_output(raw)

    assert parsed.triples == []
    assert parsed.unparsed == ["<subj> rel <obj> tail"]


def test_fields_are_trimmed():
    raw = "<trip>  Head  <subj>  rel  <obj>  Tail "

    parsed = parse_rebel_output(raw)

    assert parsed.triples == [{"head": "Head", "relation": "rel", "tail": "Tail"}]
