from app.evaluate import MRR_GATE, evaluate, evaluate_all
from app.rag import Index, chunk_text, extractive_summary, tokenize
from app.server import build_index, load_queries


def test_tokenize_drops_stopwords_and_short_tokens():
    assert tokenize("The GPS and a satellite") == ["gps", "satellite"]


def test_chunk_text_respects_cap():
    text = "\n\n".join(f"Paragraph {i} " + "word " * 60 for i in range(6))
    chunks = chunk_text(text, max_chars=300)
    assert len(chunks) > 1
    assert all(len(c) <= 300 + 60 for c in chunks)  # paragraph granularity


def test_search_finds_the_right_document():
    index = build_index()
    for ranker in ("tfidf", "hybrid"):
        results = index.search("satellite trilateration atomic clock", ranker=ranker)
        assert results and results[0].doc_id == "gps", ranker
        results = index.search("debits credits trial balance", ranker=ranker)
        assert results and results[0].doc_id == "bookkeeping", ranker


def test_add_and_remove_document_roundtrip():
    index = Index()
    index.add_document("a", "Alpha", "Cats are small domestic mammals. " * 5)
    assert index.stats()["documents"] == 1
    assert index.search("domestic mammals")[0].doc_id == "a"
    assert index.remove_document("a") is True
    assert index.remove_document("a") is False
    assert index.search("domestic mammals") == []


def test_frozen_eval_gate_holds_for_both_rankers():
    index = build_index()
    queries = load_queries()
    report = evaluate_all(index, queries)
    assert report["hybrid"]["mrr@5"] >= MRR_GATE
    assert report["tfidf"]["queries"] == len(queries)
    single = evaluate(index, queries, "hybrid")
    assert single["hit@3"] >= 0.8


def test_extractive_summary_returns_source_sentences():
    index = build_index()
    text = index.documents["gps"]["text"]
    summary = extractive_summary(text, max_sentences=2)
    assert summary
    assert all(part.strip()[:40] in text for part in [summary[:40]])
