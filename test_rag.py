from rag import build_corpus, chunk_text, find_paper_in_query, reciprocal_rank_fusion, tokenize

text = " ".join(str(i) for i in range(200))
chunks = chunk_text(text, chunk_size=150, overlap=30)
assert len(chunks) == 2
assert chunks[0].split()[-30:] == chunks[1].split()[:30]
assert chunk_text("") == []
assert len(chunk_text(" ".join(["w"] * 150))) == 1

# 2 is second in both lists, so it should beat 1 and 4 which are first in only one list
assert reciprocal_rank_fusion([[1, 2, 3], [4, 2, 5]])[0] == 2

assert tokenize("Kappa, F1-score?") == ["kappa", "f1", "score"]
assert find_paper_in_query("what dataset does Paper 3 use") == "paper3.pdf"
assert find_paper_in_query("what dataset was used") is None

try:
    build_corpus("folder_that_does_not_exist")
    assert False, "should have raised"
except FileNotFoundError:
    pass

print("all tests passed")
