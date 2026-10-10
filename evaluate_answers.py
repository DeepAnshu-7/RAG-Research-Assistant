# a second llm checks the answers. needs GROQ_API_KEY
# use --local to test the ollama model instead of groq
import os
import re
import sys

from groq import Groq

from evaluate import test_set
from rag import build_corpus, build_keyword_index, build_semantic_index, run_rag_pipeline

# not the same model that writes the answers, otherwise it may go easy on itself
JUDGE_MODEL = "qwen/qwen3.8-27b"

judge = Groq()
if "--local" in sys.argv:
    # without the key rag.py switches to ollama, judge already has the key
    del os.environ["GROQ_API_KEY"]


def ask_judge(question, chunks, answer, expected):
    context = ""
    for i, c in enumerate(chunks, start=1):
        context += f"[{i}] ({c['source']}, page {c['page']})\n{c['text']}\n\n"

    prompt = f"""You are grading an answer from a question answering system over research papers.

Context the system was given:
{context}
Question: {question}
Answer: {answer}
Expected fact (any one is enough): {", ".join(expected)}

Grade three things:
FAITHFUL: is every factual claim in the answer supported by the context? A reply saying it doesn't have enough information counts as faithful.
ANSWERED: does the answer actually answer the question (not a refusal, not off topic)?
CORRECT: does the answer contain the expected fact? Formatting differences like "13 888" vs "13,888" are fine.

Reply with exactly three lines:
FAITHFUL: YES or NO
ANSWERED: YES or NO
CORRECT: YES or NO"""

    response = judge.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        reasoning_format="hidden",
    )
    text = response.choices[0].message.content

    faithful = re.search(r"FAITHFUL:\s*YES", text, re.IGNORECASE) is not None
    answered = re.search(r"ANSWERED:\s*YES", text, re.IGNORECASE) is not None
    correct = re.search(r"CORRECT:\s*YES", text, re.IGNORECASE) is not None
    return faithful, answered, correct


if __name__ == "__main__":
    corpus = build_corpus()
    faiss_index = build_semantic_index(corpus)
    bm25_index = build_keyword_index(corpus)

    if os.environ.get("GROQ_API_KEY"):
        print(f"answer model: {os.environ.get('GROQ_MODEL', 'openai/gpt-oss-120b')} (groq)")
    else:
        print(f"answer model: {os.environ.get('OLLAMA_MODEL', 'llama3.2:3b')} (ollama)")
    print(f"judge model:  {JUDGE_MODEL}\n")

    faithful_count = 0
    answered_count = 0
    correct_count = 0

    for item in test_set:
        answer_stream, chunks = run_rag_pipeline(item["question"], corpus, faiss_index, bm25_index)
        answer = "".join(answer_stream)

        faithful, answered, correct = ask_judge(item["question"], chunks, answer, item["keywords"])
        faithful_count += faithful
        answered_count += answered
        correct_count += correct

        print(f"faithful={'Y' if faithful else 'N'} answered={'Y' if answered else 'N'} "
              f"correct={'Y' if correct else 'N'} | {item['question']}")
        if not faithful or not correct:
            print(f"    answer: {answer}")

    n = len(test_set)
    print(f"\nFaithfulness: {faithful_count / n:.0%}")
    print(f"Answered:     {answered_count / n:.0%}")
    print(f"Correct:      {correct_count / n:.0%}")
