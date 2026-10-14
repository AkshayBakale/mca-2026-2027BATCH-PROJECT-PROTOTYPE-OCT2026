from collections import Counter

CORPUS_PATH = "data/corpus.txt"
WORD_LIST_PATH = "data/word_list.txt"

unigrams, bigrams = Counter(), Counter()

# Train the language model on the crawled corpus
try:
    with open(CORPUS_PATH, encoding="utf-8") as f:
        for line in f:
            words = line.split()
            unigrams.update(words)
            bigrams.update(zip(words, words[1:]))
except FileNotFoundError:
    print("Warning: data/corpus.txt not found. Run crawl_corpus.py first.")

# Base vocabulary so common conversational words always exist
with open(WORD_LIST_PATH) as f:
    for line in f:
        w = line.strip().lower()
        if w:
            unigrams[w] += 5

TOTAL = sum(unigrams.values())
V = len(unigrams)


def p_unigram(w):
    return (unigrams[w] + 1) / (TOTAL + V)                  # add-one (Laplace) smoothing


def p_bigram(w, prev):
    return (bigrams[(prev, w)] + 1) / (unigrams[prev] + V)  # P(w | prev), smoothed


def suggest_words(prefix, prev_word=None, top_n=3, lam=0.7):
    """Rank words starting with `prefix` by interpolated bigram/unigram probability."""
    prefix = prefix.lower()
    if not prefix:
        return []
    candidates = [w for w in unigrams if w.startswith(prefix)]

    def score(w):
        s = p_unigram(w)
        if prev_word:
            s = lam * p_bigram(w, prev_word.lower()) + (1 - lam) * s
        return s

    return [w.upper() for w in sorted(candidates, key=score, reverse=True)[:top_n]]


if __name__ == "__main__":
    prefix = input("Prefix: ")
    prev = input("Previous word (optional): ").strip() or None
    print("Suggestions:", suggest_words(prefix, prev))