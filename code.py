import collections
import random
import re

# =====================================================================
# PART 1: TEXT PREPROCESSING
# =====================================================================

# Sample text
raw_text = open("Harry_Potter.txt", "r", encoding="utf-8").read()

def preprocess_text(text):
    # 1. Convert the text to lowercase.
    text = text.lower()

    # 2. Remove unnecessary punctuation if needed.
    text = re.sub(r"[^\w\s\.]", "", text)

    # 3. Tokenize the text into words.
    sentences = text.split(".")
    processed_tokens = []

    for sentence in sentences:
        words = sentence.strip().split()
        if words:  # Skip empty sentences
            # 4. Add special tokens <s> and </s> to mark sentence boundaries.
            processed_tokens.append("<s>")
            processed_tokens.extend(words)
            processed_tokens.append("</s>")

    return processed_tokens

tokens = preprocess_text(raw_text)

print("--- PART 1: TEXT PREPROCESSING ---")
print(f"Raw text first 100 characters:\n{raw_text[:100].strip()}...\n")
print(f"Tokenized text first 15 tokens:\n{tokens[:15]}\n")
print("-" * 50)

# =====================================================================
# PART 2: BUILD THE MLE LANGUAGE MODELS
# =====================================================================

# --- 2.1 Unigram Model ---
unigram_counts = collections.Counter(tokens) # generates a dictionary that shows us how many of each word there are
total_unigrams = sum(unigram_counts.values()) # calculates the total number of words in the text (including duplicates)

# MLE Probablity Calculation: P(w) = count(w) / N
unigram_probs = {
    word: count / total_unigrams for word, count in unigram_counts.items()
}

# --- 2.2 Bigram Model ---
bigram_counts = collections.defaultdict(collections.Counter) # Creates a nested (2-dimensional) dictionary structure.

for i in range(len(tokens) - 1):
    w1, w2 = tokens[i], tokens[i + 1] # Which word comes most often after which word?
    bigram_counts[w1][w2] += 1

# MLE Probablity Calculation: P(w_i | w_{i-1}) = count(w_{i-1}, w_i) / count(w_{i-1})
bigram_probs = collections.defaultdict(dict)
for w1, w2_counts in bigram_counts.items():
    total_w1_count = sum(w2_counts.values())
    for w2, count in w2_counts.items():
        bigram_probs[w1][w2] = count / total_w1_count

print("--- PART 2: EXAMPLE MLE PROBABILITIES ---")
print("Unigram Examples:")
for word in list(unigram_probs.keys())[:3]:
    print(f"  P('{word}') = {unigram_probs[word]:.4f}")

print("\nBigram Samples:")
example_w1 = "the"
if example_w1 in bigram_probs:
    for w2, prob in list(bigram_probs[example_w1].items())[:3]:
        print(f"  P('{w2}' | '{example_w1}') = {prob:.4f}")
print("-" * 50)

# =====================================================================
# PART 3: TEXT GENERATION
# =====================================================================

def generate_unigram_text(probs, length=30):
    words = list(probs.keys())
    probabilities = list(probs.values())
    # Random selection according to the probabilities
    generated = random.choices(words, weights=probabilities, k=length)
    return " ".join(generated)

def generate_bigram_text(probs, start_token="<s>", length=30):
    current_word = start_token
    generated = []

    for _ in range(length):
        # If it is not in the current word model or there is no next word option, start randomly
        if current_word not in probs or not probs[current_word]:
            current_word = random.choice(list(probs.keys()))

        next_words = list(probs[current_word].keys())
        next_probs = list(probs[current_word].values())

        # Select the next word according to its probability
        next_word = random.choices(next_words, weights=next_probs, k=1)[0]
        generated.append(next_word)
        current_word = next_word

    return " ".join(generated)

print("--- PART 3: TEXT GENERATION (5 EXAMPLES - AT LEAST 30 WORDS) ---")

print("\n[UNIGRAM MODEL OUTPUTS]")
for i in range(1, 6):
    print(f"Example {i}: {generate_unigram_text(unigram_probs, length=35)}")

print("\n[BIGRAM MODEL OUTPUTS]")
for i in range(1, 6):
    print(f"Example {i}: {generate_bigram_text(bigram_probs, length=35)}")
print("-" * 50)