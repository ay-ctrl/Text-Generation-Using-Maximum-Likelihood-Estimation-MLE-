# Text Generation Using Maximum Likelihood Estimation (MLE)

This project was developed for the **Scientific Computing** course.

## Overview
The project implements simple statistical language models using **Maximum Likelihood Estimation (MLE)** to generate text from a corpus.

The implementation includes:
- Text preprocessing
- Unigram language model
- Bigram language model
- Probability estimation using MLE
- Random text generation based on learned probability distributions

## Dataset
- **Corpus:** Harry Potter text (`Harry_Potter.txt`)
- The text is preprocessed by:
  - Converting to lowercase
  - Removing unnecessary punctuation
  - Tokenizing into words
  - Adding sentence boundary tokens (`<s>` and `</s>`)

## Project Structure

```
├── Harry_Potter.txt
├── main.py
├── Report.pdf
└── README.md
```

## Output
The program:
1. Preprocesses the input text.
2. Builds unigram and bigram language models using MLE.
3. Calculates word and conditional probabilities.
4. Generates multiple text samples using both models.
