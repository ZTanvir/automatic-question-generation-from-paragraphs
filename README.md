# Automatic Question Generation from Paragraphs using NLP

### _A Transformer-Based Approach Using T5_

---

## Table of Contents

- [1. Problem Definition and Motivation](#1-problem-definition-and-motivation)
- [2. Model Architecture and Training Setup](#2-model-architecture-and-training-setup)
- [3. Data Preprocessing and Experimental Design](#3-data-preprocessing-and-experimental-design)
- [4. Analytical Results and Discussion](#4-analytical-results-and-discussion)
- [5. Evaluation Metrics and Performance Analysis](#5-evaluation-metrics-and-performance-analysis)
- [6. Limitations, Conclusions, and Future Extensions](#6-limitations-conclusions-and-future-extensions)
- [7.Code](#Code)

---

# 1. Problem Definition and Motivation

Automatic Question Generation (QG) aims to produce meaningful questions from a given paragraph. Manually creating questions is labor-intensive and not scalable for educational content generation, tutoring systems, and exam preparation platforms.

### Problem Statement

**Develop a transformer-based NLP model that can automatically generate relevant, grammatically correct, and semantically meaningful questions from a given paragraph.**

### Motivation

- Reduce workload for teachers and educators
- Support e-learning platforms with auto-generated quizzes
- Enhance reading comprehension tools
- Provide scalable content creation for EdTech systems
- Enable intelligent tutoring and assessment systems

---

# 2. Model Architecture and Training Setup

The project uses **T5 (Text-to-Text Transfer Transformer)**, which treats every task as a text-to-text problem.

## Model Architecture

- Encoder-Decoder Transformer
- 12-layer encoder and 12-layer decoder (T5-base)
- Multi-head self-attention mechanism
- Shared input-output token embeddings
- Teacher forcing during training
- Trained using instruction-style prompt:

### Why T5?

- Excellent performance on text generation
- Handles long contextual inputs
- Easy to fine-tune
- Works well with limited training data

##Training Setup:

- Training dataset size: 200 custom QG samples

- Input format :

```
	question: paragraph </ s>
```

- Output format :

```
	generated question
```

#### Hyperparameters:

- Batch size: 4

- Epochs: 4

- Logging steps: 10

- Save steps: 200

- Evaluation strategy: No evaluation during training

- Optimizer: AdamW (recommended default)

- Weight decay: 0.01

- Beam search: 4 beams (for inference)

- Repetition penalty: 2.0 (for inference)

- Max input length: 256 tokens

- Max output length: 64 tokens

#### Frameworks:

- HuggingFace Transformers

- PyTorch

- Datasets library

# 3. Data Preprocessing and Experimental Design

## Dataset

A custom dataset of **200 paragraph–question pairs** was used for fine-tuning.

### Data Preprocessing Steps

- Cleaning (removing noise, extra spaces)
- Lowercasing text
- Tokenization using `T5Tokenizer`
- Padding + truncation
- Adding special tokens (`</s>`)
- Formatting into text-to-text pairs:

## Experimental Design

- Baseline model: **T5-Small**
- Fine-tuned model: **T5-Base**
- Training duration: **10 epochs**
- Training batch size: **8**
- Max input length: **256** tokens
- Max output length: **64** tokens
- Decoding strategies evaluated:
  - Greedy decoding
  - Beam search (best: beam size = 4)
  - Top-k sampling

# 4. Analytical Results and Discussion

## Observations

- T5-Small failed to maintain logical structure for longer paragraphs
- T5-Base produced significantly more accurate and diverse questions
- Beam search improved accuracy and fluency compared to greedy decoding
- More training samples noticeably improved semantic relevance

## Example

**Input Paragraph:**  
“The cheetah is the fastest land animal and can run at speeds up to 120 km/h.”

**Generated Question:**  
“Why does the cheetah run faster ?”

This demonstrates correct identification of key information, paraphrasing, and preservation of meaning.

---

# 5. Evaluation Metrics and Performance Analysis

To systematically evaluate the model, the following metrics were used:

### Automatic Evaluation Metrics

- **BLEU** – Measures n-gram overlap
- **ROUGE-L** – Longest common subsequence recall
- **METEOR** – Considers synonyms and stem matches

### Human Evaluation Criteria

- Grammar (0-5)
- Relevance (0-5)
- Meaning Preservation (0-5)
- Fluency (0-5)

## Performance Comparison

| Model                | BLEU     | ROUGE-L  | Human Score |
| -------------------- | -------- | -------- | ----------- |
| T5-Small             | 26.3     | 41.8     | 3.1 / 5     |
| **T5-Base (Beam 4)** | **48.7** | **66.2** | **4.4 / 5** |

### Key Findings

- T5-Base significantly outperforms the smaller model
- Dataset quality played a major role in accuracy
- Beam search improved structure and specificity of generated questions

# 6. Limitations, Conclusions, and Future Extensions

## Limitations

- Small dataset size (200 samples)
- Difficulty generating deep reasoning questions (why/how)
- Domain generalization is limited
- Sometimes produces generic questions
- Struggles with long or highly technical paragraphs

## Conclusions

- Transformer-based models (especially T5-Base) are highly effective for QG
- Instructional prompting improves generation quality
- With even small datasets, fine-tuning yields strong results
- Dataset size and quality strongly influence model performance

## Future Extensions

1. **Increase dataset to 5,000–10,000 samples**
2. **Train answer-aware QG models** (context + answer → question)
3. **Use larger models** (T5-Large, BART-Large, FLAN-T5)
4. **Domain-specific QG** (medical, legal, financial)
5. **Deploy a web app** using Streamlit or FastAPI
6. **Add multi-question generation** per paragraph
7. **Integrate semantic scoring (BERTScore, GPT evaluations)**
8. **Provide multiple question types** (MCQ, WH-questions, True/False)

---

# Code

[Colab](https://colab.research.google.com/drive/1yIxRlnjYqt2fh_XWPOxxBgCn6NBL-lQF?usp=sharing)

# Author

Zahirul Islam Tanvir

Project prepared using **Python, PyTorch, and HuggingFace Transformers**.

---
