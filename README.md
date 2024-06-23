# LLM Fine-Tuning: Python Q&A

## Overview

This project aims to improve the capabilities of a pre-trained language model to understand and generate Python programming-related responses. By fine-tuning the Flan T5 model on a dataset of Python questions and answers, we enhance its performance for automated coding assistance tools, which are essential in educational and development environments.

## Model

The chosen model for this project is the Flan T5 model, an advanced version of Google’s T5 (Text-To-Text Transfer Transformer) model. Flan T5 is fine-tuned on over 1000 additional tasks, making it exceptionally versatile.

### Key Features:
- **Model:** Flan T5 base model
- **Libraries:**
  - `nltk` for natural language processing tasks
  - `transformers` for model implementation
  - `tokenizers` for tokenizing input text
  - `evaluate` for model evaluation
  - `rouge_score` for calculating ROUGE metrics
  - `sentencepiece` for tokenization
  - `huggingface_hub` for accessing models from Hugging Face
- **Training Framework:** PyTorch using HuggingFace's `Trainer` class
- **Data Preprocessing:** Used BeautifulSoup for HTML entity conversion, pandas for data manipulation
- **Development Environment:** Google Colab, ensuring a flexible and accessible setup


## Dataset

The dataset used for fine-tuning consists of Python-related questions and answers sourced from Stack Overflow. The dataset is available on Kaggle.

### Preprocessing Steps:
1. **Cleaning:** Converted HTML entities to plain text using BeautifulSoup.
2. **Quality Filtering:** Retained answers with at least two upvotes.
3. **Diversity:** Limited to the top two answers per question based on upvotes.
4. **Merging:** Combined the cleaned questions and answers to create the final training dataset.

## Training Procedure

Training was conducted in a high-performance computing environment with the following strategies:
- **Incremental Training:** The model was trained on batches of 20,000 samples, each undergoing three epochs.
- **Data Segmentation:** Managed computational load by segmenting the data into manageable batches.

### Evaluation Metrics:
- **ROUGE Scores:** Used to quantitatively measure the overlap between model-generated answers and actual answers.
- **Human Evaluation:** Fixed set of six random Python questions to qualitatively assess the model's performance.

## Results

The model showed significant improvements in both quantitative and qualitative evaluations after each batch of fine-tuning.

### Key Improvements:
- **Before Fine-Tuning:** The model performed poorly, with ROUGE scores ranging from 0.01 to 0.05.
- **After Fine-Tuning:** Significant improvement in ROUGE scores and the relevance of generated answers.

## Future Work

Future improvements could involve:
- Fine-tuning on a larger corpus of data.
- Using more computing power for larger batch sizes.
- Sourcing additional Python-related data.
- Incorporating reinforcement learning with human feedback.

## References

- [Kaggle - Fine-tuning LLMs](https://www.kaggle.com/code/aliabdin1/llm-04a-�ne-tuning-llms)
- [Evaluate Translation or Summarization with ROUGE Similarity Score - MATLAB](https://www.mathworks.com/help/textanalytics/ref/rougeevaluationscore.html)
- [Finetuning Large language models using QLoRA](https://www.kaggle.com/code/neerajmohan)
