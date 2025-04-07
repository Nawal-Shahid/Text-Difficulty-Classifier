# Text-Difficulty-Classifier
A transformer-based text difficulty classifier for English and Urdu that categorizes content into Easy, Medium, or Hard levels for personalized learning.

# Text Difficulty Classifier

This repository implements a transformer-based text difficulty classifier designed to categorize educational content into three difficulty levels: **Easy**, **Medium**, and **Hard**. It uses **XLM-Roberta**, a multilingual transformer model from HuggingFace's Transformers library, and is optimized for both **English** and **Urdu** texts.

## 🚀 Features

- **Multilingual Support**: Handles both English and Urdu texts.
- **3 Difficulty Levels**: Classifies text into:
  - **Easy**: Grade 1-3 reading level
  - **Medium**: Grade 4-6 reading level
  - **Hard**: Grade 7+ reading level
- **Transformer Architecture**: Uses `xlm-roberta-base` for high performance in text classification tasks.
- **Trainable & Extensible**: Easily fine-tune with your own labeled data.

## 📊 Dataset Format

The dataset should be in CSV or DataFrame format with the following structure:

| text | label  |
|------|--------|
| "The cat sat on the mat." | Easy   |
| "Photosynthesis is essential for plant life." | Medium |
| "Quantum mechanics is the foundation of modern physics." | Hard |
| "یہ ایک آسان جملہ ہے۔" | Easy |
| "دماغی بیماریوں کی تشخیص اور علاج کا سائنسی عمل مشکل ہے۔" | Hard |

## 🛠️ Installation

Install the required libraries by running:

```bash
pip install transformers datasets tensorflow scikit-learn pandas
```


## Usage


