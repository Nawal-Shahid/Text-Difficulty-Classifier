# Text-Difficulty-Classifier {English & Urdu}
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

---

## Dataset

**Note:** The current implementation uses a small **synthetic dataset** for demonstration.

- 10 Easy texts (simple vocabulary and sentence structures)
- 10 Medium texts (intermediate vocabulary and concepts)
- 10 Hard texts (advanced vocabulary and complex syntax)

**For production**, replace this with a larger, professionally labeled dataset.

---

## Model Architecture

- **Base Model:** `distilbert-base-uncased`  
- **Classification Head:** Custom dense layer for 3-class classification  

**Training Parameters:**

| Parameter      | Value       |
|----------------|-------------|
| Learning Rate  | 2e-5        |
| Batch Size     | 8           |
| Epochs         | 5           |
| Weight Decay   | 0.01        |

---

## How It Works

1. **Text Preprocessing:**
   - Tokenization with `DistilBERT tokenizer`
   - Padding & truncation

2. **Feature Extraction:**
   - Embeddings from the transformer
   - Linguistic features (sentence length, rare words, punctuation)

3. **Classification:**
   - Predicts difficulty level with confidence score
   - Outputs feature-based explanation

---

## GUI Components

| Component           | Function                                   |
|---------------------|--------------------------------------------|
| Text Input Area     | Enter or paste the text to analyze         |
| Classify Button     | Triggers prediction                        |
| Reset Button        | Clears input/output                        |
| Results Display     | Shows prediction, confidence, and analysis |

---

## Usage

1. Run all notebook cells  
2. Paste input text in the GUI  
3. Click **"Classify Difficulty"**  
4. View:
   - Predicted difficulty
   - Confidence score
   - Grade-level interpretation
   - Text features explanation

---

## Evaluation Metrics

- **Accuracy:** Percentage of correct predictions  
- **F1 Score:** Weighted average of precision & recall  

**Expected (synthetic dataset):**

- Accuracy: ~90–100%  
- F1 Score: ~90–100%

---

## Customization Options

### Model Selection

```python
model_name = "bert-base-uncased"  # You can change this
```

### Training Parameters

Modify `TrainingArguments` to update learning rate, batch size, or epochs.

### Dataset

Replace with your own dataset using a `pandas DataFrame` format:

```python
df = pd.DataFrame({
  'text': [...],
  'label': [...]  # e.g., 0 = Easy, 1 = Medium, 2 = Hard
})
```

### Difficulty Levels

Modify or extend class labels:

```python
class_names = ['beginner', 'intermediate', 'advanced', 'expert']
```

---

## Limitations

- Small synthetic dataset — real-world performance may vary
- Does not account for individual reader abilities
- Depends heavily on training data quality

---

## Future Improvements

- Use real-world data with expert-labeled difficulty ratings  
- Integrate standard readability scores (Flesch-Kincaid, etc.)  
- Analyze syntactic complexity with deeper NLP techniques  
- Build a web app interface for public use  
- Add batch processing and model persistence

---

## Troubleshooting

| Issue                  | Solution                                 |
|------------------------|------------------------------------------|
| CUDA Out of Memory     | Reduce batch size or use a smaller model |
| Low Performance        | Train longer / use a bigger dataset      |
| Tokenization Errors    | Clean text inputs and encoding           |

---

## License

This project is open-source and intended for **educational and research purposes** only.  
For commercial use, please review the licenses of the following dependencies:

- [Transformers](https://github.com/huggingface/transformers/blob/main/LICENSE)
- [PyTorch](https://github.com/pytorch/pytorch/blob/main/LICENSE)
- [spaCy](https://github.com/explosion/spaCy/blob/master/LICENSE)
```

