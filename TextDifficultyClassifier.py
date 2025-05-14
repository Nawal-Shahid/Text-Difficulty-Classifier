# Install required packages
!pip install -q transformers datasets torch scikit-learn ipywidgets pandas numpy
!python -m spacy download en_core_web_sm


import numpy as np
import pandas as pd
import spacy
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset, ClassLabel
from sklearn.metrics import accuracy_score, f1_score
import ipywidgets as widgets
from IPython.display import display, clear_output
import random

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Sample dataset - in practice, use a larger labeled dataset
data = {
    'text': [
        # Easy texts (grade 1-3)
        "The cat sat on the mat. It was happy.",
        "I like to play with my dog in the park.",
        "She has a red ball and a blue toy.",
        "We went to the store to buy some milk.",
        "The sun is bright and the sky is blue.",
        "My family went to the beach last summer.",
        "The little bird flew to its nest.",
        "He ate an apple and drank some juice.",
        "We saw a big tree with green leaves.",
        "The children laughed and played games.",

        # Medium texts (grade 4-6)
        "The scientific method involves observation, hypothesis, and experimentation.",
        "Economic theories attempt to explain market behaviors and consumer choices.",
        "Photosynthesis converts light energy into chemical energy in plants.",
        "The Renaissance period marked significant cultural and intellectual changes.",
        "Democracy relies on citizen participation and fair elections.",
        "Volcanoes form when magma rises to the Earth's surface through vents.",
        "Ancient civilizations developed writing systems to record information.",
        "The human circulatory system transports nutrients and oxygen to cells.",
        "Fractions represent parts of a whole and can be added or subtracted.",
        "Climate patterns are influenced by ocean currents and atmospheric pressure.",

        # Hard texts (grade 7+)
        "Quantum entanglement posits that particles can instantaneously affect each other's state regardless of distance.",
        "Poststructuralist critiques challenge the notion of objective truth in literary analysis.",
        "The ontological argument for God's existence has been debated since Anselm's Proslogion.",
        "Neoliberal economic policies emphasize deregulation and free market capitalism.",
        "Epistemological skepticism questions the possibility of certain knowledge.",
        "The Schrödinger equation describes how quantum systems evolve over time.",
        "Deconstructionism examines the inherent contradictions within textual meaning.",
        "Metaphysical dualism proposes the existence of both mental and physical substances.",
        "Hegelian dialectics involves the process of thesis, antithesis, and synthesis.",
        "Postmodern architecture often incorporates eclectic styles and ironic elements."
    ],
    'label': [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  # easy
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  # medium
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2   # hard
    ]
}

# Create pandas DataFrame
df = pd.DataFrame(data)

# Convert to HuggingFace Dataset
class_names = ['easy', 'medium', 'hard']
class_label = ClassLabel(names=class_names)
dataset = Dataset.from_pandas(df)

# Split dataset into train and test
dataset = dataset.train_test_split(test_size=0.2, seed=42)

# Load tokenizer and model
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Prepare labels
tokenized_datasets = tokenized_datasets.map(
    lambda example: {"labels": example["label"]},
    batched=True
)

# Load pre-trained model and modify for classification
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(class_names),
    id2label={i: label for i, label in enumerate(class_names)},
    label2id={label: i for i, label in enumerate(class_names)}
)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_steps=10,
    save_strategy="no",
    report_to="none"
)

# Compute metrics function
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="weighted")
    }

# Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    compute_metrics=compute_metrics,
)

# Train the model
print("Training model...")
trainer.train()

# Evaluate the model
print("\nEvaluation results:")
eval_results = trainer.evaluate()
print(f"Accuracy: {eval_results['eval_accuracy']:.2f}")
print(f"F1 Score: {eval_results['eval_f1']:.2f}")

# Create GUI for text classification
text_input = widgets.Textarea(
    value='',
    placeholder='Enter text to analyze...',
    description='Text:',
    layout=widgets.Layout(width='80%', height='200px')
)

classify_button = widgets.Button(
    description="Classify Difficulty",
    button_style='success',
    layout=widgets.Layout(width='150px')
)

output = widgets.Output()

def classify_text(b):
    with output:
        clear_output()
        text = text_input.value.strip()
        if not text:
            print("Please enter some text to analyze.")
            return

        # Tokenize input
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

        # Get prediction
        with torch.no_grad():
            logits = model(**inputs).logits

        # Process output
        predicted_class_idx = torch.argmax(logits, dim=1).item()
        confidence = torch.softmax(logits, dim=1)[0][predicted_class_idx].item()
        predicted_label = class_names[predicted_class_idx]

        # Display results
        print("\n--- Text Difficulty Analysis ---")
        print(f"Predicted Level: {predicted_label.capitalize()}")
        print(f"Confidence: {confidence:.1%}")

        # Add detailed explanation
        print("\nExplanation:")
        if predicted_label == 'easy':
            print("This text contains simple vocabulary and sentence structures.")
            print("Suitable for elementary school readers (Grade 1-3).")
        elif predicted_label == 'medium':
            print("This text contains intermediate vocabulary and some complex sentences.")
            print("Suitable for middle school readers (Grade 4-6).")
        else:
            print("This text contains advanced vocabulary and complex concepts.")
            print("Suitable for high school and above (Grade 7+).")

        # Show feature analysis
        print("\nKey Features:")
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)

        # Calculate average sentence length
        sent_lengths = [len(sent) for sent in doc.sents]
        avg_sent_len = sum(sent_lengths) / len(sent_lengths) if sent_lengths else 0
        print(f"- Average sentence length: {avg_sent_len:.1f} characters")

        # Count rare words (assuming words not in basic vocabulary are rare)
        basic_vocab = set(word.lower() for word in nlp.vocab.strings if nlp.vocab[word].is_lower)
        rare_words = [token.text for token in doc if token.is_alpha and token.text.lower() not in basic_vocab]
        print(f"- Rare word count: {len(rare_words)}")

        # Count complex punctuation
        complex_punct = sum(1 for token in doc if token.text in [';', ':', '—', '(', ')'])
        print(f"- Complex punctuation: {complex_punct}")

classify_button.on_click(classify_text)

# Create a reset button
reset_button = widgets.Button(
    description="Reset",
    button_style='warning',
    layout=widgets.Layout(width='150px')
)

def reset_input(b):
    text_input.value = ''
    with output:
        clear_output()

reset_button.on_click(reset_input)

# Create button layout
button_layout = widgets.HBox([classify_button, reset_button])

# Display the GUI
display(widgets.VBox([
    widgets.HTML("<h2>Text Difficulty Classifier</h2>"),
    widgets.HTML("<p>Enter text to analyze its reading difficulty level:</p>"),
    text_input,
    button_layout,
    output
]))

print("\nReady to analyze text difficulty! Enter some text above and click 'Classify Difficulty'.")
