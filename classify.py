__copyright__ = """
The MIT License (MIT)

Copyright © 2025 Koray Birenheide

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), 
to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS 
IN THE SOFTWARE.
"""
__license__ = "The MIT License (MIT)"

import argparse
import joblib
import json
import re
import sys
import nltk
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import accuracy_score

# Ensure necessary NLTK resources are available
nltk.download("punkt", quiet=True)
nltk.download("wordnet", quiet=True)

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

def preprocess_message(message: str) -> str:
    """
    Preprocesses a given text message:
    - Converts to lowercase
    - Removes non-alphabetic characters (except key sentiment symbols)
    - Tokenizes and lemmatizes words
    """
    message = message.lower()
    message = re.sub(r"[^a-z\s$!?]", "", message)  # Keep key sentiment symbols
    tokens = word_tokenize(message)
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)

# Argument parsing
parser = argparse.ArgumentParser(description="Classify text using a trained model.")
parser.add_argument("model", type=str, help="Path to the trained model (.joblib).")

# Mutually exclusive classification modes
mode_group = parser.add_mutually_exclusive_group(required=True)
mode_group.add_argument("--text", type=str, help="Text input to classify.")
mode_group.add_argument("--evaluate", type=str, help="Evaluate model using a dataset file (JSON with 'text' and 'label' fields).")
mode_group.add_argument("--file", type=str, help="Classify multiple lines of text from a file (one per line).")

args = parser.parse_args()

# Load trained model
try:
    model = joblib.load(args.model)
    print(f"[LOG][INFO] Loaded model: {args.model}")
except Exception as e:
    print(f"[LOG][ERROR] Failed to load model: {e}")
    sys.exit(1)

# **SINGLE TEXT CLASSIFICATION MODE**
if args.text:
    processed_text = preprocess_message(args.text)

    # Predict classification
    try:
        prediction = model.predict([processed_text])[0]
        result = {"text": args.text, "label": int(prediction)}
        print(json.dumps(result, indent=4))
    except Exception as e:
        print(f"[LOG][ERROR] Prediction failed: {e}")
        sys.exit(1)

# **DATASET EVALUATION MODE**
elif args.evaluate:
    # Load dataset file
    try:
        with open(args.evaluate, "r") as f:
            dataset = json.load(f)
            if not isinstance(dataset, list) or not all("text" in entry and "label" in entry for entry in dataset):
                raise ValueError("Dataset must be a JSON array with 'text' and 'label' fields.")
        print(f"[LOG][INFO] Loaded dataset from {args.evaluate} with {len(dataset)} entries.")
    except Exception as e:
        print(f"[LOG][ERROR] Failed to load dataset: {e}")
        sys.exit(1)

    # Convert dataset to Pandas DataFrame for cleanup
    df = pd.DataFrame(dataset)

    # Remove null values and duplicates
    initial_size = len(df)
    df = df.dropna(subset=["text", "label"]).drop_duplicates(subset=["text"])
    cleaned_size = len(df)

    print(f"[LOG][INFO] Removed {initial_size - cleaned_size} null/duplicate entries. Evaluating {cleaned_size} entries.")

    # Preprocess texts
    texts = df["text"].apply(preprocess_message).tolist()
    true_labels = df["label"].tolist()

    # Predict classifications
    try:
        predictions = model.predict(texts)
    except Exception as e:
        print(f"[LOG][ERROR] Batch prediction failed: {e}")
        sys.exit(1)

    # Calculate accuracy
    accuracy = accuracy_score(true_labels, predictions)
    total_samples = len(true_labels)

    # Output results
    results = {
        "tested_samples": total_samples,
        "accuracy": round(accuracy * 100, 2)
    }
    print(json.dumps(results, indent=4))

    # Write to a JSON file
    with open("evaluation_results.json", "w") as f:
        json.dump(results, f, indent=4)

    print("[LOG][INFO] Evaluation results saved to evaluation_results.json")

# **FILE-BASED BATCH CLASSIFICATION MODE**
elif args.file:
    try:
        with open(args.file, "r") as f:
            lines = f.readlines()
            texts = [line.strip() for line in lines if line.strip()]
        print(f"[LOG][INFO] Loaded {len(texts)} lines from {args.file}.")
    except Exception as e:
        print(f"[LOG][ERROR] Failed to read file: {e}")
        sys.exit(1)

    # Preprocess texts
    processed_texts = [preprocess_message(text) for text in texts]

    # Predict classifications
    try:
        predictions = model.predict(processed_texts)
    except Exception as e:
        print(f"[LOG][ERROR] Batch prediction failed: {e}")
        sys.exit(1)

    # Output results
    results = [{"text": text, "label": int(pred)} for text, pred in zip(texts, predictions)]
    print(json.dumps(results, indent=4))
