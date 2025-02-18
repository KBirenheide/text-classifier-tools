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
import pandas as pd
import numpy as np
import math
import re
import nltk
import joblib
import os
import time
import json
import subprocess
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Ensure necessary NLTK resources are available
nltk.download("punkt", quiet=True)
nltk.download("wordnet", quiet=True)

# Argument parsing
parser = argparse.ArgumentParser(description="Train and fine-tune a text classification model.")
parser.add_argument("dataset", type=str, help="Path to training dataset (JSON format with 'text' and 'label' columns).")
parser.add_argument("--name", type=str, required=True, help="Classifier name (e.g., 'spam', 'sentiment', 'custom').")
parser.add_argument("--skip-data-prep", action="store_true", help="Skip text preprocessing and load preprocessed dataset directly.")
parser.add_argument("--common-words", type=int, default=20, help="Number of most common words to ignore dynamically.")
parser.add_argument("--evaluate", type=str, help="Path to test dataset to evaluate after training.")
parser.add_argument("--fine-tune", type=int, help="Number of fine-tuning steps per parameter.")
parser.add_argument("--solver", type=str, default=None,
                    choices=["liblinear", "lbfgs", "saga", "newton-cg"],
                    help="Solver for optimization (default: 'liblinear').")

# Hyperparameter tuning arguments
parser.add_argument("--C", type=float, default=None, help="Regularization strength.")
parser.add_argument("--weight", type=float, default=None, help="Class weight adjustment.")
parser.add_argument("--max-iter", type=int, default=None, help="Max iterations for training.")
parser.add_argument("--min-df", type=int, default=None, help="Minimum document frequency for TF-IDF.")
parser.add_argument("--max-df", type=float, default=None, help="Maximum document frequency for TF-IDF.")

args = parser.parse_args()

print(f"[LOG][INFO] Starting training for classifier: {args.name}")
print(f"[LOG][INFO] Using dataset: {args.dataset}")

# **DEFAULT VALUES & STEP SIZES FOR TUNING**
defaults = {
    "C": 1.0,
    "weight": 1.0,
    "max_iter": 500,
    "min_df": 4,
    "max_df": 0.85,
    "solver": "liblinear"
}
optimal = {
    "C": [1.0, 1.0],
    "weight": [1.0, 1.0],
    "max_iter": [500, 500],
    "min_df": [4, 4],
    "max_df": [0.85, 0.85],
}
steps = {
    "C": 1.0,
    "weight": 0.1,
    "max_iter": 250,
    "min_df": 2,
    "max_df": 0.05
}
limits = {
    "C": (0.1, 10),  # Min 0.1, Max 10
    "weight": (0.1, 2.0),  # Min 0.1, Max 2.0 (no zero or negatives)
    "max_iter": (100, -1),  # Min 100, No max limit (-1)
    "min_df": (1, 10),  # Min 1, Max 10
    "max_df": (0.5, 0.95)  # Min 0.5, Max 0.95
}

preprocessed_file = f"output/preprocessed_{os.path.basename(args.dataset)}"

# Load dataset
if args.skip_data_prep and os.path.exists(preprocessed_file):
    print(f"[LOG][INFO] Skipping data preparation. Loading preprocessed dataset: {preprocessed_file}")
    df = pd.read_json(preprocessed_file)
else:
    try:
        df = pd.read_json(args.dataset)
        print("[LOG][INFO] Dataset loaded successfully.")
    except Exception as e:
        print(f"[LOG][ERROR] Failed to load dataset: {e}")
        exit(1)

    # Remove null values and duplicates
    df = df.dropna(subset=["text", "label"]).drop_duplicates(subset=["text"])

    print(f"[LOG][INFO] Final dataset size after cleaning: {len(df)}")

    # Initialize lemmatizer for text preprocessing
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

    # Apply text preprocessing
    print("[LOG][INFO] Preprocessing dataset...")
    df["text"] = df["text"].apply(preprocess_message)

    # **DYNAMICALLY IDENTIFY MOST COMMON WORDS**
    all_words = " ".join(df["text"]).split()
    word_counts = Counter(all_words)
    most_common_words = [word for word, _ in word_counts.most_common(args.common_words)]

    print(f"[LOG][INFO] Most common words excluded: {', '.join(most_common_words)}")

    # Save preprocessed dataset
    df.to_json(preprocessed_file, orient="records", indent=4)
    print(f"[LOG][INFO] Preprocessed dataset saved as: {preprocessed_file}")


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.2, random_state=42)

# **MODEL TRAINING FUNCTION**
def train_and_evaluate(params):
    """
    Trains a model with given parameters and evaluates it using classify.py.
    Returns accuracy score.
    """
    pipeline = Pipeline([
        ("vectorizer", TfidfVectorizer(min_df=params["min_df"], 
                                        max_df=params["max_df"], 
                                        ngram_range=(1,3), 
                                        sublinear_tf=True)),
        ("classifier", LogisticRegression(max_iter=params["max_iter"], 
                                            solver=params["solver"], 
                                            class_weight={0: 1.0, 1: params["weight"]}, 
                                            C=params["C"]))
    ])

    pipeline.fit(X_train, y_train)

    # Save model
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    model_filename = f"output/classifier_{args.name}_{timestamp}.joblib"
    print(f"[LOG][INFO] Saving model to: {model_filename}")
    joblib.dump(pipeline, model_filename)

    # Evaluate
    if args.evaluate:
        print(f"[LOG][INFO] Starting model evaluation")
        result = subprocess.run(
            ["python3", "classify.py", model_filename, "--evaluate", args.evaluate],
            capture_output=True, text=True
        )
        # Read results from classify.py output file
        try:
            with open("output/evaluation_results.json", "r") as f:
                eval_results = json.load(f)
                accuracy = eval_results.get("accuracy", 0)
                test_samples = eval_results.get("tested_samples", 0 )
        except:
            print("[LOG][ERROR] Failed to read evaluation results.")
            accuracy = 0
        print(f"[LOG][INFO] Model accuracy: {accuracy}%")
        return accuracy, model_filename, test_samples

# **FINE-TUNING**
if args.fine_tune and args.evaluate:
    if args.solver is None:
        default_solver = "liblinear"
    else:
        default_solver = args.solver

    best_params = {
        "C": args.C if args.C is not None else defaults["C"],
        "weight": args.weight if args.weight is not None else defaults["weight"],
        "max_iter": args.max_iter if args.max_iter is not None else defaults["max_iter"],
        "min_df": args.min_df if args.min_df is not None else defaults["min_df"],
        "max_df": args.max_df if args.max_df is not None else defaults["max_df"],
        "solver": default_solver
    }

    generated_models = [] 

    test_samples = 0
    
    best_accuracy, best_model, test_samples = train_and_evaluate(best_params)

    generated_models.append(best_model)

    def delete_suboptimal_models():
        """
        Deletes suboptimal model files generated during the current run.
        """
        for model in generated_models:
            if model != best_model:
                generated_models.remove(model)
                os.remove(model)
                print(f"[LOG][INFO] Deleted suboptimal model: {model}")

    print(f"[LOG][INFO] Baseline Accuracy: {best_accuracy}%")

    if args.solver is None:
        for solver in ["lbfgs", "saga", "newton-cg"]:
            test_params = best_params.copy()
            test_params["solver"] = solver
            test_accuracy, model, test_samples = train_and_evaluate(test_params)
            if test_accuracy > best_accuracy:
                    print(f"[LOG][INFO] {solver} solver has higher accuracy. (Accuracy: {test_accuracy}%)")
                    best_accuracy = test_accuracy
                    best_params["solver"] = solver

    for param in ["C", "weight", "min_df", "max_df", "max_iter"]:
        print(f"[LOG][INFO] Fine-tuning {param} parameter.")
        if getattr(args, param.replace("-", "_"), None) is not None:
            continue  # Skip fine-tuning if explicitly set

        step_size = steps[param]
        min_limit, max_limit = limits[param]
        direction = 1  # Start increasing
        step_factor = 1.0  # First step is always 1 * step_size
        skip_steps = False

        for i in range(args.fine_tune):
            if skip_steps:
                print(f"[LOG][INFO] Skipping step: fine-tuning this parameter further yields no improvement in either direction.")
                continue
            test_value = best_params[param] + (step_size * step_factor * direction)
            if param == "min_df" or param == "max_iter":
                test_value = math.ceil(test_value)
            print(f"[LOG][INFO] {param} parameter fine-tuning step {i+1} of {args.fine_tune} with parameter value {test_value}.")
            # **Skip if test_value exceeds limits**
            if (min_limit != -1 and test_value < min_limit) or (max_limit != -1 and test_value > max_limit):
                print(f"[LOG][WARNING] Skipping {param}={test_value} (Out of Bounds)")
                continue

            test_params = best_params.copy()
            test_params[param] = test_value

            test_accuracy, test_model, test_samples = train_and_evaluate(test_params)
            generated_models.append(test_model)

            if test_accuracy > best_accuracy:
                print(f"[LOG][INFO] {param} improved! New best: {test_value} (Accuracy: {test_accuracy}%)")
                best_accuracy = test_accuracy
                best_model = test_model
                best_params[param] = test_value
                step_factor = math.log(i + 2)  # Increase step factor logarithmically
            elif direction == 1 and i == 0:
                direction = -1  # Switch to decreasing
                step_factor = 1.0  # Reset step factor
                test_value = defaults[param] - (step_size * step_factor)
                if param == "min_df" or param == "max_iter":
                    test_value = math.ceil(test_value)
                print(f"[LOG][INFO] No improvement, switching direction. Testing {param}={test_value}")

                # **Skip if test_value exceeds limits**
                if (min_limit != -1 and test_value < min_limit) or (max_limit != -1 and test_value > max_limit):
                    print(f"[LOG][WARNING] Skipping {param}={test_value} (Out of Bounds)")
                    continue

                test_params[param] = test_value
                test_accuracy, test_model, test_samples = train_and_evaluate(test_params)
                generated_models.append(test_model)

                if test_accuracy > best_accuracy:
                    print(f"[LOG][INFO] {param} decreased and improved! New best: {test_value} (Accuracy: {test_accuracy}%)")
                    best_accuracy = test_accuracy
                    best_model = test_model
                    best_params[param] = test_value                  
                    step_factor = math.log(i + 2)  # Increase step factor logarithmically
                else:
                    step_factor = 1.0  # Reset if no improvement
            else:
                print(f"[LOG][INFO] No improvement, setting step skipping. Optimum tuning for {param} is likely between {best_params[param]} and {test_value}")
                skip_steps = True
                optimal[param][0] = best_params[param]
                optimal[param][1] = test_value
                continue

        delete_suboptimal_models()

    print(f"[LOG][INFO] Best model retained: {best_model} with accuracy {best_accuracy}%.")

    model_meta = {
        "model": best_model,
        "evaluation": {
            "test_samples": test_samples,
            "accuracy": str(best_accuracy) + "%"},
        "hyperparameters": best_params,
        "fine_tuning_recommendation": optimal
    }

    # Write model metadata to a JSON file
    try:
        meta_filename = f"output/{best_model}_meta.json"
        with open(meta_filename, "w") as f:
            json.dump(model_meta, f, indent=4)
        print(f"[LOG][INFO] Model metadata saved as: {meta_filename}")
    except Exception as e:
        print(f"[LOG][ERROR] Failed to save model metadata: {e}")
    
else:
    # Train normally without fine-tuning
    final_accuracy, model_name, _ = train_and_evaluate(defaults)
    print(f"[LOG][INFO] Model name: {model_name}, accuracy: {final_accuracy}%")
