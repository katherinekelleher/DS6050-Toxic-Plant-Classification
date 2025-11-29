"""
train_rf.py

Train a Random Forest classifier on image‑derived features, optionally tune hyperparameters,
save the final model to disk, and optionally report evaluation metrics.
"""

import os
import argparse
import joblib  # recommended for sklearn model persistence
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from features.rf_features import read_images_from_paths, store_features_labels

def parse_args():
    parser = argparse.ArgumentParser(description="Train & save Random Forest classifier (classical ML).")
    parser.add_argument('--meta_csv', type=str, required=True,
                        help="Path to metadata CSV (with image paths & labels).")
    parser.add_argument('--image_root', type=str, required=True,
                        help="Root directory where images are stored.")
    parser.add_argument('--output_model', type=str, default='rf_model.joblib',
                        help="Filename for saving the trained model.")
    parser.add_argument('--n_iter', type=int, default=50,
                        help="Number of iterations for RandomizedSearchCV (tuning).")
    parser.add_argument('--test_size', type=float, default=0.3,
                        help="Fraction of data to reserve for test/validation.")
    parser.add_argument('--random_state', type=int, default=42)
    return parser.parse_args()

def main():
    args = parse_args()

    # Load metadata
    meta = pd.read_csv(args.meta_csv)
    train_meta, test_meta = train_test_split(
        meta, test_size=args.test_size, stratify=meta['toxicity'], random_state=args.random_state
    )

     # Prepend "../" to every path string
    train_meta['path'] = train_meta['path'].apply(lambda p: os.path.join(os.pardir, p))
    test_meta['path'] = test_meta['path'].apply(lambda p: os.path.join(os.pardir, p))

    # Build training data
    paths_train = train_meta['path'].tolist()
    labels_train = train_meta['toxicity'].tolist()
    imgs_train = read_images_from_paths(paths_train, base_path=args.image_root)
    X_train, y_train = store_features_labels(imgs_train, labels_train)

    # Build test data
    paths_test = test_meta['path'].tolist()
    labels_test = test_meta['toxicity'].tolist()
    imgs_test = read_images_from_paths(paths_test, base_path=args.image_root)
    X_test, y_test = store_features_labels(imgs_test, labels_test)

    print("Data shapes — X_train:", X_train.shape, "X_test:", X_test.shape)

    # Train baseline RF
    clf = RandomForestClassifier(n_estimators=100, random_state=args.random_state)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    print("Baseline RF — accuracy:", accuracy_score(y_test, y_pred))
    print("Baseline classification report:\n", classification_report(y_test, y_pred))

    # Hyperparameter tuning
    param_dist = {
        'n_estimators': [100, 200, 300, 400],
        'max_depth': [None, 10, 20, 30, 40],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', 0.3, 0.5],
        'bootstrap': [True, False]
    }
    rf = RandomForestClassifier(random_state=args.random_state, n_jobs=-1)
    search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_dist,
        n_iter=args.n_iter,
        cv=5,
        scoring='accuracy',
        random_state=args.random_state,
        verbose=2
    )
    search.fit(X_train, y_train)
    best_rf = search.best_estimator_
    print("Best params:", search.best_params_)

    # Evaluate tuned model
    y_pred_tuned = best_rf.predict(X_test)
    y_proba_tuned = best_rf.predict_proba(X_test)[:, 1]
    print("Tuned RF — accuracy:", accuracy_score(y_test, y_pred_tuned))
    print("Tuned classification report:\n", classification_report(y_test, y_pred_tuned))
    fpr, tpr, _ = roc_curve(y_test, y_proba_tuned)
    print("Tuned RF — AUC:", auc(fpr, tpr))

    # Save the tuned model
    joblib.dump(best_rf, args.output_model)
    print(f"Trained RF model saved to {args.output_model}")

if __name__ == '__main__':
    main()
