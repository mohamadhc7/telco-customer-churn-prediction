"""Reproducible churn evaluation. Run: python evaluate.py"""
from pathlib import Path
import hashlib
import json
import platform

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
    ConfusionMatrixDisplay, f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
SEED = 42
NUMERIC = ['tenure', 'MonthlyCharges', 'TotalCharges']

def load_data():
    path = ROOT / 'WA_Fn-UseC_-Telco-Customer-Churn.csv'
    df = pd.read_csv(path)
    assert df.customerID.is_unique, 'Customer IDs must be unique before splitting.'
    assert set(df.Churn.unique()) == {'Yes', 'No'}
    # Stateless type conversion learns no values from the data.
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    X = df.drop(columns=['customerID', 'Churn'])
    y = df.Churn.map({'No': 0, 'Yes': 1})
    return X, y, path

def build_models(X):
    categorical = [c for c in X.columns if c not in NUMERIC]
    preprocess = ColumnTransformer([
        ('numeric', Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('scale', StandardScaler()),
        ]), NUMERIC),
        ('categorical', Pipeline([
            ('impute', SimpleImputer(strategy='most_frequent')),
            ('encode', OneHotEncoder(handle_unknown='ignore', drop='first')),
        ]), categorical),
    ])
    estimators = {
        'Majority baseline': DummyClassifier(strategy='most_frequent'),
        'Logistic Regression': LogisticRegression(max_iter=2000, random_state=SEED),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=1),
        'Balanced Random Forest': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=SEED, n_jobs=1),
    }
    return {name: Pipeline([('preprocess', clone(preprocess)), ('model', estimator)])
            for name, estimator in estimators.items()}

def run_evaluation():
    results = ROOT / 'results'
    results.mkdir(exist_ok=True)
    X, y, path = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)
    assert set(X_train.index).isdisjoint(X_test.index)
    assert len(X_train) + len(X_test) == len(X)
    assert 'customerID' not in X.columns and 'Churn' not in X.columns
    folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(X_train, y_train))
    for fit_idx, validation_idx in folds:
        assert set(fit_idx).isdisjoint(validation_idx)
        assert set(y_train.iloc[fit_idx]) == {0, 1}
        assert set(y_train.iloc[validation_idx]) == {0, 1}
    models = build_models(X)
    scoring = {'average_precision': 'average_precision', 'roc_auc': 'roc_auc',
               'precision': 'precision', 'recall': 'recall', 'f1': 'f1'}
    # Precision is undefined for an all-negative dummy classifier. Report 0 explicitly.
    from sklearn.metrics import make_scorer
    scoring['precision'] = make_scorer(precision_score, zero_division=0)
    scoring['f1'] = make_scorer(f1_score, zero_division=0)
    rows = []
    for name, pipeline in models.items():
        print('Cross-validating:', name, flush=True)
        scores = cross_validate(pipeline, X_train, y_train, cv=folds, scoring=scoring,
                                n_jobs=1, error_score='raise')
        row = {'model': name}
        for metric in scoring:
            row[f'{metric}_mean'] = float(scores[f'test_{metric}'].mean())
            row[f'{metric}_std'] = float(scores[f'test_{metric}'].std(ddof=1))
        rows.append(row)
    cv_results = pd.DataFrame(rows)
    cv_results.to_csv(results / 'cross_validation.csv', index=False)
    # Prespecified rule: choose by mean CV average precision, tie-break with F1.
    # The holdout is not used to select the model or the decision threshold.
    selected = cv_results.sort_values(['average_precision_mean', 'f1_mean'], ascending=False).iloc[0]['model']
    final_model = clone(models[selected]).fit(X_train, y_train)
    numeric = final_model.named_steps['preprocess'].named_transformers_['numeric']
    np.testing.assert_allclose(numeric.named_steps['impute'].statistics_, X_train[NUMERIC].median().values)
    assert int(numeric.named_steps['scale'].n_samples_seen_) == len(X_train)
    expected_mean = X_train[NUMERIC].fillna(X_train[NUMERIC].median()).mean().values
    np.testing.assert_allclose(numeric.named_steps['scale'].mean_, expected_mean)
    probabilities = final_model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, predictions, labels=[0, 1]).ravel()
    metrics = {
        'selected_model': selected, 'decision_threshold': 0.5,
        'accuracy': accuracy_score(y_test, predictions),
        'precision': precision_score(y_test, predictions, zero_division=0),
        'recall': recall_score(y_test, predictions), 'f1': f1_score(y_test, predictions),
        'average_precision': average_precision_score(y_test, probabilities),
        'roc_auc': roc_auc_score(y_test, probabilities),
        'true_negatives': int(tn), 'false_positives': int(fp),
        'false_negatives': int(fn), 'true_positives': int(tp),
    }
    (results / 'holdout_metrics.json').write_text(json.dumps(metrics, indent=2)+'\n')
    metadata = {
        'seed': SEED, 'rows': len(X), 'train_rows': len(X_train), 'test_rows': len(X_test),
        'train_churn_count': int(y_train.sum()), 'test_churn_count': int(y_test.sum()),
        'missing_total_charges': int(X.TotalCharges.isna().sum()), 'cv_folds': 5,
        'selection_metric': 'mean training-fold average precision; F1 tie-break',
        'data_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'train_index_sha256': hashlib.sha256(X_train.index.to_numpy().tobytes()).hexdigest(),
        'test_index_sha256': hashlib.sha256(X_test.index.to_numpy().tobytes()).hexdigest(),
        'python': platform.python_version(), 'sklearn': sklearn.__version__,
        'pandas': pd.__version__, 'numpy': np.__version__, 'matplotlib': matplotlib.__version__,
        'checks': ['unique customers', 'disjoint stratified train/test', 'disjoint CV folds within training',
                   'target and ID excluded', 'imputer fitted on training medians', 'scaler fitted on training rows only'],
    }
    (results / 'run_metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ConfusionMatrixDisplay.from_predictions(y_test, predictions, display_labels=['Stayed', 'Churned'],
                                           colorbar=False, cmap='Blues', ax=ax)
    ax.set_title(f'{selected}\nStratified holdout, threshold 0.50')
    fig.tight_layout(); fig.savefig(results / 'confusion_matrix.png', dpi=160); plt.close(fig)
    print(json.dumps(metrics, indent=2), flush=True)
    return cv_results, metrics, metadata

if __name__ == '__main__':
    run_evaluation()
