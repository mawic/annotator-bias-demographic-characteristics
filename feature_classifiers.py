import glob
import os
import pickle
import sys
import urllib

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    f1_score,
    classification_report,
)
from sklearn.pipeline import Pipeline
import numpy as np
from tqdm.auto import tqdm
import pickle
import math
import krippendorff
from distilbert import DistilBertTrain

tqdm.pandas()


def create_feature_classifier(
    feature,
    values_groups,
    seed,
    min_count=1,
    data_dir="../",
    model_save_dir="../"
):

    print(f"Creating classifiers for {feature} with seed {seed}")

    test_detox = pd.read_csv(f"{data_dir}test_{feature}.csv")
    print(f"test data len = {len(test_detox)}")
    values = set(values_groups.values())
    values = list(values) + ["mixed"]
    results = {}
    for clf_value in values:
        comments = pd.read_csv(f"{data_dir}train_{feature}_{clf_value}_{seed}.csv")
        results[clf_value] = {}
        print(f"Fitting {clf_value} classifier with seed {seed}")

        clf = DistilBertTrain(
            comments["comment"].tolist(), 
            comments[f"{feature}_{clf_value}_attack"].tolist()
        )
        clf.fit()

        filename = f"{model_save_dir}{feature}_{clf_value}_clf_{seed}.model"
        pickle.dump(clf, open(filename, "wb"))
        clf.save(filename)

        for test_value in values:
            print(f"     Testing on {test_value}")
            test_pred = clf.predict(test_detox["comment"].tolist())
            test_true = test_detox[f"{feature}_{test_value}_attack"]
            auc = roc_auc_score(
                test_detox[f"{feature}_{test_value}_attack"],
                test_pred
            )
            f1 = f1_score(test_true, test_pred)
            report = classification_report(
                test_true, test_pred, output_dict=True
            )
            
            print("     ROC AUC Score", auc)
            print("     F1 score", f1)
            print(
                "=========================================================================="
            )
            results[clf_value][test_value] = {
                "sensitivity": report["True"]["recall"],
                "specificity": report["False"]["recall"],
                "f1": f1,
                "auc": auc,
            }
    return results
