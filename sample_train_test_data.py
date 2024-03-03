import pandas as pd
import urllib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import os
import glob
import sys
import pickle
import numpy as np
from tqdm.auto import tqdm
import math

tqdm.pandas()


def sample_train_set(
    feature,
    values_groups,
    shared=True,
    seed=42,
    min_count=6,
    input_data_dir="../",
    output_data_dir="../",
):
    """Sample train set for the given feature making sure the same number of annotators per community is preserved for the same comments

    Args:
        feature (str): which feature to use (gender, english_first_language, age_group, education)
        values_groups (dict): grouping the communities depending on feature values. e.g. : {'Under 18': 'under_30', '18-30' : 'under_30' , '30-45' : 'over_30' , '45-60' : 'over_30' , 'Over 60' : 'over_30'}
        shared (boolean, optional): Whether the comments are shared among training sets. Defaults to True.
        seed (int, optional): a randomization seed. Defaults to 42.
        min_count (int, optional): min number of annotators per community. Defaults to 3.
        input_data_dir (str, optional): where to load data files from. Defaults to "../".
        output_data_dir (str, optional): where to write output csv to. Defaults to "../".
    """
    np.random.seed(int(seed))

    # Load data
    comments = pd.read_csv(
        f"{input_data_dir}attack_annotated_comments.tsv", sep="\t"
    )
    annotations = pd.read_csv(
        f"{input_data_dir}attack_annotations.tsv", sep="\t"
    )
    annotators = pd.read_csv(
        f"{input_data_dir}attack_worker_demographics.tsv", sep="\t"
    )

    test_set = pd.read_csv(f"{output_data_dir}test_{feature}.csv")

    values = set(values_groups.values())

    # merge annotators with comments
    comments_with_annotators = annotations.merge(
        annotators, on="worker_id", how="left"
    )
    # map feature values
    comments_with_annotators[feature] = comments_with_annotators[feature].map(
        values_groups
    )
    comments_with_annotators = comments_with_annotators[
        comments_with_annotators[feature].isin(values)
    ]

    # count number of annotators per feature group
    feature_counts_df = (
        comments_with_annotators.groupby(["rev_id", feature])["worker_id"]
        .count()
        .reset_index(name="count")
        .pivot("rev_id", feature, "count")
        .fillna(0)[values]
    )
    if shared:
        query = " & ".join([f"{value} >={min_count}" for value in values])
        feature_counts_df = feature_counts_df.query(query)

        comments = comments[comments["rev_id"].isin(feature_counts_df.index)]
        comments = comments[~comments["rev_id"].isin(test_set.rev_id)]
        # comments = comments.query("split=='train'")
        annotations = annotations[
            annotations["rev_id"].isin(comments["rev_id"])
        ]
        print("train data len = ", len(annotations["rev_id"].unique()))

        # merge annotators with comments
        comments_with_annotators = annotations.merge(
            annotators, on="worker_id", how="left"
        )
        comments_with_annotators[feature] = comments_with_annotators[
            feature
        ].map(values_groups)
        comments_with_annotators = comments_with_annotators[
            comments_with_annotators[feature].isin(values)
        ]
        groups = comments_with_annotators.sample(frac=1).groupby(
            ["rev_id", feature]
        )
        feature_groups = groups.head(math.ceil(min_count))

        mixed_mean = (
            groups.head(math.ceil(min_count / len(values)))
            .groupby(["rev_id"])["attack"]
            .mean()
            .reset_index(name="mean")
        )
        features_mean = (
            feature_groups.groupby(["rev_id", feature])["attack"]
            .mean()
            .reset_index(name="mean")
            .pivot("rev_id", feature, "mean")
        )
        features_mean = features_mean[values]
        comments = mixed_mean.merge(comments, on="rev_id", how="left").rename(
            columns={"mean": f"{feature}_mixed_attack"}
        )
        comments = features_mean.merge(
            comments, on="rev_id", how="left"
        ).rename(columns={v: f"{feature}_{v}_attack" for v in values})
        values = list(values) + ["mixed"]

        for value in values:
            comments = comments[comments[f"{feature}_{value}_attack"] != -1]

        for value in values:
            comments[f"{feature}_{value}_attack"] = np.where(
                comments[f"{feature}_{value}_attack"] > 0.5, True, False
            )

        comments["comment"] = comments["comment"].apply(
            lambda x: x.replace("NEWLINE_TOKEN", " ")
        )
        comments["comment"] = comments["comment"].apply(
            lambda x: x.replace("TAB_TOKEN", " ")
        )

        print("train data size  = ", len(comments))
        for v in values:
            fname = f"{output_data_dir}train_{feature}_{v}_{seed}.csv"
            comments.to_csv(fname)
            print(f"saved {v} train data to ", fname)

    else:
        feature_comments = {}
        values = list(values) + ["mixed"]
        for v in values:
            if v != "mixed":
                feature_df = feature_counts_df.query(f"{v} >={min_count}")
                train_comments = comments[
                    comments["rev_id"].isin(feature_df.index)
                ]
            train_comments = train_comments[
                ~train_comments["rev_id"].isin(test_set.rev_id)
            ]
            train_annotations = annotations[
                annotations["rev_id"].isin(train_comments["rev_id"])
            ]

            train_comments_with_annotators = train_annotations.merge(
                annotators, on="worker_id", how="left"
            )
            train_comments_with_annotators[
                feature
            ] = train_comments_with_annotators[feature].map(values_groups)
            train_comments_with_annotators = train_comments_with_annotators[
                train_comments_with_annotators[feature].isin(values)
            ]

            groups = train_comments_with_annotators.sample(frac=1).groupby(
                ["rev_id", feature]
            )
            if v == "mixed":
                mixed_mean = (
                    groups.head(math.ceil(min_count / len(values)))
                    .groupby(["rev_id"])["attack"]
                    .mean()
                    .reset_index(name="mean")
                )
                train_comments = mixed_mean.merge(
                    train_comments, on="rev_id", how="left"
                ).rename(columns={"mean": f"{feature}_mixed_attack"})
            else:
                feature_groups = groups.head(math.ceil(min_count))

                features_mean = (
                    feature_groups.groupby(["rev_id", feature])["attack"]
                    .mean()
                    .reset_index(name="mean")
                    .pivot("rev_id", feature, "mean")
                )
                features_mean = features_mean[[v]]
                train_comments = features_mean.merge(
                    train_comments, on="rev_id", how="left"
                ).rename(columns={v: f"{feature}_{v}_attack"})
            train_comments = train_comments[
                train_comments[f"{feature}_{v}_attack"] != -1
            ]

            train_comments[f"{feature}_{v}_attack"] = np.where(
                train_comments[f"{feature}_{v}_attack"] > 0.5, True, False
            )

            train_comments["comment"] = train_comments["comment"].apply(
                lambda x: x.replace("NEWLINE_TOKEN", " ")
            )
            train_comments["comment"] = train_comments["comment"].apply(
                lambda x: x.replace("TAB_TOKEN", " ")
            )
            feature_comments[v] = train_comments
        min_size = min(
            [len(comments) for comments in feature_comments.values()]
        )
        print("train data size  = ", min_size)
        feature_comments = {
            v: comments[:min_size]
            for v, comments in feature_comments.items()
        }
        for v, comments in feature_comments.items():
            fname = f"{output_data_dir}train_{feature}_{v}_{seed}.csv"
            comments.to_csv(fname)
            print(f"saved {v} train data to ", fname)


def sample_test_set(
    feature,
    values_groups,
    min_count=3,
    input_data_dir="../",
    output_data_dir="../",
):
    """Sample test set for the given feature making sure the same number of annotators per community is preserved.
    A csv file is saved in the output_data_dir, columns are added with the format {feature}_{value}_attack.

    Args:
        feature (str): which feature to use (gender, english_first_language, age_group, education)
        values_groups (dict): grouping the communities depending on feature values. e.g. : {'Under 18': 'under_30', '18-30' : 'under_30' , '30-45' : 'over_30' , '45-60' : 'over_30' , 'Over 60' : 'over_30'}
        min_count (int, optional): min number of annotators per community. Defaults to 3.
        input_data_dir (str, optional): where to load data files from. Defaults to "../".
        output_data_dir (str, optional): where to write output csv to. Defaults to "../".
    """
    print(f"Sampling test data for {feature}....")
    # load data and demographics of workers
    comments = pd.read_csv(
        f"{input_data_dir}attack_annotated_comments.tsv", sep="\t"
    )
    annotations = pd.read_csv(
        f"{input_data_dir}attack_annotations.tsv", sep="\t"
    )
    annotators = pd.read_csv(
        f"{input_data_dir}attack_worker_demographics.tsv", sep="\t"
    )
    # remove all training data, leave test data
    # test_comments = comments.query("split=='test'")
    test_comments = comments
    test_annotations = annotations[
        annotations["rev_id"].isin(test_comments["rev_id"])
    ]

    values = set(values_groups.values())

    # merge annotators with comments
    comments_with_annotators = test_annotations.merge(
        annotators, on="worker_id", how="left"
    )
    comments_with_annotators[feature] = comments_with_annotators[feature].map(
        values_groups
    )

    feature_counts_df = (
        comments_with_annotators.groupby(["rev_id", feature])["worker_id"]
        .count()
        .reset_index(name="count")
        .pivot("rev_id", feature, "count")
        .fillna(0)[values]
    )
    query = " & ".join([f"{value} >={min_count}" for value in values])
    feature_counts_df = feature_counts_df.query(query)

    test_comments = test_comments[
        test_comments["rev_id"].isin(feature_counts_df.index)
    ]
    test_comments = test_comments.sample(frac=0.2)

    test_annotations = annotations[
        annotations["rev_id"].isin(test_comments["rev_id"])
    ]

    # merge annotators with comments
    comments_with_annotators = test_annotations.merge(
        annotators, on="worker_id", how="left"
    )
    comments_with_annotators[feature] = comments_with_annotators[feature].map(
        values_groups
    )
    comments_with_annotators = comments_with_annotators[
        comments_with_annotators[feature].isin(values)
    ]
    groups = comments_with_annotators.sample(frac=1).groupby(
        ["rev_id", feature]
    )
    mixed_mean = (
        groups.head(math.ceil(min_count / len(values)))
        .groupby(["rev_id"])["attack"]
        .mean()
        .reset_index(name="mean")
    )
    features_mean = (
        groups.head(min_count)
        .groupby(["rev_id", feature])["attack"]
        .mean()
        .reset_index(name="mean")
        .pivot("rev_id", feature, "mean")
    )
    features_mean = features_mean[values]

    test_comments = mixed_mean.merge(
        test_comments, on="rev_id", how="left"
    ).rename(columns={"mean": f"{feature}_mixed_attack"})
    test_comments = features_mean.merge(
        test_comments, on="rev_id", how="left"
    ).rename(columns={v: f"{feature}_{v}_attack" for v in values})
    values = list(values) + ["mixed"]

    for value in values:
        test_comments = test_comments[
            test_comments[f"{feature}_{value}_attack"] != -1
        ]

    for value in values:
        test_comments[f"{feature}_{value}_attack"] = np.where(
            test_comments[f"{feature}_{value}_attack"] > 0.5, True, False
        )

    test_comments["comment"] = test_comments["comment"].apply(
        lambda x: str(x).replace("NEWLINE_TOKEN", " ")
    )
    test_comments["comment"] = test_comments["comment"].apply(
        lambda x: str(x).replace("TAB_TOKEN", " ")
    )
    fname = f"{output_data_dir}test_{feature}.csv"
    test_comments.to_csv(fname)
    print("saved to ", fname)
