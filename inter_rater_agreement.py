import pandas as pd
import krippendorff
import math
import numpy as np


def calculate_inter_rate_agreement(
    feature, values_groups, input_data_dir, min_count=6
):
    comments = pd.read_csv(
        f"{input_data_dir}attack_annotated_comments.tsv", sep="\t"
    )
    annotations = pd.read_csv(
        f"{input_data_dir}attack_annotations.tsv", sep="\t"
    )
    annotators = pd.read_csv(
        f"{input_data_dir}attack_worker_demographics.tsv", sep="\t"
    )

    values = set(values_groups.values())
    comments_with_annotators = annotations.merge(
        annotators, on="worker_id", how="left"
    )
    comments_with_annotators[feature] = comments_with_annotators[feature].map(
        values_groups
    )
    # count number of annotators per feature group
    feature_counts_df = (
        comments_with_annotators.groupby(["rev_id", feature])["worker_id"]
        .count()
        .reset_index(name="count")
        .pivot("rev_id", feature, "count")
        .fillna(0)[values]
    )
    feature_comments = {}
    for v in values_groups.values():
        feature_df = feature_counts_df.query(f"{v} >={min_count}")
        train_comments = comments[comments["rev_id"].isin(feature_df.index)]
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

        feature_groups = groups.head(math.ceil(min_count))

        feature_comments[v] = feature_groups
    min_size = min([len(comments) for comments in feature_comments.values()])
    print("train data size  = ", min_size)
    feature_comments = {
        v: comments[:min_size] for v, comments in feature_comments.items()
    }
    mixed_data = pd.DataFrame([])
    for value, feature_groups in feature_comments.items():
        print(f'{feature}=="{value}"')
        reliability_data = (
            feature_groups.query(f'{feature}=="{value}"')[
                ["rev_id", "worker_id", "attack"]
            ]
            .pivot("worker_id", columns="rev_id", values="attack")
            .values
        )
        mixed_data = pd.concat(
            [
                mixed_data,
                feature_groups.query(f'{feature}=="{value}"')[
                    ["rev_id", "worker_id", "attack"]
                ],
            ]
        )
        print(
            f"Inter-rater agreement of {value} = ",
            krippendorff.alpha(
                reliability_data=reliability_data,
                level_of_measurement="nominal",
            ),
        )
    reliability_data = (
        mixed_data.sample(int(len(mixed_data) / 2))
        .pivot("worker_id", columns="rev_id", values="attack")
        .values
    )
    print(
        f"Inter-rater agreement of both = ",
        krippendorff.alpha(
            reliability_data=reliability_data, level_of_measurement="nominal",
        ),
    )

