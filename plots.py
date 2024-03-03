import matplotlib.pyplot as plt
import pandas as pd

import numpy as np
from wordcloud import WordCloud, STOPWORDS


def plot_classifier_results(results, values_groups, markers):
    values = list(set(values_groups.values())) + ["mixed"]
    for test_data in values:
        sp = {v: [] for v in values}
        sn = {v: [] for v in values}

        for train_data in values:
            for i in range(len(results)):
                sp[train_data].append(
                    results[i][train_data][test_data]["specificity"]
                )
                sn[train_data].append(
                    results[i][train_data][test_data]["sensitivity"]
                )

        for train_data in values:
            plt.scatter(
                sp[train_data],
                sn[train_data],
                label=f"{train_data} train data",
                color=markers[train_data][1],
                marker=markers[train_data][0],
            )

        plt.xlabel("Specificity")
        plt.ylabel("Sensitivity")
        plt.title(test_data)
        plt.legend()

        plt.show()


def analysis_plots(feature, values_groups, dataset, data_dir="../"):
    comments = pd.read_csv(
        f"{data_dir}attack_annotated_comments.tsv", sep="\t"
    )
    annotations = pd.read_csv(f"{data_dir}attack_annotations.tsv", sep="\t")
    annotators = pd.read_csv(
        f"{data_dir}/attack_worker_demographics.tsv", sep="\t"
    )

    values = set(values_groups.values())
    comments_with_annotators = annotations.merge(
        annotators, on="worker_id", how="left"
    )
    comments_with_annotators[feature] = comments_with_annotators[feature].map(
        values_groups
    )
    all_feature_counts_df = (
        comments_with_annotators.groupby(["rev_id", feature])["worker_id"]
        .count()
        .reset_index(name="count")
        .pivot("rev_id", feature, "count")
        .fillna(0)[values]
    )

    query = " & ".join([f"{value} >=1" for value in values])

    feature_counts_df = all_feature_counts_df.query(query)

    comments = comments[comments["rev_id"].isin(feature_counts_df.index)]
    if dataset:
        comments = comments.query(f"split=='{dataset}'")
    annotations = annotations[annotations["rev_id"].isin(comments["rev_id"])]
    print(
        f"Number of {dataset} comments : ", len(annotations["rev_id"].unique())
    )

    # merge annotators with comments
    comments_with_annotators = annotations.merge(
        annotators, on="worker_id", how="left"
    )
    comments_with_annotators[feature] = comments_with_annotators[feature].map(
        values_groups
    )

    mixed_mean = (
        comments_with_annotators.groupby(["rev_id"])["attack"]
        .mean()
        .reset_index(name="mean")
    )
    features_mean = (
        comments_with_annotators.groupby(["rev_id", feature])["attack"]
        .mean()
        .reset_index(name="mean")
        .pivot("rev_id", feature, "mean")
    )
    features_mean = features_mean[values]
    comments = mixed_mean.merge(comments, on="rev_id", how="left").rename(
        columns={"mean": f"{feature}_mixed_attack"}
    )
    comments = features_mean.merge(comments, on="rev_id", how="left").rename(
        columns={v: f"{feature}_{v}_attack" for v in values}
    )

    print("Histograms")
    for value in values:
        print(f"{feature} {value}")
        all_feature_counts_df[value].hist(
            bins=int(all_feature_counts_df[value].max())
        )
        plt.show()

    print("============================================")
    print()
    for value1 in values:
        value1_comments = comments[
            comments[f"{feature}_{value1}_attack"] == True
        ]
        print(
            f"total number of comments that {value1} finds hateful {len(value1_comments)}"
        )
        for value2 in values:
            if value1 != value2:
                value2_comments = comments[
                    comments[f"{feature}_{value2}_attack"] == True
                ]
                intersection = value2_comments[
                    value2_comments["rev_id"].isin(value1_comments["rev_id"])
                ]
                print(
                    f"     among these, {value2} finds {len(intersection)} also hateful"
                )
        print()
    print("============================================")

    values = list(values) + ["mixed"]

    for value in values:
        comments = comments[comments[f"{feature}_{value}_attack"] != -1]

    for value in values:
        comments[f"{feature}_{value}_attack"] = np.where(
            comments[f"{feature}_{value}_attack"] > 0.5, True, False
        )

    print("Value counts:")
    for value in values:
        print(f"{feature} {value}")
        plt.pie(
            comments[f"{feature}_{value}_attack"].value_counts(),
            labels=["False", "True"],
            autopct="%1.1f%%",
            shadow=True,
            startangle=90,
        )
        plt.show()

    comments["comment"] = comments["comment"].apply(
        lambda x: x.replace("NEWLINE_TOKEN", " ")
    )
    comments["comment"] = comments["comment"].apply(
        lambda x: x.replace("TAB_TOKEN", " ")
    )
    print("Word clouds:")
    for value in values:
        print(f"{feature} {value}")
        text = comments[comments[f"{feature}_{value}_attack"] == True][
            "comment"
        ].values
        text = " ".join(text)
        wordcloud = WordCloud().generate(text)
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.show()
