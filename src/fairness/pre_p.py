import argparse
from scipy import stats
import random
from typing import *
from collections import Counter
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors as KNN
from sklearn.linear_model import LogisticRegression
from aif360.algorithms import Transformer as AifTransformer
from aif360.datasets import BinaryLabelDataset
from datetime import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


class Fairway(AifTransformer):
    """
    Paper: J Chakraborty, et al. 2020. Fairway: a way to build fair ML software. ESEC/FSE 2020.
    """

    def __init__(
            self, sensitive_name: str, label_name: str, LR_clf_params: dict,
            verbose: bool = False) -> None:
        super().__init__(sensitive_name=sensitive_name,
                         label_name=label_name,
                         LR_clf_params=LR_clf_params,
                         verbose=verbose)
        self.sensitive_name = sensitive_name
        self.label_name = label_name
        self.LR_clf_params = LR_clf_params
        # LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=100)
        self.priviledged_clf = LogisticRegression(**LR_clf_params)
        self.unpriviledged_clf = LogisticRegression(**LR_clf_params)

    def fit_transform(self, dataset: BinaryLabelDataset) -> BinaryLabelDataset:
        origin_df, _ = dataset.convert_to_dataframe()
        # 1. split dataset into 2 groups by sensitive (privileged, favorable)
        priv_df = origin_df[origin_df[self.sensitive_name] == 1]
        priv_df = priv_df.sample(frac=1).reset_index(drop=True)
        unpriv_df = origin_df[origin_df[self.sensitive_name] == 0]
        unpriv_df = unpriv_df.sample(frac=1).reset_index(drop=True)

        # 2. train 2 classifiers
        priv_X = priv_df.drop([self.label_name, self.sensitive_name], axis=1).values
        priv_y = priv_df[self.label_name].values
        unpriv_X = unpriv_df.drop([self.label_name, self.sensitive_name], axis=1).values
        unpriv_y = unpriv_df[self.label_name].values
        self.priviledged_clf.fit(priv_X, priv_y)
        self.unpriviledged_clf.fit(unpriv_X, unpriv_y)

        # 3. predict labels for all samples
        clf_inputs = origin_df.drop([self.label_name, self.sensitive_name], axis=1).values
        priv_preds = self.priviledged_clf.predict(clf_inputs)
        unpriv_preds = self.unpriviledged_clf.predict(clf_inputs)
        tweaked_df = origin_df.iloc[(priv_preds == unpriv_preds), :]

        # 4. convert to BinaryLabelDataset
        tweaked_dataset = BinaryLabelDataset(
            favorable_label=1,
            unfavorable_label=0,
            df=tweaked_df,
            label_names=[self.label_name, ],
            protected_attribute_names=[self.sensitive_name, ]
        )
        return tweaked_dataset


class FairSmote(AifTransformer):
    """
    Paper: J Chakraborty, et al. 2020. Bias in machine learning software: why? how? what to do? ESEC/FSE 2021.
    """

    def __init__(
            self, sensitive_name: str, label_name: str, cate_cols: List[str] = [],
            num_cols: List[str] = [],
            verbose: bool = False) -> None:
        """
        :param auto: if True, all minority classes will be over-sampled as much as majority class
        :param up_to_num: size of minorities to over-sampling, only valid when auto is True
        :param cate_cols: categorical columns
        :param num_cols: numerical columns
        """
        super().__init__(sensitive_name=sensitive_name,
                         label_name=label_name,
                         cate_cols=cate_cols,
                         num_cols=num_cols,
                         verbose=verbose)
        self.sensitive_name = sensitive_name
        self.label_name = label_name
        self.cate_cols = cate_cols
        self.num_cols = num_cols
        assert len(cate_cols) + len(num_cols) > 0, "No columns to over-sample"

    def fit_transform(
            self, dataset: BinaryLabelDataset, crossover_freq: float = 0.8,
            mutation_amount: float = 0.8) -> BinaryLabelDataset:
        def generate_new_sample(obj_df: pd.DataFrame, inc_num: int) -> pd.DataFrame:
            # values = obj_df.values.tolist()
            knn = KNN(n_neighbors=5, algorithm='auto').fit(obj_df.values)

            for _ in range(inc_num):
                rand_idx = random.randint(0, obj_df.shape[0] - 1)
                parent_candidate = obj_df.iloc[rand_idx]
                try:
                    candidate_idx = knn.kneighbors(
                        obj_df.iloc[rand_idx].values.reshape(1, -1), 3,
                        return_distance=False)[0]
                except ValueError:
                    print(rand_idx)
                    print(obj_df.iloc[rand_idx])
                    continue
                # candidate_idx = knn.kneighbors(
                #     parent_candidate.values.reshape(1, -1), 3,
                #     return_distance=False)[0]

                new_candidate = []
                for key, value in parent_candidate.items():
                    if key in self.cate_cols or key == self.label_name:
                        new_value = random.choice(
                            [obj_df.iloc[candidate_idx[0]][key],
                             obj_df.iloc[candidate_idx[1]][key],
                             obj_df.iloc[candidate_idx[2]][key]]
                        )
                        new_candidate.append(new_value)
                    elif key in self.num_cols:
                        new_value = abs(
                            obj_df.iloc[candidate_idx[0]][key] +
                            mutation_amount *
                            (obj_df.iloc[candidate_idx[1]][key] -
                             obj_df.iloc[candidate_idx[2]][key])
                        )
                        new_candidate.append(new_value)
                    else:
                        new_candidate.append(parent_candidate[key])
                add_df = pd.DataFrame([new_candidate], columns=obj_df.columns)
                obj_df = pd.concat([obj_df, add_df], ignore_index=True)
            return obj_df

        origin_df, _ = dataset.convert_to_dataframe()
        # 1. split dataset into 4 groups by label and sensitive (privileged, favorable)
        unp_unf_df = origin_df[(origin_df[self.label_name] == 0) &
                               (origin_df[self.sensitive_name] == 0)]
        unp_fav_df = origin_df[(origin_df[self.label_name] == 1) &
                               (origin_df[self.sensitive_name] == 0)]
        priv_unf_df = origin_df[(origin_df[self.label_name] == 0) &
                                (origin_df[self.sensitive_name] == 1)]
        priv_fav_df = origin_df[(origin_df[self.label_name] == 1) &
                                (origin_df[self.sensitive_name] == 1)]

        unp_unf_num = unp_unf_df.shape[0]
        unp_fav_num = unp_fav_df.shape[0]
        priv_unf_num = priv_unf_df.shape[0]
        priv_fav_num = priv_fav_df.shape[0]
        maximum = max(unp_unf_num, unp_fav_num, priv_unf_num, priv_fav_num)

        unp_unf_inc = maximum - unp_unf_num
        unp_fav_inc = maximum - unp_fav_num
        priv_unf_inc = maximum - priv_unf_num

        # 2. generate new samples for minority classes
        if unp_unf_inc > 0:
            unp_unf_df = generate_new_sample(unp_unf_df, unp_unf_inc)
        if unp_fav_inc > 0:
            unp_fav_df = generate_new_sample(unp_fav_df, unp_fav_inc)
        if priv_unf_inc > 0:
            priv_unf_df = generate_new_sample(priv_unf_df, priv_unf_inc)

        # 3. merge 4 groups
        new_df = pd.concat([unp_unf_df, unp_fav_df, priv_unf_df, priv_fav_df], ignore_index=True)
        new_df = new_df.sample(frac=1).reset_index(drop=True)

        ret_Dataset = BinaryLabelDataset(favorable_label=1.0,
                                         unfavorable_label=0.0,
                                         df=new_df,
                                         label_names=[self.label_name],
                                         protected_attribute_names=[self.sensitive_name],
                                         )

        return ret_Dataset


class LTDD(AifTransformer):
    """
    Paper: Y Li, et al. Training Data Debugging for the Fairness of Machine Learning Software. ICSE 2022.
    """

    def __init__(
            self, sensitive_name: str, label_name: str, num_cols: List[str],
            verbose: bool = False) -> None:
        super().__init__(sensitive_name=sensitive_name,
                         label_name=label_name,
                         num_cols=num_cols,
                         verbose=verbose)
        self.sensitive_name = sensitive_name
        self.label_name = label_name
        self.num_cols = num_cols
        self.modified_cols: Dict[str, Tuple[float, float]] = dict()

    def fit(self, dataset: BinaryLabelDataset) -> None:
        df, _ = dataset.convert_to_dataframe()
        sensitive = df[self.sensitive_name].copy()
        for col in self.num_cols:
            slope, intercept, r_value, p_value, stderr = stats.linregress(
                sensitive, df[col])
            if p_value < 0.05:
                self.modified_cols[col] = (slope, intercept)
        return

    def transform(self, dataset: BinaryLabelDataset) -> BinaryLabelDataset:
        df, _ = dataset.convert_to_dataframe()
        sensitive_attr = df[self.sensitive_name].copy()
        for col in self.modified_cols:
            slope, intercept = self.modified_cols[col]
            if col in df.columns:
                df[col] = df[col]-(slope*sensitive_attr+intercept)

        df[self.sensitive_name] = np.random.randint(0, 2, df.shape[0])
        modified_dataset = BinaryLabelDataset(favorable_label=1.0,
                                              unfavorable_label=0.0,
                                              df=df,
                                              label_names=[self.label_name],
                                              protected_attribute_names=[self.sensitive_name],
                                              )
        return modified_dataset


class FairMask(AifTransformer):
    """
    Paper: K Peng, et al. FairMask: Better Fairness via Model-based Rebalancing of Protected Attributes. TSE 2022.
    Note: The returned dataset contains a masked sensitive attribute and previous sensitive attribute.
    """

    def __init__(
            self, sensitive_name: str, label_name: str, verbose: bool = False) -> None:
        super().__init__(sensitive_name=sensitive_name,
                         label_name=label_name,
                         verbose=verbose)
        self.sensitive_name = sensitive_name
        self.label_name = label_name
        self.sensitive_clf = DecisionTreeRegressor()
        self.relabel_clf = RandomForestClassifier()

    def fit_transform(
            self, train_dataset: BinaryLabelDataset, test_dataset: BinaryLabelDataset
    ) -> Tuple[BinaryLabelDataset, str]:
        train_df, _ = train_dataset.convert_to_dataframe()
        test_df, _ = test_dataset.convert_to_dataframe()

        X_sensitive = train_df.drop(
            [self.sensitive_name, self.label_name], axis=1)
        y_sensitive = train_df[self.sensitive_name]
        self.sensitive_clf.fit(X_sensitive, y_sensitive)

        X_relabel = test_df.drop([self.sensitive_name, self.label_name], axis=1)
        pred_relabel = self.sensitive_clf.predict(X_relabel)
        # test_df.rename(columns={self.sensitive_name: f'prev_{self.sensitive_name}'}, inplace=True)
        test_df[self.sensitive_name] = pred_relabel

        modified_dataset = BinaryLabelDataset(favorable_label=1.0,
                                              unfavorable_label=0.0,
                                              df=test_df,
                                              label_names=[self.label_name],
                                              protected_attribute_names=[self.sensitive_name],
                                              )
        return modified_dataset


class Prep4CDS(AifTransformer):
    """
    Process data for computing Causal Discrimination Score (CDS).
    """

    def __init__(self, sensitive_name: str) -> None:
        super().__init__(sensitive_name=sensitive_name)
        self.sensitive_name = sensitive_name

    def fit_transform(self, obj_dataset: BinaryLabelDataset) -> BinaryLabelDataset:
        obj_df, _ = obj_dataset.convert_to_dataframe()
        assert len(obj_dataset.label_names) == 1

        obj_df[self.sensitive_name] = obj_df[self.sensitive_name].map(
            lambda x: 1 if x == 0 else 0)

        modified_dataset = BinaryLabelDataset(
            favorable_label=1.0,
            unfavorable_label=0.0,
            df=obj_df,
            label_names=[obj_dataset.label_names[0]],
            protected_attribute_names=[self.sensitive_name],
        )
        return modified_dataset


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--method", "-m", type=str,
#                         choices=["LTDD", ],
#                         default="LTDD")
#     # parser.add_argument("--if_exec", "-e", type=bool,
#     #                     default=False, action="store_true")
#     parser.add_argument("--ratio", "-r", type=float,
#                         default=0.0)
#     args = parser.parse_args()

#     if args.method == "LTDD":
#         score = LTDD()
#         print(score)
#     elif args.method == "XXX":
#         pass
#     else:
#         raise ValueError("Unknown dataset")
