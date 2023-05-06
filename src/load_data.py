import numpy as np
import pandas as pd
import os
from scipy import stats
import argparse
from typing import *
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split


class MyDataset(object):
    def __init__(self, path: str, if_discretize: bool = False, if_drop_lot: bool = False):
        self.name = None
        self.cate_cols = []
        self.num_cols = []
        self.sensitive_cols = []
        self.drop_cols = []
        self.drop_lot_cols = []
        self.feature_cols = []
        self.label_col = None
        self.split_ratio = 0.3
        self.original_data = None
        self.processed_data = None
        self.train_df = None
        self.test_df = None
        self.if_discretize = if_discretize
        self.if_drop_lot = if_drop_lot
        self.set_config()
        self.read_data(path)

    def read_data(self, path: str) -> None:
        raise NotImplementedError

    def set_config(self) -> None:
        raise NotImplementedError

    def preprocess(self, object_df):
        raise NotImplementedError

    def get_datasets(self, if_shuffle: bool = True, random_state: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if random_state is not None:
            self.train_df, self.test_df = train_test_split(
                self.processed_data, test_size=self.split_ratio,
                shuffle=if_shuffle, random_state=random_state
            )
        else:
            self.train_df, self.test_df = train_test_split(
                self.processed_data, test_size=self.split_ratio,
                shuffle=if_shuffle
            )

        return self.train_df, self.test_df

    def get_dataframe(self) -> pd.DataFrame:
        return self.processed_data

    def get_original_dataframe(self) -> pd.DataFrame:
        return self.original_data


class AdultDataset(MyDataset):
    def __init__(self, path, if_discretize=False, if_drop_lot=False):
        super().__init__(path, if_discretize, if_drop_lot)
        self.processed_data = self.preprocess(self.original_data)
        self.name = "adult"

    def read_data(self, path: str) -> None:
        train_df = pd.read_csv(os.path.join(path, "adult.data"), sep=", ", engine='python')
        test_df = pd.read_csv(os.path.join(path, "adult.test"), sep=", ", engine='python')
        self.original_data = pd.concat([train_df, test_df], axis=0)

    def set_config(self):
        self.cate_cols = ['workclass', 'marital-status',
                          'occupation', 'relationship', 'race', 'sex',
                          'native-country', ]
        self.num_cols = ['age', 'education-num', 'hours-per-week', 'capital']
        # we drop the fnlwgt column because it is a weight instead of a feature
        self.drop_cols = ['fnlwgt', 'education',
                          'capital-gain', 'capital-loss', ]
        self.drop_lot_cols = ['workclass', 'fnlwgt', 'education', 'marital-status',
                              'occupation', 'relationship', 'capital', 'capital-gain',
                              'capital-loss', 'hours-per-week', 'native-country']
        # self.drop_lot_cols = [
        #     'workclass', 'marital-status', 'occupation', 'relationship', 'native-country', 'fnlwgt',
        #     'education', 'capital-gain', 'capital-loss', ]

        self.sensitive_cols = ['race', 'sex']
        self.label_col = 'label'

    def preprocess(self, object_df: pd.DataFrame) -> pd.DataFrame:
        def group_edu(x):
            if x <= 5:
                return 5
            elif x >= 13:
                return 13
            else:
                return x

        def group_age(x):
            if x >= 70:
                return 70
            else:
                return x

        def group_hours(x):
            if x >= 60:
                return 60
            else:
                return x

        def group_capital(x):
            if x >= 50000:
                return 50000
            elif x < 0:
                return -1
            else:
                return x

        object_df = object_df.dropna()
        object_df['capital'] = object_df['capital-gain']-object_df['capital-loss']
        object_df['native-country'] = object_df['native-country'].apply(
            lambda x: 1.0 if x == 'United-States' else 0.0)

        # Change symbolics to numerics
        object_df['sex'] = object_df['sex'].replace(
            {'Female': 0.0, 'Male': 1.0})
        object_df['race'] = object_df['race'].apply(
            lambda x: 1.0 if x == 'White' else 0.0)
        object_df['label'] = object_df['label'].replace(
            {'<=50K': 0.0, '>50K': 1.0})

        if self.if_drop_lot:
            object_df.drop(self.drop_lot_cols, axis=1, inplace=True)
        else:
            object_df.drop(self.drop_cols, axis=1, inplace=True)
            # Change categorical to integer
            for col in ['workclass', 'marital-status',
                        'occupation', 'relationship']:
                object_df[col] = object_df[col].astype('category').cat.codes

        if self.if_discretize:
            binarize_features = self.num_cols
            object_df['age'] = object_df['age'].apply(lambda x: x//10*10)
            object_df['age'] = object_df['age'].apply(lambda x: group_age(x))
            object_df['education-num'] = object_df['education-num'].apply(
                lambda x: group_edu(x))
            if self.if_drop_lot:
                binarize_features = ['age', 'education-num']
            else:
                object_df['capital'] = object_df['capital'].apply(
                    lambda x: x//10000*10000)
                object_df['capital'] = object_df['capital'].apply(
                    lambda x: group_capital(x))
                # object_df['capital'] = object_df['capital'].apply(
                #     lambda x: 1 if x > 0 else 0)
                object_df['hours-per-week'] = object_df['hours-per-week'].apply(
                    lambda x: x//10*10)
                object_df['hours-per-week'] = object_df['hours-per-week'].apply(
                    lambda x: group_hours(x))

            object_df = pd.get_dummies(
                object_df, columns=binarize_features, prefix_sep='=')

        # for col in self.num_cols:
        #     object_df[col]=object_df[col].astype('int64')

        # object_df.drop(self.drop_cols, axis=1, inplace=True)

        scaler = StandardScaler()
        # scale all the numerical columns
        self.feature_cols = list(object_df.columns)
        self.feature_cols.remove(self.label_col)
        to_scale = [col for col in self.feature_cols if col not in self.sensitive_cols]
        object_df[to_scale] = scaler.fit_transform(
            object_df[to_scale])

        return object_df


class CompasDataset(MyDataset):
    def __init__(self, path, if_discretize=False, if_drop_lot=False):
        super().__init__(path, if_discretize, if_drop_lot)
        self.processed_data = self.preprocess(self.original_data)
        self.name = "compas"

    def read_data(self, path: str) -> None:
        self.original_data = pd.read_csv(os.path.join(path, "compas-scores-two-years.csv"), sep=",")

    def set_config(self) -> None:
        self.cate_cols = ['sex', 'race', 'c_charge_degree', 'c_charge_desc', 'age_cat']
        self.num_cols = ['age', 'juv_fel_count',
                         'juv_misd_count', 'juv_other_count', 'priors_count', ]
        self.drop_cols = []
        self.sensitive_cols = ['sex', 'race']
        self.label_col = 'two_year_recid'

    def preprocess(self, object_df: pd.DataFrame) -> pd.DataFrame:
        # only keep the columns we need
        kept_cols = self.cate_cols + self.num_cols + [self.label_col]
        self.drop_cols = list(set([col for col in object_df.columns if col not in kept_cols]))
        object_df = object_df.loc[:, kept_cols]
        object_df.dropna(inplace=True)
        object_df.reset_index(drop=True, inplace=True)

        # since no recidivism is favaourable, we reverse the label
        object_df['two_year_recid'] = object_df['two_year_recid'].replace(
            {0: 1.0, 1: 0.0}).astype('float64')

        # Change symbolics to numerics
        object_df['sex'] = object_df['sex'].replace(
            {'Female': 1.0, 'Male': 0.0})
        object_df['race'] = object_df['race'].apply(
            lambda x: 1.0 if x == 'Caucasian' else 0.0)
        object_df['c_charge_degree'] = object_df['c_charge_degree'].replace(
            {'F': 1.0, 'M': 0.0})
        object_df['c_charge_desc'] = object_df['c_charge_desc'].astype('category').cat.codes
        object_df['age_cat'] = object_df['age_cat'].astype('category').cat.codes

        scaler = StandardScaler()
        # scale all the numerical columns
        self.feature_cols = list(object_df.columns)
        self.feature_cols.remove(self.label_col)
        to_scale = [col for col in self.feature_cols if col not in self.sensitive_cols]
        object_df[to_scale] = scaler.fit_transform(
            object_df[to_scale])

        return object_df


class GermanDataset(MyDataset):
    def __init__(self, path, if_discretize=False, if_drop_lot=False):
        super().__init__(path, if_discretize, if_drop_lot)
        self.processed_data = self.preprocess(self.original_data)
        self.name = "german"

    def read_data(self, path: str) -> None:
        self.original_data = pd.read_csv(os.path.join(path, "german.data"), sep=" ")

    def set_config(self) -> None:
        self.cate_cols = ['sex', 'age', 'status', 'credit_history', 'purpose',
                          'savings', 'employment', 'other_debtors', 'property',
                          'other_installment', 'housing', 'job', 'telephone',
                          'foreign_worker']
        self.num_cols = ['duration', 'credit_amount', 'installment_rate',
                         'residence', 'number_of_credits', 'people_liable_for']
        self.drop_cols = []
        self.sensitive_cols = ['sex', 'age']
        self.label_col = 'credit'

    def preprocess(self, object_df: pd.DataFrame) -> pd.DataFrame:
        object_df.dropna(inplace=True)
        object_df.reset_index(drop=True, inplace=True)

        # since no recidivism is favaourable, we reverse the label
        object_df['credit'] = object_df['credit'].replace(
            {1: 1.0, 2: 0.0}).astype('float64')

        # Change symbolics to numerics
        object_df['sex'] = object_df['sex'].replace(
            {'A91': 1.0, 'A93': 1.0, 'A94': 1.0,
             'A92': 0.0, 'A95': 0.0})
        object_df['age'] = object_df['age'].apply(
            lambda x: 1.0 if x > 30 else 0.0)
        
        # Change categorical to integer
        for col in [col for col in self.cate_cols if col not in self.sensitive_cols]:
            object_df[col] = object_df[col].astype('category').cat.codes

        scaler = StandardScaler()
        # scale all the numerical columns
        self.feature_cols = list(object_df.columns)
        self.feature_cols.remove(self.label_col)
        to_scale = [col for col in self.feature_cols if col not in self.sensitive_cols]
        object_df[to_scale] = scaler.fit_transform(
            object_df[to_scale])

        return object_df

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--dataset", "-d", type=str,
#                         choices=["compas", "adult", "german", "crime"],
#                         default="adult")
#     args = parser.parse_args()

#     if args.dataset == "compas":
#         ret_dataset = CompasDataset(
#             "./data/compas/compas-scores-two-years.csv")
#     elif args.dataset == "adult":
#         ret_dataset = AdultDataset("./data/adult/")
#     elif args.dataset == "german":
#         # load_german("./data/german/")
#         pass
#     elif args.dataset == "crime":
#         # load_crime("./data/crime/")
#         pass
#     else:
#         raise ValueError("Unknown dataset")

#     print("Done")
