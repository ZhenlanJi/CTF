import os
import argparse
import numpy as np
import pandas as pd
import torch
import torchattacks
from typing import *
from datetime import datetime
from tqdm import tqdm, trange
import sklearn
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import precision_recall_fscore_support
from aif360.algorithms import Transformer as AifTransformer
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from aif360.algorithms.preprocessing.reweighing import Reweighing
from aif360.algorithms.preprocessing import DisparateImpactRemover
from aif360.algorithms.inprocessing.exponentiated_gradient_reduction import ExponentiatedGradientReduction
from aif360.algorithms.postprocessing.eq_odds_postprocessing import EqOddsPostprocessing
from aif360.algorithms.postprocessing.calibrated_eq_odds_postprocessing import CalibratedEqOddsPostprocessing
from aif360.algorithms.postprocessing.reject_option_classification import RejectOptionClassification
from art.estimators.classification.pytorch import PyTorchClassifier
from art.attacks.inference.membership_inference import MembershipInferenceBlackBox, LabelOnlyDecisionBoundary
# import warnings
# warnings.filterwarnings('always')

import utils
from utils import *
from models import MyFNNetClf
from load_data import MyDataset, AdultDataset, CompasDataset, GermanDataset
from fairness.pre_p import Fairway, FairSmote, FairMask, LTDD, Prep4CDS
from fairness.in_p import PRemoveClf, AdDebiasClf


def adversarial_attack(model_torch: torch.nn.Module,
                       test_dataset_sens: BinaryLabelDataset,
                       test_pred: np.ndarray, device: torch.device) -> Tuple[float, float]:

    test_pred = torch.from_numpy(test_pred).long().to(device)
    X_test, y_test = test_dataset_sens.features, test_dataset_sens.labels.ravel()
    number_of_samples = test_pred.shape[0]

    atk_fgsm = torchattacks.FGSM(model_torch)
    fgsm_x = atk_fgsm(torch.from_numpy(X_test).float().to(device),
                      torch.from_numpy(y_test).long().to(device))
    fgsm_y = model_torch(fgsm_x).argmax(dim=1)
    fgsm_sr = (fgsm_y != test_pred.view_as(fgsm_y)).sum().item() / number_of_samples

    atk_pgd = torchattacks.PGD(model_torch)
    pgd_x = atk_pgd(torch.from_numpy(X_test).float().to(device),
                    torch.from_numpy(y_test).long().to(device))
    pgd_y = model_torch(pgd_x).argmax(dim=1)
    pgd_sr = (pgd_y != test_pred.view_as(pgd_y)).sum().item() / number_of_samples

    return fgsm_sr, pgd_sr


def membership_inference(model: MyFNNetClf,
                         train_dataset_sens: BinaryLabelDataset,
                         test_dataset_sens: BinaryLabelDataset,
                         device_type: str) -> float:

    assert isinstance(model, MyFNNetClf), "Model is not initialized correctly"
    model_torch = model.get_model()
    X_train, y_train = train_dataset_sens.features, train_dataset_sens.labels.ravel()
    X_test, y_test = test_dataset_sens.features, test_dataset_sens.labels.ravel()

    attack_train_size = int(0.5*X_train.shape[0])
    attack_test_size = int(0.5*X_test.shape[0])

    art_model = PyTorchClassifier(
        model=model_torch, loss=model.loss_fn,
        optimizer=model.optimizer,
        input_shape=(X_test.shape[1],),
        nb_classes=2, device_type=device_type)

    mi_attack_bb = MembershipInferenceBlackBox(
        art_model, attack_model_type='nn')
    mi_attack_bb.fit(
        X_train[: attack_train_size].astype(np.float32),
        y_train[: attack_train_size],
        X_test[: attack_test_size].astype(np.float32),
        y_test[: attack_test_size])
    mi_inferred_train_bb = mi_attack_bb.infer(
        X_train[attack_train_size:].astype(np.float32),
        y_train[attack_train_size:])
    mi_inferred_test_bb = mi_attack_bb.infer(
        X_test[attack_test_size:].astype(np.float32),
        y_test[attack_test_size:])
    # check accuracy
    mi_train_acc_bb = np.sum(mi_inferred_train_bb) / len(mi_inferred_train_bb)
    mi_test_acc_bb = 1 - (np.sum(mi_inferred_test_bb) / len(mi_inferred_test_bb))
    mi_acc_bb = (mi_train_acc_bb * len(mi_inferred_train_bb) + mi_test_acc_bb *
                 len(mi_inferred_test_bb)) / (len(mi_inferred_train_bb) +
                                              len(mi_inferred_test_bb))

    return mi_acc_bb


def CDS_score(model: Union[MyFNNetClf, sklearn.base.BaseEstimator],
              dataset: BinaryLabelDataset,
              pred: np.ndarray, sensitive_name: str) -> float:
    flip_transformer = Prep4CDS(sensitive_name)
    flipped_x = flip_transformer.fit_transform(dataset).features
    assert isinstance(model, MyFNNetClf) or isinstance(
        model, sklearn.base.BaseEstimator), "Model is not initialized correctly"
    flipped_pred = model.predict(flipped_x).reshape(-1, 1)
    assert flipped_pred.shape == pred.shape, 'Predictions shape mismatch'
    score = (sum(flipped_pred != pred)/pred.shape[0]).item()
    return score


class FairExecutor(object):
    def __init__(self, dataset: MyDataset, sensitive_name: str, save_path: str,
                 device_type: str = 'gpu', repeat_num: int = 10) -> None:
        self.name: Optional[str] = None
        self.save_path = save_path
        self.dataset = dataset
        self.all_df = self.dataset.get_dataframe()
        self.sensitive_name = sensitive_name
        assert self.sensitive_name in self.dataset.sensitive_cols, "Sensitive name not in dataset!"
        other_sensitive_cols = [x for x in self.dataset.sensitive_cols if x != self.sensitive_name]
        assert len(other_sensitive_cols) == 1, "More than one other sensitive columns!"
        self.other_sens_name = other_sensitive_cols[0]
        self.model: Optional[MyFNNetClf] = None
        self.base_model: Optional[MyFNNetClf] = None
        self.collected_df = pd.DataFrame(
            columns=utils.ALL_COLUMNS)
        self.model_width_option = [4]
        self.ratio_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        # self.ratio_list = [1.0]
        self.repeat_num = repeat_num
        self.device_type = device_type
        if device_type is not None:
            self.device = torch.device(
                'cuda:0' if torch.cuda.is_available() else 'cpu')

    @time_statistic
    def compute_ae(
            self, data_row: Dict[str, float],
            test_dataset_sens: BinaryLabelDataset,
            test_pred: np.ndarray) -> Dict[str, float]:

        assert isinstance(self.model, MyFNNetClf), "Model is not initialized correctly"
        model_torch = self.model.get_model()
        fgsm_sr, pgd_sr = adversarial_attack(model_torch, test_dataset_sens, test_pred, self.device)
        data_row["AE_FGSM"] = fgsm_sr
        data_row["AE_PGD"] = pgd_sr

        return data_row

    @time_statistic
    def compute_ae2model(
            self, data_row: Dict[str, float],
            test_dataset_sens: BinaryLabelDataset,
            test_pred: np.ndarray, ratio: float) -> Dict[str, float]:

        assert isinstance(self.model, MyFNNetClf), "Model is not initialized correctly"
        assert isinstance(self.base_model, MyFNNetClf), "Base model is not initialized correctly"

        model_torch = self.model.get_model()
        fgsm_sr_fair, pgd_sr_fair = adversarial_attack(
            model_torch, test_dataset_sens, test_pred, self.device)
        base_model_torch = self.base_model.get_model()
        fgsm_sr_base, pgd_sr_base = adversarial_attack(
            base_model_torch, test_dataset_sens, test_pred, self.device)

        data_row["AE_FGSM"] = fgsm_sr_fair*ratio + fgsm_sr_base*(1-ratio)
        data_row["AE_PGD"] = pgd_sr_fair*ratio + pgd_sr_base*(1-ratio)
        return data_row

    @time_statistic
    def compute_mi(self, data_row: Dict[str, float],
                   train_dataset_sens: BinaryLabelDataset,
                   test_dataset_sens: BinaryLabelDataset,) -> Dict[str, float]:

        mi_acc_bb = membership_inference(
            self.model, train_dataset_sens, test_dataset_sens, self.device_type)
        data_row['MI_BlackBox'] = mi_acc_bb
        return data_row

    @time_statistic
    def compute_mi2model(self, data_row: Dict[str, float],
                         train_dataset_sens: BinaryLabelDataset,
                         test_dataset_sens: BinaryLabelDataset,
                         ratio: float) -> Dict[str, float]:

        mi_acc_bb_fair = membership_inference(
            self.model, train_dataset_sens, test_dataset_sens, self.device_type)
        mi_acc_bb_base = membership_inference(
            self.base_model, train_dataset_sens, test_dataset_sens, self.device_type)

        data_row['MI_BlackBox'] = mi_acc_bb_fair*ratio + mi_acc_bb_base*(1-ratio)
        return data_row

    @time_statistic
    def compute_data_metrics(self, data_row: Dict[str, float],
                             dataset_sens: BinaryLabelDataset) -> Dict[str, float]:
        dataset_sens_metrics = BinaryLabelDatasetMetric(
            dataset_sens, unprivileged_groups=[{self.sensitive_name: 0.0}],
            privileged_groups=[{self.sensitive_name: 1.0}])
        data_row['Data_Cons'] = dataset_sens_metrics.consistency().item()
        data_row['Data_sens_DP'] = dataset_sens_metrics.disparate_impact()
        data_row['Data_sens_SPD'] = dataset_sens_metrics.statistical_parity_difference()

        train_df, _ = dataset_sens.convert_to_dataframe()
        dataset_other = BinaryLabelDataset(
            favorable_label=1.0,
            unfavorable_label=0.0,
            df=train_df.copy(),
            label_names=[self.dataset.label_col],
            protected_attribute_names=[self.other_sens_name],
        )
        dataset_other_metrics = BinaryLabelDatasetMetric(
            dataset_other, unprivileged_groups=[{self.other_sens_name: 0.0}],
            privileged_groups=[{self.other_sens_name: 1.0}])
        data_row['Data_other_DP'] = dataset_other_metrics.disparate_impact()
        data_row['Data_other_SPD'] = dataset_other_metrics.statistical_parity_difference()

        return data_row

    @time_statistic
    def compute_model_metrics(
            self, data_row: Dict[str, float],
            train_dataset_sens: BinaryLabelDataset, test_dataset_sens: BinaryLabelDataset,
            train_pred: np.ndarray, test_pred: np.ndarray,
            train_proba: Optional[np.ndarray] = None, test_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
        # Train metrics
        train_pred_dataset_sens = train_dataset_sens.copy(deepcopy=True)
        train_pred_dataset_sens.labels = train_pred
        if train_proba is not None:
            train_pred_dataset_sens.scores = train_proba
        else:
            train_pred_dataset_sens.scores = train_pred_dataset_sens.labels.copy()
        train_pred_metrics_sens = BinaryLabelDatasetMetric(
            train_pred_dataset_sens, unprivileged_groups=[{self.sensitive_name: 0.0}],
            privileged_groups=[{self.sensitive_name: 1.0}])
        data_row['Train_Cons'] = train_pred_metrics_sens.consistency().item()

        train_metrics_sens = ClassificationMetric(
            train_dataset_sens, train_pred_dataset_sens,
            unprivileged_groups=[{self.sensitive_name: 0.0}],
            privileged_groups=[{self.sensitive_name: 1.0}]
        )
        data_row['Train_Acc'] = train_metrics_sens.accuracy()
        train_recall = train_metrics_sens.recall()
        train_precision = train_metrics_sens.precision()
        data_row['Train_F1'] = 2 * train_recall * train_precision / (train_recall + train_precision)
        data_row['Train_TI'] = train_metrics_sens.theil_index()
        data_row['Train_sens_DP'] = train_metrics_sens.disparate_impact()
        data_row['Train_sens_SPD'] = train_metrics_sens.statistical_parity_difference()
        data_row['Train_sens_AOD'] = train_metrics_sens.average_odds_difference()

        # flip_transformer = Prep4CDS(self.sensitive_name)
        # flipped_x_train = flip_transformer.fit_transform(train_dataset_sens).features
        # assert isinstance(self.model, MyFNNetClf), "Model is not initialized correctly"
        # flipped_train_pred = self.model.predict(flipped_x_train)
        # assert flipped_train_pred.shape == train_pred.shape, 'Predictions shape mismatch'
        # data_row['Train_CDS'] = (sum(flipped_train_pred != train_pred)/train_pred.shape[0]).item()

        train_dataset_others = BinaryLabelDataset(
            favorable_label=1.0,
            unfavorable_label=0.0,
            df=train_dataset_sens.convert_to_dataframe()[0].copy(),
            label_names=[self.dataset.label_col],
            protected_attribute_names=[self.other_sens_name],
        )
        train_pred_dataset_others = train_dataset_others.copy(deepcopy=True)
        train_pred_dataset_others.labels = train_pred
        if train_proba is not None:
            train_pred_dataset_others.scores = train_proba
        else:
            train_pred_dataset_others.scores = train_pred_dataset_others.labels.copy()
        train_metrics_others = ClassificationMetric(
            train_dataset_others, train_pred_dataset_others,
            unprivileged_groups=[{self.other_sens_name: 0.0}],
            privileged_groups=[{self.other_sens_name: 1.0}]
        )
        data_row['Train_other_DP'] = train_metrics_others.disparate_impact()
        data_row['Train_other_SPD'] = train_metrics_others.statistical_parity_difference()
        data_row['Train_other_AOD'] = train_metrics_others.average_odds_difference()

        # Test metrics
        test_pred_dataset_sens = test_dataset_sens.copy(deepcopy=True)
        test_pred_dataset_sens.labels = test_pred
        if test_proba is not None:
            test_pred_dataset_sens.scores = test_proba
        else:
            test_pred_dataset_sens.scores = test_pred_dataset_sens.labels.copy()
        test_pred_metrics_sens = BinaryLabelDatasetMetric(
            test_pred_dataset_sens, unprivileged_groups=[{self.sensitive_name: 0.0}],
            privileged_groups=[{self.sensitive_name: 1.0}])
        data_row['Test_Cons'] = test_pred_metrics_sens.consistency().item()

        test_metrics_sens = ClassificationMetric(
            test_dataset_sens, test_pred_dataset_sens,
            unprivileged_groups=[{self.sensitive_name: 0.0}],
            privileged_groups=[{self.sensitive_name: 1.0}]
        )
        data_row['Test_Acc'] = test_metrics_sens.accuracy()
        test_recall = test_metrics_sens.recall()
        test_precision = test_metrics_sens.precision()
        data_row['Test_F1'] = 2 * test_recall * test_precision / (test_recall + test_precision)
        data_row['Test_TI'] = test_metrics_sens.theil_index()
        data_row['Test_sens_DP'] = test_metrics_sens.disparate_impact()
        data_row['Test_sens_SPD'] = test_metrics_sens.statistical_parity_difference()
        data_row['Test_sens_AOD'] = test_metrics_sens.average_odds_difference()

        # flipped_x_test = flip_transformer.fit_transform(test_dataset_sens).features
        # assert isinstance(self.model, MyFNNetClf), "Model is not initialized correctly"
        # flipped_test_pred = self.model.predict(flipped_x_test)
        # assert flipped_test_pred.shape == test_pred.shape, 'Predictions shape mismatch'
        # data_row['Test_CDS'] = (sum(flipped_test_pred != test_pred)/test_pred.shape[0]).item()

        test_dataset_others = BinaryLabelDataset(
            favorable_label=1.0,
            unfavorable_label=0.0,
            df=test_dataset_sens.convert_to_dataframe()[0].copy(),
            label_names=[self.dataset.label_col],
            protected_attribute_names=[self.other_sens_name],
        )
        test_pred_dataset_others = test_dataset_others.copy(deepcopy=True)
        test_pred_dataset_others.labels = test_pred
        if test_proba is not None:
            test_pred_dataset_others.scores = test_proba
        else:
            test_pred_dataset_others.scores = test_pred_dataset_others.labels.copy()
        test_metrics_others = ClassificationMetric(
            test_dataset_others, test_pred_dataset_others,
            unprivileged_groups=[{self.other_sens_name: 0.0}],
            privileged_groups=[{self.other_sens_name: 1.0}]
        )
        data_row['Test_other_DP'] = test_metrics_others.disparate_impact()
        data_row['Test_other_SPD'] = test_metrics_others.statistical_parity_difference()
        data_row['Test_other_AOD'] = test_metrics_others.average_odds_difference()

        return data_row

    def pre_proc_run(self, fair_proc_cls: AifTransformer, fair_proc_args: dict) -> None:
        for model_width in tqdm(self.model_width_option):
            for ratio in tqdm(self.ratio_list):
                for idx in trange(self.repeat_num):
                    data_row = {}
                    data_row['Model_Width'] = model_width
                    data_row['Ratio'] = ratio

                    train_df, test_df = self.dataset.get_datasets()
                    # train_dataset_sens = BinaryLabelDataset(
                    #     favorable_label=1.0, unfavorable_label=0.0, df=train_df.copy(),
                    #     label_names=[self.dataset.label_col],
                    #     protected_attribute_names=[self.sensitive_name],)
                    test_dataset_sens = BinaryLabelDataset(
                        favorable_label=1.0, unfavorable_label=0.0, df=test_df.copy(),
                        label_names=[self.dataset.label_col],
                        protected_attribute_names=[self.sensitive_name],)

                    fair_processor = fair_proc_cls(**fair_proc_args)
                    if ratio < 1.0:
                        train_df_selected = train_df.sample(frac=ratio)
                        train_df_others = train_df.drop(train_df_selected.index)
                        temp_dataset = BinaryLabelDataset(
                            favorable_label=1.0, unfavorable_label=0.0, df=train_df_selected.copy(),
                            label_names=[self.dataset.label_col],
                            protected_attribute_names=[self.sensitive_name],)
                        train_df_selected = fair_processor.fit_transform(
                            temp_dataset).convert_to_dataframe()[0]
                        train_df_transf = pd.concat(
                            [train_df_selected, train_df_others],
                            ignore_index=True)
                        train_df_transf = train_df_transf.sample(frac=1).reset_index(drop=True)
                        train_dataset_transf = BinaryLabelDataset(
                            favorable_label=1.0, unfavorable_label=0.0, df=train_df_transf.copy(),
                            label_names=[self.dataset.label_col],
                            protected_attribute_names=[self.sensitive_name],)
                    else:
                        temp_dataset = BinaryLabelDataset(
                            favorable_label=1.0, unfavorable_label=0.0, df=train_df.copy(),
                            label_names=[self.dataset.label_col],
                            protected_attribute_names=[self.sensitive_name],)
                        train_dataset_transf = fair_processor.fit_transform(
                            temp_dataset)
                    data_row = self.compute_data_metrics(data_row, train_dataset_transf)

                    X_train = train_dataset_transf.features
                    y_train = train_dataset_transf.labels.ravel()
                    X_test = test_dataset_sens.features
                    y_test = test_dataset_sens.labels.ravel()
                    for _ in range(100):
                        del self.model
                        self.model = MyFNNetClf(
                            inp_features=X_train.shape[1],
                            max_epochs=500, early_stopping=True, width=model_width,
                            device=self.device_type, verbose=False)
                        self.model.fit(X_train, y_train)
                        test_pred = self.model.predict(X_test)
                        train_pred = self.model.predict(X_train)
                        if len(np.unique(test_pred)) == len(np.unique(y_test)) and \
                                len(np.unique(train_pred)) == len(np.unique(y_train)):
                            precision, recall, _, _ = precision_recall_fscore_support(
                                y_test, test_pred, average='binary')
                            if precision > 0 and recall > 0:
                                break
                    train_loss = self.model.compute_loss(X_train, y_train)
                    data_row["Train_Loss"] = train_loss
                    test_loss = self.model.compute_loss(X_test, y_test)
                    data_row["Test_Loss"] = test_loss

                    data_row = self.compute_model_metrics(
                        data_row, train_dataset_transf, test_dataset_sens, train_pred, test_pred)
                    data_row['Train_CDS'] = CDS_score(
                        self.model, train_dataset_transf, train_pred, self.sensitive_name)
                    data_row['Test_CDS'] = CDS_score(
                        self.model, test_dataset_sens, test_pred, self.sensitive_name)
                    data_row = self.compute_ae(data_row, test_dataset_sens, test_pred)
                    data_row = self.compute_mi(
                        data_row, train_dataset_transf, test_dataset_sens)
                    df_data_row = pd.DataFrame([data_row])
                    self.append_to_csv(df_data_row)
                    self.collected_df = pd.concat(
                        [self.collected_df, df_data_row],
                        ignore_index=True,)

    def in_proc_run(
            self, fair_clf_cls: MyFNNetClf, fair_clf_args: Dict[str, Any],
            target_para: str, ratio_coeff: float = 1) -> None:
        for model_width in tqdm(self.model_width_option):
            for ratio in tqdm(self.ratio_list):
                for idx in trange(self.repeat_num):
                    data_row = {}
                    data_row['Model_Width'] = model_width
                    data_row['Ratio'] = ratio

                    train_df, test_df = self.dataset.get_datasets()
                    train_dataset_sens = BinaryLabelDataset(
                        favorable_label=1.0, unfavorable_label=0.0, df=train_df.copy(),
                        label_names=[self.dataset.label_col],
                        protected_attribute_names=[self.sensitive_name],)
                    test_dataset_sens = BinaryLabelDataset(
                        favorable_label=1.0, unfavorable_label=0.0, df=test_df.copy(),
                        label_names=[self.dataset.label_col],
                        protected_attribute_names=[self.sensitive_name],)
                    data_row = self.compute_data_metrics(data_row, train_dataset_sens)

                    sensitive_idx = train_dataset_sens.feature_names.index(self.sensitive_name)
                    assert sensitive_idx == test_dataset_sens.feature_names.index(self.sensitive_name), \
                        "Sensitive attribute index is not the same in train and test data"

                    X_train = train_dataset_sens.features
                    y_train = train_dataset_sens.labels.ravel()
                    X_test = test_dataset_sens.features
                    y_test = test_dataset_sens.labels.ravel()
                    fair_clf_args[target_para] = ratio*ratio_coeff
                    for _ in range(100):
                        del self.model
                        self.model = fair_clf_cls(inp_features=X_train.shape[1],
                                                  width=model_width, device=self.device_type,
                                                  **fair_clf_args,)
                        # self.model.print_params()
                        self.model.fit(X_train, y_train, sensitive_index=sensitive_idx)
                        test_pred = self.model.predict(X_test)
                        train_pred = self.model.predict(X_train)
                        if len(np.unique(test_pred)) == len(np.unique(y_test)) and \
                                len(np.unique(train_pred)) == len(np.unique(y_train)):
                            precision, recall, _, _ = precision_recall_fscore_support(
                                y_test, test_pred, average='binary')
                            if precision > 0 and recall > 0:
                                break
                    train_loss = self.model.compute_loss(X_train, y_train)
                    data_row["Train_Loss"] = train_loss
                    test_loss = self.model.compute_loss(X_test, y_test)
                    data_row["Test_Loss"] = test_loss

                    data_row = self.compute_model_metrics(
                        data_row, train_dataset_sens, test_dataset_sens, train_pred, test_pred)
                    data_row['Train_CDS'] = CDS_score(
                        self.model, train_dataset_sens, train_pred, self.sensitive_name)
                    data_row['Test_CDS'] = CDS_score(
                        self.model, test_dataset_sens, test_pred, self.sensitive_name)
                    data_row = self.compute_ae(data_row, test_dataset_sens, test_pred)
                    data_row = self.compute_mi(
                        data_row, train_dataset_sens, test_dataset_sens)
                    df_data_row = pd.DataFrame([data_row])
                    self.append_to_csv(df_data_row)
                    self.collected_df = pd.concat(
                        [self.collected_df, df_data_row],
                        ignore_index=True,)

    def post_proc_run(self, fair_proc_cls: AifTransformer, fair_proc_args: Dict[str, Any]) -> None:
        for model_width in tqdm(self.model_width_option):
            for ratio in tqdm(self.ratio_list):
                for idx in trange(self.repeat_num):
                    data_row = {}
                    data_row['Model_Width'] = model_width
                    data_row['Ratio'] = ratio

                    train_df, test_df = self.dataset.get_datasets()
                    train_dataset_sens = BinaryLabelDataset(
                        favorable_label=1.0, unfavorable_label=0.0, df=train_df.copy(),
                        label_names=[self.dataset.label_col],
                        protected_attribute_names=[self.sensitive_name],)
                    test_dataset_sens = BinaryLabelDataset(
                        favorable_label=1.0, unfavorable_label=0.0, df=test_df.copy(),
                        label_names=[self.dataset.label_col],
                        protected_attribute_names=[self.sensitive_name],)
                    data_row = self.compute_data_metrics(data_row, train_dataset_sens)

                    X_train = train_dataset_sens.features
                    y_train = train_dataset_sens.labels.ravel()
                    X_test = test_dataset_sens.features
                    y_test = test_dataset_sens.labels.ravel()

                    for _ in range(100):
                        del self.model
                        self.model = MyFNNetClf(
                            inp_features=X_train.shape[1],
                            max_epochs=500, early_stopping=True, width=model_width,
                            device=self.device_type, verbose=False)
                        self.model.fit(X_train, y_train)
                        base_test_pred = self.model.predict(X_test)
                        train_pred = self.model.predict(X_train)
                        if len(np.unique(base_test_pred)) == len(np.unique(y_test)) and \
                                len(np.unique(train_pred)) == len(np.unique(y_train)):
                            precision, recall, _, _ = precision_recall_fscore_support(
                                y_test, base_test_pred, average='binary')
                            if precision > 0 and recall > 0:
                                break

                    train_loss = self.model.compute_loss(X_train, y_train)
                    data_row["Train_Loss"] = train_loss
                    test_loss = self.model.compute_loss(X_test, y_test)
                    data_row["Test_Loss"] = test_loss

                    fair_processor = fair_proc_cls(**fair_proc_args)
                    train_dataset_orig_pred = train_dataset_sens.copy(deepcopy=True)
                    train_dataset_orig_pred.labels = train_pred
                    train_dataset_orig_pred.scores = self.model.predict_proba(X_train)
                    test_dataset_orig_pred = test_dataset_sens.copy(deepcopy=True)
                    test_dataset_orig_pred.labels = base_test_pred
                    test_dataset_orig_pred.scores = self.model.predict_proba(X_test)
                    fair_processor.fit(
                        train_dataset_sens, train_dataset_orig_pred)
                    test_dataset_trans_pred = fair_processor.predict(test_dataset_orig_pred)
                    fair_test_pred = test_dataset_trans_pred.labels.copy()
                    del fair_processor, train_dataset_orig_pred, test_dataset_orig_pred, test_dataset_trans_pred
                    if ratio < 1.0:
                        # concate the original and transformed predictions
                        fair_num = int(ratio * fair_test_pred.shape[0])
                        test_pred = np.concatenate(
                            [fair_test_pred[:fair_num],
                             base_test_pred[fair_num:]])
                    else:
                        test_pred = fair_test_pred.copy()

                    data_row = self.compute_model_metrics(
                        data_row, train_dataset_sens, test_dataset_sens, train_pred, test_pred)
                    data_row['Train_CDS'] = CDS_score(
                        self.model, train_dataset_sens, train_pred, self.sensitive_name)
                    data_row['Test_CDS'] = CDS_score(
                        self.model, test_dataset_sens, test_pred, self.sensitive_name)
                    data_row = self.compute_ae(data_row, test_dataset_sens, test_pred)
                    data_row = self.compute_mi(
                        data_row, train_dataset_sens, test_dataset_sens)
                    df_data_row = pd.DataFrame([data_row])
                    self.append_to_csv(df_data_row)
                    self.collected_df = pd.concat(
                        [self.collected_df, df_data_row],
                        ignore_index=True,)

    def run(self,) -> None:
        raise NotImplementedError

    def save_to_csv(self,) -> None:
        file_path = os.path.join(self.save_path, self.dataset.name,
                                 f"{self.name}_{self.sensitive_name}.csv")
        self.collected_df.to_csv(file_path, index=False)

    def append_to_csv(self, add_df: pd.DataFrame) -> None:
        file_path = os.path.join(self.save_path, self.dataset.name,
                                 f"{self.name}_{self.sensitive_name}.csv")
        add_df = add_df.reindex(utils.ALL_COLUMNS, axis=1)
        add_df.to_csv(file_path, mode='a', header=not os.path.exists(file_path), index=False)


class BaselineExecutor(FairExecutor):
    def __init__(self, dataset: MyDataset, sensitive_name: str, save_path: str,
                 device_type: str = 'gpu', repeat_num: int = 10) -> None:
        super().__init__(dataset=dataset,
                         sensitive_name=sensitive_name,
                         save_path=save_path,
                         device_type=device_type,
                         repeat_num=repeat_num)
        self.name: str = "Baseline"

    def run(self,) -> None:
        for model_width in tqdm(self.model_width_option):
            for _ in trange(self.repeat_num):
                data_row = {}
                data_row['Model_Width'] = model_width
                data_row['Ratio'] = 0

                train_df, test_df = self.dataset.get_datasets()
                train_dataset_sens = BinaryLabelDataset(
                    favorable_label=1.0, unfavorable_label=0.0, df=train_df.copy(),
                    label_names=[self.dataset.label_col],
                    protected_attribute_names=[self.sensitive_name],)
                test_dataset_sens = BinaryLabelDataset(
                    favorable_label=1.0, unfavorable_label=0.0, df=test_df.copy(),
                    label_names=[self.dataset.label_col],
                    protected_attribute_names=[self.sensitive_name],)

                data_row = self.compute_data_metrics(data_row, train_dataset_sens)

                X_train = train_dataset_sens.features
                y_train = train_dataset_sens.labels.ravel()
                X_test = test_dataset_sens.features
                y_test = test_dataset_sens.labels.ravel()

                for _ in range(100):
                    del self.model
                    self.model = MyFNNetClf(
                        inp_features=X_train.shape[1],
                        max_epochs=500, early_stopping=True, width=model_width,
                        device=self.device_type, verbose=False)
                    self.model.fit(X_train, y_train)
                    test_pred = self.model.predict(X_test)
                    train_pred = self.model.predict(X_train)
                    if len(np.unique(test_pred)) == len(np.unique(y_test)) and \
                            len(np.unique(train_pred)) == len(np.unique(y_train)):
                        precision, recall, _, _ = precision_recall_fscore_support(
                            y_test, test_pred, average='binary')
                        if precision > 0 and recall > 0:
                            break

                train_loss = self.model.compute_loss(X_train, y_train)
                data_row["Train_Loss"] = train_loss
                test_loss = self.model.compute_loss(X_test, y_test)
                data_row["Test_Loss"] = test_loss

                data_row = self.compute_model_metrics(
                    data_row, train_dataset_sens, test_dataset_sens, train_pred, test_pred)
                data_row['Train_CDS'] = CDS_score(
                    self.model, train_dataset_sens, train_pred, self.sensitive_name)
                data_row['Test_CDS'] = CDS_score(
                    self.model, test_dataset_sens, test_pred, self.sensitive_name)
                data_row = self.compute_ae(data_row, test_dataset_sens, test_pred)
                data_row = self.compute_mi(
                    data_row, train_dataset_sens, test_dataset_sens)

                df_data_row = pd.DataFrame([data_row])
                self.append_to_csv(df_data_row)
                self.collected_df = pd.concat(
                    [self.collected_df, df_data_row],
                    ignore_index=True,)


class ReweighingExecutor(FairExecutor):
    def __init__(self, dataset: MyDataset, sensitive_name: str, save_path: str,
                 device_type: str = 'gpu', repeat_num: int = 10) -> None:
        super().__init__(dataset=dataset,
                         sensitive_name=sensitive_name,
                         save_path=save_path,
                         device_type=device_type,
                         repeat_num=repeat_num)
        self.name: str = "RW"

    def run(self,) -> None:
        for model_width in tqdm(self.model_width_option):
            for ratio in tqdm(self.ratio_list):
                for idx in trange(self.repeat_num):
                    data_row = {}
                    data_row['Model_Width'] = model_width
                    data_row['Ratio'] = ratio

                    train_df, test_df = self.dataset.get_datasets()
                    train_dataset_sens = BinaryLabelDataset(
                        favorable_label=1.0, unfavorable_label=0.0, df=train_df.copy(),
                        label_names=[self.dataset.label_col],
                        protected_attribute_names=[self.sensitive_name],)
                    test_dataset_sens = BinaryLabelDataset(
                        favorable_label=1.0, unfavorable_label=0.0, df=test_df.copy(),
                        label_names=[self.dataset.label_col],
                        protected_attribute_names=[self.sensitive_name],)
                    # Reweighing does not change dataset features
                    data_row = self.compute_data_metrics(data_row, train_dataset_sens)

                    fair_processor = Reweighing(unprivileged_groups=[{self.sensitive_name: 0.0}],
                                                privileged_groups=[{self.sensitive_name: 1.0}])
                    fair_processor.fit(train_dataset_sens)
                    train_dataset_transf = fair_processor.transform(train_dataset_sens)

                    X_train = train_dataset_transf.features
                    y_train = train_dataset_transf.labels.ravel()
                    W_train = train_dataset_transf.instance_weights.ravel()
                    if ratio < 1.0:
                        sample_idx = np.random.choice(
                            np.arange(len(X_train)), int(len(X_train) * (1-ratio)), replace=False)
                        W_train[sample_idx] = 1.0
                    X_test = test_dataset_sens.features
                    y_test = test_dataset_sens.labels.ravel()
                    for _ in range(100):
                        del self.model
                        self.model = MyFNNetClf(
                            inp_features=X_train.shape[1],
                            max_epochs=500, early_stopping=True, width=model_width,
                            device=self.device_type, verbose=False)
                        self.model.fit(X_train, y_train, W_train)
                        test_pred = self.model.predict(X_test)
                        train_pred = self.model.predict(X_train)
                        if len(np.unique(test_pred)) == len(np.unique(y_test)) and \
                                len(np.unique(train_pred)) == len(np.unique(y_train)):
                            precision, recall, _, _ = precision_recall_fscore_support(
                                y_test, test_pred, average='binary')
                            if precision > 0 and recall > 0:
                                break

                    train_loss = self.model.compute_loss(X_train, y_train, W_train)
                    data_row["Train_Loss"] = train_loss
                    test_loss = self.model.compute_loss(X_test, y_test)
                    data_row["Test_Loss"] = test_loss

                    # When compute metrics, we do not consider the instance weights, since some of metrics can not support it.
                    data_row = self.compute_model_metrics(
                        data_row, train_dataset_sens, test_dataset_sens, train_pred, test_pred)
                    data_row['Train_CDS'] = CDS_score(
                        self.model, train_dataset_sens, train_pred, self.sensitive_name)
                    data_row['Test_CDS'] = CDS_score(
                        self.model, test_dataset_sens, test_pred, self.sensitive_name)
                    data_row = self.compute_ae(data_row, test_dataset_sens, test_pred)
                    data_row = self.compute_mi(
                        data_row, train_dataset_sens, test_dataset_sens)
                    df_data_row = pd.DataFrame([data_row])
                    self.append_to_csv(df_data_row)
                    self.collected_df = pd.concat(
                        [self.collected_df, df_data_row],
                        ignore_index=True,)


class FairwayExecutor(FairExecutor):
    def __init__(self, dataset: MyDataset, sensitive_name: str, save_path: str,
                 device_type: str = 'gpu', repeat_num: int = 10) -> None:
        super().__init__(dataset=dataset,
                         sensitive_name=sensitive_name,
                         save_path=save_path,
                         device_type=device_type,
                         repeat_num=repeat_num)
        self.name: str = "Fairway"

    def run(self,) -> None:
        proc_args = {
            "sensitive_name": self.sensitive_name,
            "label_name": self.dataset.label_col,
            "LR_clf_params": {'C': 1.0, 'penalty': 'l2',
                              'solver': 'liblinear', 'max_iter': 500}
        }

        self.pre_proc_run(fair_proc_cls=Fairway, fair_proc_args=proc_args)


class FairSmoteExecutor(FairExecutor):
    def __init__(self, dataset: MyDataset, sensitive_name: str, save_path: str,
                 device_type: str = 'gpu', repeat_num: int = 10) -> None:
        super().__init__(dataset=dataset,
                         sensitive_name=sensitive_name,
                         save_path=save_path,
                         device_type=device_type,
                         repeat_num=repeat_num)
        self.name: str = "FairSmote"

    def run(self,) -> None:
        proc_args = {
            "sensitive_name": self.sensitive_name,
            "label_name": self.dataset.label_col,
            "cate_cols": self.dataset.cate_cols,
            "num_cols": self.dataset.num_cols,
        }
        self.pre_proc_run(fair_proc_cls=FairSmote, fair_proc_args=proc_args)


class DIRExecutor(FairExecutor):
    def __init__(self, dataset: MyDataset, sensitive_name: str, save_path: str,
                 device_type: str = 'gpu', repeat_num: int = 10) -> None:
        super().__init__(dataset=dataset,
                         sensitive_name=sensitive_name,
                         save_path=save_path,
                         device_type=device_type,
                         repeat_num=repeat_num)
        self.name: str = "DIR"

    def run(self,) -> None:
        for model_width in tqdm(self.model_width_option):
            for ratio in tqdm(self.ratio_list):
                for idx in trange(self.repeat_num):
                    data_row = {}
                    data_row['Model_Width'] = model_width
                    data_row['Ratio'] = ratio

                    train_df, test_df = self.dataset.get_datasets()
                    train_dataset_sens = BinaryLabelDataset(
                        favorable_label=1.0, unfavorable_label=0.0, df=train_df.copy(),
                        label_names=[self.dataset.label_col],
                        protected_attribute_names=[self.sensitive_name],)
                    test_dataset_sens = BinaryLabelDataset(
                        favorable_label=1.0, unfavorable_label=0.0, df=test_df.copy(),
                        label_names=[self.dataset.label_col],
                        protected_attribute_names=[self.sensitive_name],)

                    fair_processor = DisparateImpactRemover(repair_level=ratio)
                    train_dataset_transf = fair_processor.fit_transform(train_dataset_sens)
                    test_dataset_transf = fair_processor.fit_transform(test_dataset_sens)

                    data_row = self.compute_data_metrics(data_row, train_dataset_transf)

                    X_train = train_dataset_transf.features
                    y_train = train_dataset_transf.labels.ravel()
                    X_test = test_dataset_transf.features
                    y_test = test_dataset_transf.labels.ravel()
                    for _ in range(100):
                        del self.model
                        self.model = MyFNNetClf(
                            inp_features=X_train.shape[1],
                            max_epochs=500, early_stopping=True, width=model_width,
                            device=self.device_type, verbose=False)
                        self.model.fit(X_train, y_train)
                        test_pred = self.model.predict(X_test)
                        train_pred = self.model.predict(X_train)
                        if len(np.unique(test_pred)) == len(np.unique(y_test)) and \
                                len(np.unique(train_pred)) == len(np.unique(y_train)):
                            precision, recall, _, _ = precision_recall_fscore_support(
                                y_test, test_pred, average='binary')
                            if precision > 0 and recall > 0:
                                break
                    train_loss = self.model.compute_loss(X_train, y_train)
                    data_row["Train_Loss"] = train_loss
                    test_loss = self.model.compute_loss(X_test, y_test)
                    data_row["Test_Loss"] = test_loss

                    data_row = self.compute_model_metrics(
                        data_row, train_dataset_transf, test_dataset_sens, train_pred, test_pred)
                    data_row['Train_CDS'] = CDS_score(
                        self.model, train_dataset_transf, train_pred, self.sensitive_name)
                    data_row['Test_CDS'] = CDS_score(
                        self.model, test_dataset_sens, test_pred, self.sensitive_name)
                    data_row = self.compute_ae(data_row, test_dataset_sens, test_pred)
                    data_row = self.compute_mi(
                        data_row, train_dataset_transf, test_dataset_sens)
                    df_data_row = pd.DataFrame([data_row])
                    self.append_to_csv(df_data_row)
                    self.collected_df = pd.concat(
                        [self.collected_df, df_data_row],
                        ignore_index=True,)


class LTDDExecutor(FairExecutor):
    def __init__(self, dataset: MyDataset, sensitive_name: str, save_path: str,
                 device_type: str = 'gpu', repeat_num: int = 10) -> None:
        super().__init__(dataset=dataset,
                         sensitive_name=sensitive_name,
                         save_path=save_path,
                         device_type=device_type,
                         repeat_num=repeat_num)
        self.name: str = "LTDD"

    def pre_proc_run(self, fair_proc_cls: AifTransformer, fair_proc_args: dict) -> None:
        for model_width in tqdm(self.model_width_option):
            for ratio in tqdm(self.ratio_list):
                for idx in trange(self.repeat_num):
                    data_row = {}
                    data_row['Model_Width'] = model_width
                    data_row['Ratio'] = ratio

                    train_df, test_df = self.dataset.get_datasets()
                    train_dataset_sens = BinaryLabelDataset(
                        favorable_label=1.0, unfavorable_label=0.0, df=train_df.copy(),
                        label_names=[self.dataset.label_col],
                        protected_attribute_names=[self.sensitive_name],)
                    test_dataset_sens = BinaryLabelDataset(
                        favorable_label=1.0, unfavorable_label=0.0, df=test_df.copy(),
                        label_names=[self.dataset.label_col],
                        protected_attribute_names=[self.sensitive_name],)

                    fair_processor = fair_proc_cls(**fair_proc_args)
                    fair_processor.fit(train_dataset_sens)
                    train_dataset_transf = fair_processor.transform(
                        train_dataset_sens)
                    test_dataset_transf = fair_processor.transform(
                        test_dataset_sens)
                    data_row = self.compute_data_metrics(data_row, train_dataset_transf)

                    X_train = train_dataset_transf.features
                    y_train = train_dataset_transf.labels.ravel()
                    X_test = test_dataset_transf.features
                    y_test = test_dataset_transf.labels.ravel()
                    X_train_ori = train_dataset_sens.features
                    y_train_ori = train_dataset_sens.labels.ravel()
                    X_test_ori = test_dataset_sens.features
                    y_test_ori = test_dataset_sens.labels.ravel()

                    for _ in range(100):
                        del self.base_model, self.model
                        self.base_model = MyFNNetClf(
                            inp_features=X_train.shape[1],
                            max_epochs=500, early_stopping=True, width=model_width,
                            device=self.device_type, verbose=False)
                        self.base_model.fit(X_train_ori, y_train_ori)
                        self.model = MyFNNetClf(
                            inp_features=X_train.shape[1],
                            max_epochs=500, early_stopping=True, width=model_width,
                            device=self.device_type, verbose=False)
                        self.model.fit(X_train, y_train)

                        train_num = X_train.shape[0]
                        base_train_pred = self.base_model.predict(X_train_ori)
                        fair_train_pred = self.model.predict(X_train)
                        train_pred = np.concatenate(
                            (fair_train_pred[:int(train_num*ratio)],
                             base_train_pred[int(train_num*ratio):]))
                        test_num = X_test.shape[0]
                        base_test_pred = self.base_model.predict(X_test_ori)
                        fair_test_pred = self.model.predict(X_test)
                        test_pred = np.concatenate(
                            (fair_test_pred[:int(test_num*ratio)],
                             base_test_pred[int(test_num*ratio):]))

                        if len(np.unique(test_pred)) == len(np.unique(y_test)) and \
                                len(np.unique(train_pred)) == len(np.unique(y_train)):
                            precision, recall, _, _ = precision_recall_fscore_support(
                                y_test, test_pred, average='binary')
                            if precision > 0 and recall > 0:
                                break
                    train_loss = self.model.compute_loss(X_train, y_train) * ratio + \
                        self.base_model.compute_loss(X_train_ori, y_train_ori) * (1-ratio)
                    data_row["Train_Loss"] = train_loss
                    test_loss = self.model.compute_loss(X_test, y_test)*ratio + \
                        self.base_model.compute_loss(X_test_ori, y_test_ori)*(1-ratio)
                    data_row["Test_Loss"] = test_loss

                    data_row = self.compute_model_metrics(
                        data_row, train_dataset_sens, test_dataset_sens, train_pred, test_pred)

                    train_base_CDS = CDS_score(
                        self.base_model, train_dataset_sens, base_train_pred, self.sensitive_name)
                    train_fair_CDS = CDS_score(
                        self.model, train_dataset_transf, fair_train_pred, self.sensitive_name)
                    data_row['Train_CDS'] = train_base_CDS * (1-ratio) + train_fair_CDS * ratio
                    test_base_CDS = CDS_score(
                        self.base_model, test_dataset_sens, base_test_pred, self.sensitive_name)
                    test_fair_CDS = CDS_score(
                        self.model, test_dataset_transf, fair_test_pred, self.sensitive_name)
                    data_row['Test_CDS'] = test_base_CDS * (1-ratio) + test_fair_CDS * ratio

                    data_row = self.compute_ae2model(data_row, test_dataset_sens, test_pred, ratio)
                    data_row = self.compute_mi2model(
                        data_row, train_dataset_transf, test_dataset_sens, ratio)
                    df_data_row = pd.DataFrame([data_row])
                    self.append_to_csv(df_data_row)
                    self.collected_df = pd.concat(
                        [self.collected_df, df_data_row],
                        ignore_index=True,)

    def run(self,) -> None:
        proc_args = {
            "sensitive_name": self.sensitive_name,
            "label_name": self.dataset.label_col,
            "num_cols": self.dataset.num_cols,
        }
        self.pre_proc_run(fair_proc_cls=LTDD, fair_proc_args=proc_args)


class FairMaskExecutor(FairExecutor):
    def __init__(self, dataset: MyDataset, sensitive_name: str, save_path: str,
                 device_type: str = 'gpu', repeat_num: int = 10) -> None:
        super().__init__(dataset=dataset,
                         sensitive_name=sensitive_name,
                         save_path=save_path,
                         device_type=device_type,
                         repeat_num=repeat_num)
        self.name: str = "FairMask"

    def run(self,) -> None:
        for model_width in tqdm(self.model_width_option):
            for ratio in tqdm(self.ratio_list):
                for idx in trange(self.repeat_num):
                    data_row = {}
                    data_row['Model_Width'] = model_width
                    data_row['Ratio'] = ratio

                    train_df, test_df = self.dataset.get_datasets()
                    train_dataset_sens = BinaryLabelDataset(
                        favorable_label=1.0, unfavorable_label=0.0, df=train_df.copy(),
                        label_names=[self.dataset.label_col],
                        protected_attribute_names=[self.sensitive_name],)
                    test_dataset_sens = BinaryLabelDataset(
                        favorable_label=1.0, unfavorable_label=0.0, df=test_df.copy(),
                        label_names=[self.dataset.label_col],
                        protected_attribute_names=[self.sensitive_name],)
                    # FairMask only change the sensitive attribute for test data
                    data_row = self.compute_data_metrics(data_row, train_dataset_sens)

                    fair_processor = FairMask(self.sensitive_name, self.dataset.label_col)
                    test_df_fair = fair_processor.fit_transform(
                        train_dataset_sens, test_dataset_sens).convert_to_dataframe()[0]
                    if ratio < 1:
                        transf_num = int(ratio * len(test_df))
                        test_df_transf = pd.concat(
                            [test_df_fair.iloc[:transf_num, :],
                             test_df.iloc[transf_num:, :]],
                            axis=0)
                    else:
                        test_df_transf = test_df_fair
                    test_dataset_transf = BinaryLabelDataset(
                        favorable_label=1.0, unfavorable_label=0.0, df=test_df_transf.copy(),
                        label_names=[self.dataset.label_col],
                        protected_attribute_names=[self.sensitive_name]
                    )
                    X_train = train_dataset_sens.features
                    y_train = train_dataset_sens.labels.ravel()
                    X_test = test_dataset_transf.features
                    y_test = test_dataset_transf.labels.ravel()

                    for _ in range(100):
                        del self.model
                        self.model = MyFNNetClf(
                            inp_features=X_train.shape[1],
                            max_epochs=500, early_stopping=True, width=model_width,
                            device=self.device_type, verbose=False)
                        self.model.fit(X_train, y_train)
                        test_pred = self.model.predict(X_test)
                        train_pred = self.model.predict(X_train)
                        if len(np.unique(test_pred)) == len(np.unique(y_test)) and \
                                len(np.unique(train_pred)) == len(np.unique(y_train)):
                            precision, recall, _, _ = precision_recall_fscore_support(
                                y_test, test_pred, average='binary')
                            if precision > 0 and recall > 0:
                                break

                    train_loss = self.model.compute_loss(X_train, y_train)
                    data_row["Train_Loss"] = train_loss
                    test_loss = self.model.compute_loss(X_test, y_test)
                    data_row["Test_Loss"] = test_loss

                    # when compute fairness metrics, the original test data should be used
                    data_row = self.compute_model_metrics(
                        data_row, train_dataset_sens, test_dataset_sens, train_pred, test_pred)
                    data_row['Train_CDS'] = CDS_score(
                        self.model, train_dataset_sens, train_pred, self.sensitive_name)
                    data_row['Test_CDS'] = 1*ratio + (1-ratio)*CDS_score(self.model,
                                                                         test_dataset_sens, test_pred, self.sensitive_name)
                    # while for robustness metrics, the transformed test data should be used
                    data_row = self.compute_ae(data_row, test_dataset_transf, test_pred)
                    data_row = self.compute_mi(
                        data_row, train_dataset_sens, test_dataset_transf)
                    df_data_row = pd.DataFrame([data_row])
                    self.append_to_csv(df_data_row)
                    self.collected_df = pd.concat(
                        [self.collected_df, df_data_row],
                        ignore_index=True,)


class PRExecutor(FairExecutor):
    def __init__(self, dataset: MyDataset, sensitive_name: str, save_path: str,
                 device_type: str = 'gpu', repeat_num: int = 10) -> None:
        super().__init__(dataset=dataset,
                         sensitive_name=sensitive_name,
                         save_path=save_path,
                         device_type=device_type,
                         repeat_num=repeat_num)
        self.name: str = "PR"

    def run(self) -> None:
        # self.ratio_list = [0.02, 0.04, 0.06, 0.08, 0.1]
        clf_args = {
            "max_epochs": 500,
            "early_stopping": True,
            "verbose": False,
        }
        self.in_proc_run(fair_clf_cls=PRemoveClf, fair_clf_args=clf_args,
                         target_para="eta", ratio_coeff=0.1)


class AdDebiasExecutor(FairExecutor):
    def __init__(self, dataset: MyDataset, sensitive_name: str, save_path: str,
                 device_type: str = 'gpu', repeat_num: int = 10) -> None:
        super().__init__(dataset=dataset,
                         sensitive_name=sensitive_name,
                         save_path=save_path,
                         device_type=device_type,
                         repeat_num=repeat_num)
        self.name: str = "AdDebias"

    def run(self) -> None:
        clf_args = {
            "pre_clf_epochs": 100,
            "pre_adv_epochs": 100,
            "max_epochs": 500,
            "early_stopping": True,
            "verbose": False,
        }
        self.in_proc_run(
            fair_clf_cls=AdDebiasClf, fair_clf_args=clf_args,
            target_para="adversary_loss_weight")


class EGRExecutor(FairExecutor):
    def __init__(self, dataset: MyDataset, sensitive_name: str, save_path: str,
                 device_type: str = 'gpu', repeat_num: int = 10) -> None:
        super().__init__(dataset=dataset,
                         sensitive_name=sensitive_name,
                         save_path=save_path,
                         device_type=device_type,
                         repeat_num=repeat_num)
        self.name: str = "EGR"

    def run(self) -> None:
        for model_width in tqdm(self.model_width_option):
            for ratio in tqdm(self.ratio_list):
                for idx in trange(self.repeat_num):
                    data_row = {}
                    data_row['Model_Width'] = model_width
                    data_row['Ratio'] = ratio

                    train_df, test_df = self.dataset.get_datasets()
                    train_dataset_sens = BinaryLabelDataset(
                        favorable_label=1.0, unfavorable_label=0.0, df=train_df.copy(),
                        label_names=[self.dataset.label_col],
                        protected_attribute_names=[self.sensitive_name],)
                    test_dataset_sens = BinaryLabelDataset(
                        favorable_label=1.0, unfavorable_label=0.0, df=test_df.copy(),
                        label_names=[self.dataset.label_col],
                        protected_attribute_names=[self.sensitive_name],)
                    data_row = self.compute_data_metrics(data_row, train_dataset_sens)

                    X_train = train_dataset_sens.features
                    y_train = train_dataset_sens.labels.ravel()
                    X_test = test_dataset_sens.features
                    y_test = test_dataset_sens.labels.ravel()

                    for _ in range(100):
                        del self.base_model, self.model
                        self.base_model = MyFNNetClf(
                            inp_features=X_train.shape[1],
                            max_epochs=500, early_stopping=True, width=model_width,
                            device=self.device_type, verbose=False)
                        self.base_model.fit(X_train, y_train)

                        X_train_df = pd.DataFrame(X_train, columns=train_dataset_sens.feature_names)
                        X_test_df = pd.DataFrame(X_test, columns=test_dataset_sens.feature_names)
                        estimator = MyFNNetClf(
                            inp_features=X_train.shape[1],
                            max_epochs=500, early_stopping=True, width=model_width,
                            device=self.device_type, verbose=False)
                        self.model = ExponentiatedGradientReduction(estimator=estimator,
                                                                    constraints="EqualizedOdds",
                                                                    drop_prot_attr=False).model
                        self.model.prot_attr = [self.sensitive_name]
                        self.model.fit(X_train_df, y_train)

                        train_num = X_train.shape[0]
                        base_train_pred = self.base_model.predict(X_train)
                        fair_train_pred = self.model.predict(X_train_df).reshape(-1, 1)
                        train_pred = np.concatenate(
                            (fair_train_pred[:int(train_num*ratio)],
                             base_train_pred[int(train_num*ratio):]))
                        test_num = X_test.shape[0]
                        base_test_pred = self.base_model.predict(X_test)
                        fair_test_pred = self.model.predict(X_test_df).reshape(-1, 1)
                        test_pred = np.concatenate(
                            (fair_test_pred[:int(test_num*ratio)],
                             base_test_pred[int(test_num*ratio):]))

                        if len(np.unique(test_pred)) == len(np.unique(y_test)) and \
                                len(np.unique(train_pred)) == len(np.unique(y_train)):
                            precision, recall, _, _ = precision_recall_fscore_support(
                                y_test, test_pred, average='binary')
                            if precision > 0 and recall > 0:
                                break
                    train_pred_prob = self.model.predict_proba(X_train_df)
                    train_pred_prob = torch.from_numpy(train_pred_prob).float().to(self.device)
                    fair_train_loss = torch.nn.functional.cross_entropy(
                        train_pred_prob, torch.from_numpy(y_train).long().to(self.device)).item()
                    base_train_loss = self.base_model.compute_loss(X_train, y_train)
                    train_loss = fair_train_loss * ratio + base_train_loss * (1-ratio)
                    data_row["Train_Loss"] = train_loss
                    test_pred_prob = self.model.predict_proba(X_test_df)
                    test_pred_prob = torch.from_numpy(test_pred_prob).float().to(self.device)
                    fair_test_loss = torch.nn.functional.cross_entropy(
                        test_pred_prob, torch.from_numpy(y_test).long().to(self.device)).item()
                    base_test_loss = self.base_model.compute_loss(X_test, y_test)
                    test_loss = fair_test_loss * ratio + base_test_loss * (1-ratio)
                    data_row["Test_Loss"] = test_loss

                    data_row = self.compute_model_metrics(
                        data_row, train_dataset_sens, test_dataset_sens, train_pred, test_pred)

                    train_base_CDS = CDS_score(
                        self.base_model, train_dataset_sens, base_train_pred, self.sensitive_name)
                    train_fair_CDS = CDS_score(
                        self.model, train_dataset_sens, fair_train_pred, self.sensitive_name)
                    data_row['Train_CDS'] = train_base_CDS * (1-ratio) + train_fair_CDS * ratio
                    test_base_CDS = CDS_score(
                        self.base_model, test_dataset_sens, base_test_pred, self.sensitive_name)
                    test_fair_CDS = CDS_score(
                        self.model, test_dataset_sens, fair_test_pred, self.sensitive_name)
                    data_row['Test_CDS'] = test_base_CDS * (1-ratio) + test_fair_CDS * ratio

                    df_data_row = pd.DataFrame([data_row])
                    self.append_to_csv(df_data_row)
                    self.collected_df = pd.concat(
                        [self.collected_df, df_data_row],
                        ignore_index=True,)


class CEOExecutor(FairExecutor):
    def __init__(self, dataset: MyDataset, sensitive_name: str, save_path: str,
                 device_type: str = 'gpu', repeat_num: int = 10) -> None:
        super().__init__(dataset=dataset,
                         sensitive_name=sensitive_name,
                         save_path=save_path,
                         device_type=device_type,
                         repeat_num=repeat_num)
        self.name: str = "CEO"

    def run(self,) -> None:
        proc_args = {
            "privileged_groups": [{self.sensitive_name: 1}],
            "unprivileged_groups": [{self.sensitive_name: 0}],
            "cost_constraint": "weighted",
        }
        self.post_proc_run(fair_proc_cls=CalibratedEqOddsPostprocessing, fair_proc_args=proc_args)


class EOExecutor(FairExecutor):
    def __init__(self, dataset: MyDataset, sensitive_name: str, save_path: str,
                 device_type: str = 'gpu', repeat_num: int = 10) -> None:
        super().__init__(dataset=dataset,
                         sensitive_name=sensitive_name,
                         save_path=save_path,
                         device_type=device_type,
                         repeat_num=repeat_num)
        self.name: str = "EO"

    def run(self,) -> None:
        proc_args = {
            "privileged_groups": [{self.sensitive_name: 1}],
            "unprivileged_groups": [{self.sensitive_name: 0}],
        }
        self.post_proc_run(fair_proc_cls=EqOddsPostprocessing, fair_proc_args=proc_args)


class ROCExecutor(FairExecutor):
    def __init__(self, dataset: MyDataset, sensitive_name: str, save_path: str,
                 device_type: str = 'gpu', repeat_num: int = 10) -> None:
        super().__init__(dataset=dataset,
                         sensitive_name=sensitive_name,
                         save_path=save_path,
                         device_type=device_type,
                         repeat_num=repeat_num)
        self.name: str = "ROC"
        # self.model_width_option = [2]

    def run(self,) -> None:
        proc_args = {
            "privileged_groups": [{self.sensitive_name: 1}],
            "unprivileged_groups": [{self.sensitive_name: 0}],
            "low_class_thresh": 0.01,
            "high_class_thresh": 0.99,
            "num_class_thresh": 100,
            "num_ROC_margin": 50,
            "metric_name": "Average odds difference",
            "metric_ub": 0.05,
            "metric_lb": -0.05
        }
        self.post_proc_run(fair_proc_cls=RejectOptionClassification, fair_proc_args=proc_args)


class CombinedExecutor(FairExecutor):
    def __init__(self, dataset: MyDataset, sensitive_name: str, save_path: str,
                 device_type: str = 'gpu', repeat_num: int = 10) -> None:
        super().__init__(dataset=dataset,
                         sensitive_name=sensitive_name,
                         save_path=save_path,
                         device_type=device_type,
                         repeat_num=repeat_num)
        self.name: str = "DIR+PR"

    def append_to_csv(self, add_df: pd.DataFrame) -> None:
        file_path = os.path.join(self.save_path, self.dataset.name,
                                 f"{self.name}_{self.sensitive_name}.csv")
        # add_df = add_df.reindex(utils.ALL_COLUMNS, axis=1)
        add_df.to_csv(file_path, mode='a', header=not os.path.exists(file_path), index=False)

    def run(self,) -> None:
        for model_width in tqdm(self.model_width_option):
            for d_ratio in tqdm(self.ratio_list):
                for p_ratio in tqdm(self.ratio_list):
                    for idx in trange(self.repeat_num):
                        data_row = {}
                        data_row['Model_Width'] = model_width
                        data_row['DIR'] = d_ratio
                        data_row['PR'] = p_ratio

                        train_df, test_df = self.dataset.get_datasets()
                        train_dataset_sens = BinaryLabelDataset(
                            favorable_label=1.0, unfavorable_label=0.0, df=train_df.copy(),
                            label_names=[self.dataset.label_col],
                            protected_attribute_names=[self.sensitive_name],)
                        test_dataset_sens = BinaryLabelDataset(
                            favorable_label=1.0, unfavorable_label=0.0, df=test_df.copy(),
                            label_names=[self.dataset.label_col],
                            protected_attribute_names=[self.sensitive_name],)

                        fair_processor = DisparateImpactRemover(repair_level=d_ratio)
                        train_dataset_transf = fair_processor.fit_transform(train_dataset_sens)
                        test_dataset_transf = fair_processor.fit_transform(test_dataset_sens)

                        sensitive_idx = train_dataset_sens.feature_names.index(self.sensitive_name)
                        assert sensitive_idx == test_dataset_sens.feature_names.index(self.sensitive_name), \
                            "Sensitive attribute index is not the same in train and test data"

                        X_train = train_dataset_transf.features
                        y_train = train_dataset_transf.labels.ravel()
                        X_test = test_dataset_transf.features
                        y_test = test_dataset_transf.labels.ravel()
                        for _ in range(100):
                            del self.model
                            self.model = PRemoveClf(inp_features=X_train.shape[1],
                                                    width=model_width, device=self.device_type,
                                                    max_epochs=500, early_stopping=True,
                                                    verbose=False)

                            # self.model.print_params()
                            self.model.fit(X_train, y_train, sensitive_index=sensitive_idx)
                            test_pred = self.model.predict(X_test)
                            train_pred = self.model.predict(X_train)
                            if len(np.unique(test_pred)) == len(np.unique(y_test)) and \
                                    len(np.unique(train_pred)) == len(np.unique(y_train)):
                                precision, recall, _, _ = precision_recall_fscore_support(
                                    y_test, test_pred, average='binary')
                                if precision > 0 and recall > 0:
                                    break

                        train_loss = self.model.compute_loss(X_train, y_train)
                        data_row["Train_Loss"] = train_loss
                        test_loss = self.model.compute_loss(X_test, y_test)
                        data_row["Test_Loss"] = test_loss

                        data_row = self.compute_model_metrics(
                            data_row, train_dataset_transf, test_dataset_sens, train_pred, test_pred)
                        data_row['Test_CDS'] = CDS_score(
                            self.model, test_dataset_sens, test_pred, self.sensitive_name)
                        df_data_row = pd.DataFrame([data_row])
                        self.append_to_csv(df_data_row)



METHOD_DICT = {
    # pre-processing
    "baseline": BaselineExecutor,
    "rw": ReweighingExecutor,
    "fairway": FairwayExecutor,
    "fairsmote": FairSmoteExecutor,
    "ltdd": LTDDExecutor,
    "dir": DIRExecutor,
    "fairmask": FairMaskExecutor,
    # in-processing
    "pr": PRExecutor,
    "ad": AdDebiasExecutor,
    "egr": EGRExecutor,
    # post-processing
    "ceo": CEOExecutor,
    "eo": EOExecutor,
    "roc": ROCExecutor,
    # combined
    "combined": CombinedExecutor,
}


@ time_statistic
def main(args):
    if args.dataset == "adult":
        obj_dataset = AdultDataset("./data/adult/", if_discretize=False, if_drop_lot=False)
    elif args.dataset == "compas":
        obj_dataset = CompasDataset("./data/compas/", if_discretize=False, if_drop_lot=False)
    elif args.dataset == "german":
        obj_dataset = GermanDataset("./data/german/", if_discretize=False, if_drop_lot=False)
    elif args.dataset == "crime":
        pass
    else:
        raise ValueError("Dataset not supported!")

    executer = METHOD_DICT[args.method](
        dataset=obj_dataset, sensitive_name=args.sensitive_name, save_path=args.save_path, repeat_num=args.repeat_num)
    executer.run()
    # executer.save_to_csv()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", type=str,
                        choices=["adult", "compas", "german", "crime"],
                        default="adult")
    parser.add_argument("--method", "-m", type=str,
                        choices=METHOD_DICT.keys(),
                        default="ceo",)
    parser.add_argument("--sensitive_name", "-sn", type=str,
                        choices=["sex", "race", "age"], default="sex")
    parser.add_argument("--save_path", "-sp", type=str,
                        default="")
    parser.add_argument("--repeat_num", "-rn", type=int,
                        default=5)

    # parser.add_argument("--if_exec", "-e", type=bool,
    #                     default=False, action="store_true")
    # parser.add_argument("--ratio", "-r", type=float,
    #                     default=0.0)
    args = parser.parse_args()
    print(args)
    main(args)
    print_time_statistic()
