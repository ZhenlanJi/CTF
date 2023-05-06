import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import math
from typing import *
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt

from models import MyFNNetClf, FFNet, AdversaryNet


class PRemoveClf(MyFNNetClf):
    """
    Note: When eta is 0, this is equivalent to the original MyFNNetClf.
    In-processing: Prejudice Remover FFNN Classifier
    Paper: T. Kamishima, et al. Fairness-Aware Classifier with Prejudice
        Remover Regularizer, Joint European Conference on Machine Learning
        and Knowledge Discovery in Databases, 2012.
    """

    def __init__(
            self, inp_features: int, num_classes: int = 2, width: int = 4, max_epochs: int = 200,
            tol: float = 1e-4, early_stopping: bool = True, validation_fraction: float = 0.1, n_iter_no_change: int = 10,
            warm_start_epochs: int = 50, early_stopping_threshold: float = 0.01, device: Optional[str] = None,
            verbose: bool = True, sensitive_name: str = "", eta: float = 0.1, l2_reg: float = 0.0) -> None:
        super().__init__(inp_features, num_classes, width, max_epochs,
                         tol, early_stopping, validation_fraction,
                         n_iter_no_change, warm_start_epochs, early_stopping_threshold,
                         device, verbose)
        # assert sensitive_name != "", "Sensitive attribute must be specified"
        self.sensitive_name = sensitive_name
        self.eta = eta
        self.l2_reg = l2_reg

    def fit(self, x: Union[pd.core.frame.DataFrame, pd.core.series.Series, np.ndarray],
            y: Union[pd.core.frame.DataFrame, pd.core.series.Series, np.ndarray],
            sample_weight: Union[pd.core.series.Series, np.ndarray, None] = None,
            sensitive_index: int = -1) -> None:
        # x: must be pandas dataframe, otherwise sensitive_index must be specified
        # y: pandas series or numpy array
        # sample_weight: pandas series or numpy array
        # get the sensitive attribute column number
        if sensitive_index == -1:
            assert type(
                x) == pd.core.frame.DataFrame, "Sensitive attribute column number must be specified, otherwise x must be pandas dataframe"
            assert self.sensitive_name != "", "Sensitive attribute must be specified"
            sensitive_index = x.columns.get_loc(self.sensitive_name)
        if type(x) == pd.core.frame.DataFrame or type(x) == pd.core.series.Series:
            x = x.values
        if type(y) == pd.core.frame.DataFrame or type(y) == pd.core.series.Series:
            y = y.values
        if sample_weight is not None:
            if type(sample_weight) == pd.core.series.Series:
                sample_weight = sample_weight.values
            sample_weight = torch.from_numpy(sample_weight).float().to(self.device)

        if self.validation_fraction > 0:
            # split into train and validation
            if sample_weight is not None:
                x_train, x_val, y_train, y_val, sw_train, sw_val = train_test_split(
                    x, y, sample_weight, shuffle=True, test_size=self.validation_fraction)
                sw_train = torch.from_numpy(sw_train).float().to(self.device)
                sw_val = torch.from_numpy(sw_val).float().to(self.device)
            else:
                x_train, x_val, y_train, y_val = train_test_split(
                    x, y, shuffle=True, test_size=self.validation_fraction)
                sw_train = None
                sw_val = None

            # convert to torch tensors
            x_train = torch.from_numpy(x_train).float().to(self.device)
            y_train = torch.from_numpy(y_train).long().to(self.device)
            x_val = torch.from_numpy(x_val).float().to(self.device)
            y_val = torch.from_numpy(y_val).long().to(self.device)
        else:
            x_train = torch.from_numpy(x).float().to(self.device)
            y_train = torch.from_numpy(y).long().to(self.device)

        train_num = y_train.shape[0]
        # count the number of samples when sensitive attribute is 0 and 1
        num_s0 = torch.sum(x_train[:, sensitive_index] == 0).item()
        num_s1 = torch.sum(x_train[:, sensitive_index] == 1).item()
        assert train_num == num_s0 + num_s1, "Sensitive attribute column must be binary"

        for epoch in range(self.max_epochs):
            self.model.train()

            self.optimizer.zero_grad()
            output = self.model(x_train)
            loss = self.loss_fn(output, y_train, reduction='none')
            if sample_weight is not None:
                loss = loss * sw_train
            loss = loss.mean()
            # add L2 regularization
            l2_norm = torch.tensor(0.).to(self.device)
            for param in self.model.parameters():
                l2_norm += torch.linalg.norm(param)
            loss += self.l2_reg * l2_norm
            # add the prejudice remover loss
            sens_attr = x_train[:, sensitive_index].detach().clone()
            pred_prob = F.softmax(output, dim=1)
            pr_y0 = torch.sum(pred_prob[:, 0])/train_num
            pr_y1 = torch.sum(pred_prob[:, 1])/train_num
            pr_y0_cond_s0 = torch.sum(
                pred_prob[x_train[:, sensitive_index] == 0, 0])/num_s0
            pr_y0_cond_s1 = torch.sum(
                pred_prob[x_train[:, sensitive_index] == 1, 0])/num_s1
            pr_y1_cond_s0 = torch.sum(
                pred_prob[x_train[:, sensitive_index] == 0, 1])/num_s0
            pr_y1_cond_s1 = torch.sum(
                pred_prob[x_train[:, sensitive_index] == 1, 1])/num_s1
            pr_loss = (1-sens_attr) * (pred_prob[:, 0]*torch.log(pr_y0_cond_s0/pr_y0) + pred_prob[:, 1]*torch.log(pr_y1_cond_s0/pr_y1)) + \
                sens_attr * (pred_prob[:, 0]*torch.log(pr_y0_cond_s1/pr_y0) + pred_prob[:, 1]*torch.log(pr_y1_cond_s1/pr_y1))

            loss += self.eta * pr_loss.sum()
            loss.backward()
            self.optimizer.step()
            self.loss_list.append(loss.item())
            if self.verbose and self.validation_fraction == 0:
                print(f"Epoch {epoch}: Loss {loss.item()}")

            if len(self.loss_list) > self.n_iter_no_change:
                tmp_best_loss = np.min(
                    self.loss_list[-self.n_iter_no_change:-2])
                if tmp_best_loss-self.loss_list[-1] < self.tol:
                    break

            if self.validation_fraction > 0:
                # val_score = self.score(x_val, y_val)
                # val_loss = self.loss_fn(
                #     self.model(x_val), y_val)
                val_score, val_loss = self.evaluate(x_val, y_val, sw_val)
                if self.verbose:
                    print(
                        f"Epoch {epoch}: Train Loss {loss.item()}, Val Loss {val_loss}, Val Score {val_score}")
                self.val_loss_list.append(val_loss)
                if self.early_stopping:
                    if epoch > self.warm_start_epochs:
                        if self.val_score_list[-1] - val_score > 0.1:
                            # print("Early stopping")
                            break
                self.val_score_list.append(val_score)

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        return super().get_params(deep) | {
            "sensitive_name": self.sensitive_name, "eta": self.eta,
            "l2_regularization": self.l2_reg,
        }
    


class AdDebiasClf(MyFNNetClf):
    """
    In-processing: Adversarial debiasing classifier
    paper: B Zhang, et al. Mitigating Unwanted Biases with Adversarial
        Learning. AAAI/ACM Conference on Artificial Intelligence, Ethics,
        and Society, 2018.
    """

    def __init__(self, inp_features: int,
                 num_classes: int = 2, width: int = 4,
                 max_epochs: int = 200, tol: float = 1e-4,
                 early_stopping: bool = True,
                 validation_fraction: float = 0.1,
                 n_iter_no_change: int = 10,
                 warm_start_epochs: int = 50,
                 early_stopping_threshold: float = 0.01,
                 device: Optional[str] = None, verbose: bool = True,
                 sensitive_name: str = "",
                 adversary_loss_weight: float = 0.1,
                 pre_clf_epochs: int = 100,
                 pre_adv_epochs: int = 50,
                 ) -> None:
        super().__init__(inp_features, num_classes, width, max_epochs,
                         tol, early_stopping, validation_fraction,
                         n_iter_no_change, warm_start_epochs,
                         early_stopping_threshold,
                         device, verbose)
        # assert sensitive_name != "", "Sensitive attribute must be specified"
        self.sensitive_name = sensitive_name
        self.adversary_loss_weight = adversary_loss_weight
        self.adversary = AdversaryNet(
            # inp_features=num_classes+1
        ).to(self.device)
        self.adv_optimizer = torch.optim.Adam(
            self.adversary.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08,
        )
        self.adv_loss_fn = F.binary_cross_entropy_with_logits
        self.pre_clf_epochs = pre_clf_epochs
        self.pre_adv_epochs = pre_adv_epochs
        self.sum_loss_list: List[float] = []

    def fit(self, x: Union[pd.core.frame.DataFrame, pd.core.series.Series, np.ndarray],
            y: Union[pd.core.frame.DataFrame, pd.core.series.Series, np.ndarray],
            sample_weight: Union[pd.core.series.Series, np.ndarray, None] = None,
            sensitive_index: int = -1) -> None:
        # x: must be pandas dataframe, otherwise sensitive_index must be specified
        # y: pandas series or numpy array
        # sample_weight: pandas series or numpy array
        # get the sensitive attribute column number
        if sensitive_index == -1:
            assert type(
                x) == pd.core.frame.DataFrame, "Sensitive attribute column number must be specified, otherwise x must be pandas dataframe"
            assert self.sensitive_name != "", "Sensitive attribute must be specified"
            sensitive_index = x.columns.get_loc(self.sensitive_name)

        if type(x) == pd.core.frame.DataFrame or type(x) == pd.core.series.Series:
            x = x.values
        if type(y) == pd.core.frame.DataFrame or type(y) == pd.core.series.Series:
            y = y.values
        if sample_weight is not None:
            if type(sample_weight) == pd.core.series.Series:
                sample_weight = sample_weight.values
            sample_weight = torch.from_numpy(sample_weight).float().to(self.device)

        if self.validation_fraction > 0:
            # split into train and validation
            if sample_weight is not None:
                x_train, x_val, y_train, y_val, sw_train, sw_val = train_test_split(
                    x, y, sample_weight, shuffle=True, test_size=self.validation_fraction)
                sw_train = torch.from_numpy(sw_train).float().to(self.device)
                sw_val = torch.from_numpy(sw_val).float().to(self.device)
            else:
                x_train, x_val, y_train, y_val = train_test_split(
                    x, y, shuffle=True, test_size=self.validation_fraction)
                sw_train = None
                sw_val = None

            # convert to torch tensors
            x_train = torch.from_numpy(x_train).float().to(self.device)
            y_train = torch.from_numpy(y_train).long().to(self.device)
            x_val = torch.from_numpy(x_val).float().to(self.device)
            y_val = torch.from_numpy(y_val).long().to(self.device)
        else:
            x_train = torch.from_numpy(x).float().to(self.device)
            y_train = torch.from_numpy(y).long().to(self.device)

        adv_y_train = x_train[:, sensitive_index].detach().clone().view(-1)
        adv_y_val = x_val[:, sensitive_index].detach().clone().view(-1)

        # pre-train the model
        for epoch in range(self.pre_clf_epochs):
            self.model.train()
            self.optimizer.zero_grad()
            output = self.model(x_train)
            loss = self.loss_fn(output, y_train, reduction='none')
            if sample_weight is not None:
                loss = loss * sw_train
            loss = loss.mean()
            loss.backward()
            self.optimizer.step()
            # self.loss_list.append(loss.item())
            if self.verbose and self.validation_fraction == 0:
                print(f"Pre-train classifier Epoch {epoch}: Loss {loss.item()}")
            if self.validation_fraction > 0:
                val_score, val_loss = self.evaluate(x_val, y_val, sw_val)
                if self.verbose:
                    print(
                        f"Pre-train classifier Epoch {epoch}: Train Loss {loss.item()}, Val Loss {val_loss}, Val Score {val_score}")

        # pre-train the adversary
        for epoch in range(self.pre_adv_epochs):
            self.adversary.train()
            self.adv_optimizer.zero_grad()
            with torch.no_grad():
                pred_logits = self.model(x_train)
            pred_probs = F.softmax(pred_logits, dim=1)
            adv_inp = torch.cat((pred_probs[:, 1].view(-1, 1), y_train.view(-1, 1)), dim=1)
            adv_output = self.adversary(adv_inp)
            adv_loss = self.adv_loss_fn(
                adv_output, adv_y_train.view(adv_output.shape),
                reduction='none')
            if sample_weight is not None:
                adv_loss = adv_loss * sw_train
            adv_loss = adv_loss.mean()
            adv_loss.backward()
            self.adv_optimizer.step()
            # self.adv_loss_list.append(adv_loss.item())
            if self.verbose and self.validation_fraction == 0:
                print(f"Pre-train adversary Epoch {epoch}: Loss {loss.item()}")
            if self.validation_fraction > 0:
                val_score, val_loss = self.adv_evaluate(x_val, adv_y_val, sw_val)
                if self.verbose:
                    print(
                        f"Pre-train adversary Epoch {epoch}: Train Loss {adv_loss.item()}, Val Loss {val_loss}, Val Score {val_score}")

        # Adversary debiasing
        def normalize(x): return x / (torch.norm(x) + np.finfo(np.float32).tiny)
        for epoch in range(self.max_epochs):
            self.model.train()
            self.optimizer.zero_grad()

            # with torch.no_grad():
            pred_logits = self.model(x_train)
            # pred_probs = F.softmax(pred_logits, dim=1)
            clf_loss = self.loss_fn(pred_logits, y_train, reduction='none')
            if sample_weight is not None:
                clf_loss = clf_loss * sw_train
            clf_loss = clf_loss.mean()
            clf_grad = torch.autograd.grad(clf_loss, self.model.parameters(),)

            self.optimizer.zero_grad()
            pred_logits = self.model(x_train)
            pred_probs = F.softmax(pred_logits, dim=1)
            adv_inp = torch.cat((pred_probs[:, 1].view(-1, 1), y_train.view(-1, 1)), dim=1)
            adv_output = self.adversary(adv_inp)
            adv_loss = self.adv_loss_fn(
                adv_output, adv_y_train.view(adv_output.shape),
                reduction='none')
            if sample_weight is not None:
                adv_loss = adv_loss * sw_train
            adv_loss = adv_loss.mean()
            adv_grad = torch.autograd.grad(
                adv_loss, self.model.parameters(),)

            # for index, (grad, var) in enumerate(
            #         zip(clf_grad, self.model.parameters())):
            #     unit_adv_grad = normalize(adv_grad[index])
            #     grad -= torch.sum(grad * unit_adv_grad) * unit_adv_grad
            #     grad -= self.adversary_loss_weight * adv_grad[index]
            #     new_clf_grads.append((grad, var))
            # self.optimizer.zero_grad()
            # self.optimizer.step(new_clf_grads)
            # self.loss_list.append(clf_loss.item())

            with torch.no_grad():
                for index, para in enumerate(self.model.parameters()):
                    unit_adv_grad = normalize(adv_grad[index])
                    para.grad = clf_grad[index] - torch.sum(clf_grad[index]
                                                            * unit_adv_grad) * unit_adv_grad
                    para.grad -= self.adversary_loss_weight * adv_grad[index]
            self.optimizer.step()
            self.sum_loss_list.append(clf_loss.item()+adv_loss.item())

            self.adversary.train()
            self.adv_optimizer.zero_grad()
            with torch.no_grad():
                pred_logits = self.model(x_train)
            pred_probs = F.softmax(pred_logits, dim=1)
            adv_inp = torch.cat((pred_probs[:, 1].view(-1, 1), y_train.view(-1, 1)), dim=1)
            adv_output = self.adversary(adv_inp)
            new_adv_loss = self.adv_loss_fn(
                adv_output, adv_y_train.view(adv_output.shape),
                reduction='none')
            if sample_weight is not None:
                new_adv_loss = new_adv_loss * sw_train
            new_adv_loss = new_adv_loss.mean()
            new_adv_loss.backward()
            self.adv_optimizer.step()
            # self.adv_loss_list.append(new_adv_loss.item())

            if self.verbose and self.validation_fraction == 0:
                print(
                    f"(Adversarial Debiasing) epoch {epoch}: Loss {clf_loss.item()}, Adv Loss {new_adv_loss.item()}")

            if self.validation_fraction > 0:
                # val_score = self.score(x_val, y_val)
                # val_loss = self.loss_fn(
                #     self.model(x_val), y_val)
                val_score, val_loss = self.evaluate(x_val, y_val, sw_val)
                fairness = self.calc_eo(x_val, y_val, sensitive_index)
                val_score -= fairness
                if self.verbose:
                    print(
                        f"(Adversarial Debiasing) epoch {epoch}: Train Loss {clf_loss.item()}, Adv Loss {new_adv_loss.item()}, Val Loss {val_loss}, Val Score {val_score}")
                self.val_loss_list.append(val_loss)
                if self.early_stopping:
                    if epoch > self.warm_start_epochs:
                        if self.val_score_list[-1] - val_score > 0.01:
                            # print("Early stopping")
                            break
                self.val_score_list.append(val_score)

    def get_params(self, deep=True):
        return super().get_params(deep) | {
            'adversary_loss_weight': self.adversary_loss_weight,
            'pretrain_classifier_epochs': self.pre_clf_epochs,
            'pretrain_adversary_epochs': self.pre_adv_epochs,
        }

    def get_adversary(self) -> nn.Module:
        return self.adversary

    def save_model(self, path: str) -> None:
        # get current time (mm-dd-hh-mm-ss)
        now = datetime.now()
        dt_string = now.strftime("%m-%d-%H-%M-%S")
        torch.save(self.model.state_dict(), os.path.join(path, dt_string+".pt"))
        torch.save(self.adversary.state_dict(), os.path.join(path, dt_string+"_adv.pt"))

    def load_model(self, path: str) -> None:
        self.model.load_state_dict(torch.load(path))
        model_name = path.split("/")[-1].split(".")[0]
        self.adversary.load_state_dict(torch.load(path.replace(model_name, model_name+"_adv")))

    def adv_evaluate(self, x: Union[pd.core.frame.DataFrame, pd.core.series.Series, np.ndarray],
                     y: Union[pd.core.frame.DataFrame, pd.core.series.Series, np.ndarray],
                     sample_weight: Union[pd.core.series.Series, np.ndarray, torch.Tensor, None] = None) -> Tuple[float, float]:
        if type(x) == pd.core.frame.DataFrame or type(x) == pd.core.series.Series:
            x = x.values
        if type(y) == pd.core.frame.DataFrame or type(y) == pd.core.series.Series:
            y = y.values
        if type(x) == np.ndarray:
            x = torch.from_numpy(x).float().to(self.device)
            y = torch.from_numpy(y).long().to(self.device)
        if sample_weight is not None:
            if type(sample_weight) == pd.core.series.Series:
                sample_weight = sample_weight.values
            if type(sample_weight) == np.ndarray:
                sample_weight = torch.from_numpy(sample_weight).float().to(self.device)

        self.adversary.eval()
        with torch.no_grad():
            pred_logits = self.model(x)
            pred_probs = F.softmax(pred_logits, dim=1)
            adv_inp = torch.cat((pred_probs[:, 1].view(-1, 1), y.view(-1, 1)), dim=1)
            adv_output = self.adversary(adv_inp)
            # sigmod and round to get prediction
            pred = torch.round(torch.sigmoid(adv_output))
            correct = pred.eq(y.view_as(pred)).sum().item()
            adv_loss = self.adv_loss_fn(
                adv_output, y.view(adv_output.shape),
                reduction='none')
            if sample_weight is not None:
                adv_loss = adv_loss * sample_weight
            adv_loss = adv_loss.mean()
        return correct / len(y), adv_loss.item()

    def calc_dp(self, inputs: torch.Tensor, sensitive_index: int) -> float:
        self.model.eval()
        with torch.no_grad():
            output = self.model(inputs)
            preds = output.argmax(dim=1, keepdim=True)
            inputs = inputs.detach().cpu().numpy()
            preds = preds.detach().cpu().numpy()
            num_s0 = np.sum(inputs[:, sensitive_index] == 0)
            num_s1 = np.sum(inputs[:, sensitive_index] == 1)
            fav_s0 = np.sum(preds[inputs[:, sensitive_index] == 0] == 1)
            fav_s1 = np.sum(preds[inputs[:, sensitive_index] == 1] == 1)
            dp = abs(fav_s0/num_s0-fav_s1/num_s1)
        return dp

    def calc_eo(self, inputs: torch.Tensor, labels: torch.Tensor, sensitive_index: int) -> float:
        self.model.eval()
        with torch.no_grad():
            output = self.model(inputs)
            preds = output.argmax(dim=1, keepdim=True)
            inputs = inputs.detach().cpu().numpy()
            preds = preds.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()
            allPos_s0 = np.sum(labels[inputs[:, sensitive_index] == 0] == 1)
            allPos_s1 = np.sum(labels[inputs[:, sensitive_index] == 1] == 1)
            truePos_s0 = np.sum(np.logical_and(
                (preds[inputs[:, sensitive_index] == 0] == 1).ravel(),
                (labels[inputs[:, sensitive_index] == 0] == 1).ravel()))
            truePos_s1 = np.sum(np.logical_and(
                (preds[inputs[:, sensitive_index] == 1] == 1).ravel(),
                (labels[inputs[:, sensitive_index] == 1] == 1).ravel()))
            eo = abs(truePos_s0/allPos_s0-truePos_s1/allPos_s1)
        return eo
