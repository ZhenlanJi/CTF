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


class FFNet(nn.Module):
    def __init__(
            self, inp_features: int, num_classes: int = 2, width: int = 4) -> None:
        super().__init__()
        self.fc1 = nn.Linear(inp_features, width*16)
        self.fc2 = nn.Linear(width*16, width*8)
        self.fc3 = nn.Linear(width*8, width*4)
        self.fc4 = nn.Linear(width*4, width*2)
        self.fc5 = nn.Linear(width*2, width)
        self.fc6 = nn.Linear(width, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)
        return x


class AdversaryNet(nn.Module):
    def __init__(self, inp_features: int = 2, num_classes: int = 1,) -> None:
        super().__init__()
        self.fc1 = nn.Linear(inp_features, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 4)
        self.fc4 = nn.Linear(4, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

    # def __init__(self,) -> None:
    #     super().__init__()
    #     self.c = nn.Parameter(torch.tensor(1.0,requires_grad=True))
    #     self.w1 = nn.Parameter(torch.randn(3, 1, requires_grad=True))
    #     self.b1 = nn.Parameter(torch.tensor(0.0, requires_grad=True))

    # def forward(self, x):
    #     pred_logits = x[:, 0].view(-1, 1)
    #     true_labels = x[:, 1].view(-1, 1)
    #     s = torch.sigmoid((1 + torch.abs(self.c)) * pred_logits)
    #     ret = torch.matmul(
    #         torch.cat(
    #             [s, s*true_labels, s*(1.0-true_labels)], dim=1),
    #         self.w1) + self.b1
    #     return ret


class MyFNNetClf(object):
    def __init__(
            self, inp_features: int, num_classes: int = 2, width: int = 4,
            max_epochs: int = 200, tol: float = 1e-4,
            early_stopping: bool = True, validation_fraction: float = 0.1,
            n_iter_no_change: int = 10, warm_start_epochs: int = 50,
            early_stopping_threshold: float = 0.01, device: Optional[str] = None,
            verbose: bool = True) -> None:
        self.inp_features = inp_features
        self.num_classes = num_classes
        self.width = width
        self.model = FFNet(self.inp_features, self.num_classes, self.width)
        self.loss_fn = F.cross_entropy
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08,
        )
        self.max_epochs = max_epochs
        self.tol = tol
        self.verbose = verbose
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.warm_start_epochs = warm_start_epochs
        self.early_stopping_threshold = early_stopping_threshold
        if validation_fraction <= 0:
            self.early_stopping = False
        if device is not None:
            self.device = torch.device(
                'cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device('cpu')
        self.model = self.model.to(self.device)
        self.loss_list: List[float] = []
        self.val_score_list: List[float] = []
        self.val_loss_list: List[float] = []

    def fit(self, x: Union[pd.core.frame.DataFrame, pd.core.series.Series, np.ndarray],
            y: Union[pd.core.frame.DataFrame, pd.core.series.Series, np.ndarray],
            sample_weight: Union[pd.core.series.Series, np.ndarray, None] = None) -> None:
        if type(x) == pd.core.frame.DataFrame or type(x) == pd.core.series.Series:
            x = x.values
        if type(y) == pd.core.frame.DataFrame or type(y) == pd.core.series.Series:
            y = y.values
        if type(sample_weight) == pd.core.series.Series:
            sample_weight = sample_weight.values

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

        for epoch in range(self.max_epochs):
            self.model.train()

            self.optimizer.zero_grad()
            output = self.model(x_train)
            loss = self.loss_fn(output, y_train, reduction='none')
            if sample_weight is not None:
                loss = loss * sw_train
            loss = loss.mean()
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

                # if self.early_stopping and len(self.val_loss_list) > self.n_iter_no_change:
                #     tmp_best_val_loss = np.min(
                #         self.val_loss_list[-self.n_iter_no_change:-2])
                #     tmp_best_val_score = np.max(
                #         self.val_score_list[-self.n_iter_no_change:-2])
                #     if tmp_best_val_loss-self.val_loss_list[-1] < 0 and \
                #             self.val_score_list[-1]-tmp_best_val_score < 0:
                #         print("Early stopping")
                #         break

    def predict(self, x: Union[pd.core.frame.DataFrame, pd.core.series.Series, np.ndarray]) -> np.ndarray:
        if type(x) == pd.core.frame.DataFrame or type(x) == pd.core.series.Series:
            x = x.values
        if type(x) == np.ndarray:
            x = torch.from_numpy(x).float().to(self.device)
        self.model.eval()
        with torch.no_grad():
            output = self.model(x)
            pred = output.argmax(dim=1, keepdim=True)
        return pred.detach().cpu().numpy()

    def predict_proba(self, x: Union[pd.core.frame.DataFrame, pd.core.series.Series, np.ndarray]) -> np.ndarray:
        if type(x) == pd.core.frame.DataFrame or type(x) == pd.core.series.Series:
            x = x.values
        if type(x) == np.ndarray:
            x = torch.from_numpy(x).float().to(self.device)
        self.model.eval()
        with torch.no_grad():
            output = self.model(x)
            pred = output.softmax(dim=1)
            pred = pred[:, 1].view(-1, 1)
        return pred.detach().cpu().numpy()

    def evaluate(self, x: Union[pd.core.frame.DataFrame, pd.core.series.Series, np.ndarray],
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

        self.model.eval()
        with torch.no_grad():
            output = self.model(x)
            pred = output.argmax(dim=1, keepdim=True)
            correct = pred.eq(y.view_as(pred)).sum().item()
            loss = self.loss_fn(output, y, reduction='none')
            if sample_weight is not None:
                loss = loss * sample_weight
            loss = loss.mean()
        return correct / len(y), loss.item()

    def score(self, x: Union[pd.core.frame.DataFrame, pd.core.series.Series, np.ndarray],
              y: Union[pd.core.frame.DataFrame, pd.core.series.Series, np.ndarray]) -> float:
        if type(x) == pd.core.frame.DataFrame or type(x) == pd.core.series.Series:
            x = x.values
        if type(y) == pd.core.frame.DataFrame or type(y) == pd.core.series.Series:
            y = y.values
        if type(x) == np.ndarray:
            x = torch.from_numpy(x).float().to(self.device)
            y = torch.from_numpy(y).long().to(self.device)

        self.model.eval()
        with torch.no_grad():
            output = self.model(x)
            pred = output.argmax(dim=1, keepdim=True)
            correct = pred.eq(y.view_as(pred)).sum().item()
        return correct / len(y)

    def f1_score(self, x: Union[pd.core.frame.DataFrame, pd.core.series.Series, np.ndarray],
                 y: Union[pd.core.frame.DataFrame, pd.core.series.Series, np.ndarray]) -> float:
        if type(x) == pd.core.frame.DataFrame or type(x) == pd.core.series.Series:
            x = x.values
        if type(y) == pd.core.frame.DataFrame or type(y) == pd.core.series.Series:
            y = y.values
        if type(x) == np.ndarray:
            x = torch.from_numpy(x).float().to(self.device)
            y = torch.from_numpy(y).long().to(self.device)

        self.model.eval()
        with torch.no_grad():
            output = self.model(x)
            pred = output.argmax(dim=1, keepdim=True)
            _, _, fscore, _ = precision_recall_fscore_support(
                y.detach().cpu().numpy(),
                pred.detach().cpu().numpy(),
                average='binary')
        return fscore

    def compute_loss(self, x: Union[pd.core.frame.DataFrame, pd.core.series.Series, np.ndarray],
                     y: Union[pd.core.frame.DataFrame, pd.core.series.Series, np.ndarray],
                     sample_weight: Union[np.ndarray, pd.core.series.Series, torch.Tensor, None] = None) -> float:
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

        self.model.eval()
        with torch.no_grad():
            output = self.model(x)
            loss = self.loss_fn(output, y, reduction='none')
            if sample_weight is not None:
                loss = loss * sample_weight
            loss = loss.mean()
        return loss.item()

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        return {
            "inp_features": self.inp_features, "num_classes": self.num_classes,
            "width": self.width, "max_epochs": self.max_epochs,
            "verbose": self.verbose, "early_stopping": self.early_stopping,
            "early_stopping_threshold": self.early_stopping_threshold, "tol": self.tol,
            "validation_fraction": self.validation_fraction, "n_iter_no_change": self.n_iter_no_change,
            "warm_start_epochs": self.warm_start_epochs, "device": None if self.device == torch.device('cpu') else 'gpu',
        }
    
    def print_params(self) -> None:
        print(self.get_params())

    def get_model(self) -> nn.Module:
        return self.model

    def save_model(self, path: str) -> None:
        # get current time (mm-dd-hh-mm-ss)
        now = datetime.now()
        dt_string = now.strftime("%m-%d-%H-%M-%S")
        torch.save(self.model.state_dict(), os.path.join(path, dt_string+".pt"))

    def load_model(self, path: str) -> None:
        self.model.load_state_dict(torch.load(path))

    def plot_val_score(self) -> None:
        plt.plot(self.val_score_list)
        plt.xlabel("Epoch")
        plt.ylabel("Validation score")
        plt.show()

    def plot_loss(self) -> None:
        if self.val_loss_list:
            plt.plot(self.loss_list, label="Train loss")
            plt.plot(self.val_loss_list, label="Validation loss")
            plt.legend()
        else:
            plt.plot(self.loss_list)

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.show()
