# %%
import warnings
import os
import pandas as pd
import random
import econml
from econml.dml import DML, LinearDML, SparseLinearDML, CausalForestDML
import numpy as np
import networkx as nx
from networkx.algorithms import tournament
import sklearn
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import (Lasso, LassoCV, LogisticRegression,
                                  LogisticRegressionCV, LinearRegression,
                                  MultiTaskElasticNet, MultiTaskElasticNetCV)
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib
from sklearn.model_selection import train_test_split
from dowhy import CausalModel
# import pgmpy
# from pgmpy.base import DAG
# from pgmpy.metrics import log_likelihood_score, correlation_score, structure_score
# from pgmpy.estimators import BDeuScore, BDsScore, BicScore, K2Score

import utils
from utils import *

METHOD_COLUMNS = ["FairMask", "Fairway", "FairSmote", "LTDD",
                  "DIR", "RW", "AdDebias", "EGR", "PR", "EO", "CEO", "ROC", ]

# disable warnings
warnings.filterwarnings("ignore")
random.seed(0)
np.random.seed(0)

# %%
def read_data(folder: str, dataset_name: str, sensitive_name: str, if_aggregate: bool = False) -> pd.DataFrame:
    obj_path = os.path.join(folder, dataset_name)
    combined_df = pd.DataFrame()
    for dir_path, _, file_names in os.walk(obj_path):
        for file_name in file_names:
            if file_name.endswith('.csv'):
                csv_name = file_name.split('.')[0]
                method_name, sn = csv_name.split('_')
                # assert method_name in METHOD_COLUMNS or method_name == "Baseline", f"Method name {method_name} not in {METHOD_COLUMNS}"
                if method_name != "Baseline" and method_name not in METHOD_COLUMNS:
                    continue
                if sn == sensitive_name:
                    file_path = os.path.join(dir_path, file_name)
                    df = pd.read_csv(file_path)
                    if method_name == "Baseline":
                        df.drop(columns=["Ratio"], inplace=True)
                        normal_df = df.copy()
                        if if_aggregate:
                            df = df.groupby(["Model_Width"]).mean().reset_index()
                    else:
                        if if_aggregate:
                            df = df.groupby(["Ratio", "Model_Width"]).mean().reset_index()
                            # print(f"Aggregate {file_name} to {df.shape[0]} rows")
                        df = df.rename(columns={"Ratio": method_name})
                    print(f"Read {file_name} with {df.shape[0]} rows")
                    for col in METHOD_COLUMNS:
                        if col not in df.columns:
                            df[col] = 0
                    expected_columns = METHOD_COLUMNS+utils.ALL_COLUMNS
                    expected_columns.remove("Ratio")
                    assert set(df.columns) == set(
                        expected_columns), f"Columns {df.columns} not equal to {METHOD_COLUMNS+utils.ALL_COLUMNS}"
                    combined_df = pd.concat([combined_df, df], axis=0, ignore_index=True)
    combined_df["MI_Rule"] = combined_df["Train_Acc"] * 0.7 + (1-combined_df["Test_Acc"]) * 0.3
    combined_df.to_csv(os.path.join(folder, f"{dataset_name}_{sensitive_name}.csv"), index=False)
    return combined_df, normal_df


def compute_ate(parent, child, data_df, ref_df, dg, T0, T1):
    parent_parents = list(dg.predecessors(parent))
    child_parents = list(dg.predecessors(child))
    X_cols = list(set(parent_parents + child_parents))
    # rq3
    if "EGR" in X_cols:
        X_cols.remove("EGR")

    if parent in X_cols:
        X_cols.remove(parent)
    X = data_df[X_cols]
    T = data_df[parent]
    Y = data_df[child]

    est = LinearDML(
        model_y=RandomForestRegressor(),
        model_t=RandomForestRegressor(),
        # featurizer=PolynomialFeatures(degree=1, include_bias=False),
        random_state=0)
    est.fit(Y, T, X=X)
    ate = est.ate(X=ref_df[X_cols], T0=T0, T1=T1)

    return ate


# %%
save_path = ""
dataset = "german"
sensitive_name = "age"

# %%
graph_path = os.path.join(
    '', dataset+'_'+sensitive_name+'.txt')
with open(graph_path, 'r') as fin:
    origin_gfile = fin.read().splitlines()
collected_df, normal_df = read_data(save_path, dataset, sensitive_name, if_aggregate=True)
collected_df.replace([np.inf, -np.inf], np.nan, inplace=True)
# collected_df.dropna(inplace=True)
collected_df.fillna(0, inplace=True)
collected_df = collected_df.sample(frac=1).reset_index(drop=True)
print(f"Collected data shape: {collected_df.shape}")
print(f"Collected data columns: {collected_df.columns}")
data_df = collected_df.copy()
# data_df = (data_df - data_df.mean()) / (data_df.std())
ref_df = normal_df.groupby(["Model_Width"]).mean().reset_index()
ref_df['MI_Rule'] = ref_df['Train_Acc'] * 0.7 + (1-ref_df['Test_Acc']) * 0.3
for m in METHOD_COLUMNS:
    ref_df[m] = 0

# for col in ref_df.columns:
#     if col in data_df.columns:
#         ref_df[col] = (ref_df[col]-data_df[col].mean())/data_df[col].std()


# %%
trace_list = []
for edge in origin_gfile:
    parent, child = edge.split(' -> ')
    trace_list.append((parent, child))

# dg = nx.DiGraph()
# dg.add_edges_from(trace_list)
# nx.is_directed_acyclic_graph(dg)

dg = nx.DiGraph()
dg.add_nodes_from(data_df.columns)
dg.add_edges_from(trace_list)

temp_graph = "digraph {" + \
    ' '.join(origin_gfile) + "}"
sorted_nodes = list(nx.topological_sort(dg))


# %%
ref_df['MI_Rule']

# %%
# improve direction
# RQ1
# rules = {
#     "Test_Acc": "+", "Test_F1": "+",
#     "Test_sens_DP": "+", "Test_sens_SPD": "+", "Test_sens_AOD": "+",
#     "Test_Cons": "+", "Test_TI": "-", "Test_CDS": "-"
# }

# # RQ2
# rules = {
#     "Test_sens_DP": "+", "Test_sens_SPD": "+", "Test_sens_AOD": "+",
#     "Test_other_DP": "+", "Test_other_SPD": "+", "Test_other_AOD": "+",
# }

# RQ3
rules = {
    "Test_sens_DP": "+", "Test_sens_SPD": "+", "Test_sens_AOD": "+",
    "Test_Cons": "+", "Test_TI": "-", "Test_CDS": "-",
    "AE_FGSM": "-", "AE_PGD": "-", "MI_BlackBox": "-", "MI_Rule": "-"
}
metrics = list(rules.keys())
fair_m = ["Test_sens_DP", "Test_sens_SPD", "Test_sens_AOD",
          "Test_Cons", "Test_TI", "Test_CDS"]
robust_m = ["AE_FGSM", "AE_PGD", "MI_BlackBox", "MI_Rule"]


# %%
fair_effect={}

for mf in METHOD_COLUMNS:
    for mt in metrics:
        if mt in nx.descendants(dg, mf):
            model = CausalModel(
                data=data_df,
                treatment=mf,
                outcome=mt,
                graph=temp_graph
            )
            identified_estimand = model.identify_effect(
                proceed_when_unidentifiable=True)

            causal_estimate = model.estimate_effect(identified_estimand,
                                                    method_name="backdoor.linear_regression",
                                                    control_value=min(data_df[mf]),
                                                    treatment_value=max(data_df[mf]),
                                                    confidence_intervals=True,
                                                    # evaluate_effect_strength=True,
                                                    test_significance=True)
            # print(f"{mf} -> {mt}: {causal_estimate.value}")
            fair_effect[(mf, mt)] = causal_estimate.value

            # model = sklearn.linear_model.LinearRegression()
            # all_cols=list(data_df.columns)
            # all_cols.remove(mt)
            # model.fit(data_df[all_cols], data_df[mt])
            # Y0 = model.predict(ref_df[all_cols]).mean()
            # new_input = ref_df[all_cols].copy()
            # new_input[mf] = 1
            # Y1 = model.predict(new_input).mean()

            # Y0 = ref_df[mt].mean()
            # Y1 = data_df[data_df[mf] == 1][mt].mean()
            # print(f"{mf} -> {mt}: {Y1-Y0}")

        else:
            # print(f"{mf} -> {mt}: No causal effect")
            fair_effect[(mf, mt)] = 0
    # print('='*20)

# %%
out_path = os.path.join(save_path, "tradeoff", dataset+'_'+sensitive_name+'_rq3.csv')

# for m_id in range(len(metrics)):
#     for n_id in range(m_id+1, len(metrics)):
        # # for rq2
        # m_sn = metrics[m_id].split('_')[1]
        # n_sn = metrics[n_id].split('_')[1]
        # if m_sn == n_sn:
        #     continue
        # metric_A = metrics[m_id]
        # metric_B = metrics[n_id]

for metric_A in fair_m:
    for metric_B in robust_m:
        print("\n"+"="*30)
        print(f"{metric_A} vs {metric_B}")
        for fair_method in METHOD_COLUMNS:
            if fair_effect[(fair_method, metric_A)] == 0 or fair_effect[(fair_method, metric_B)] == 0:
                continue
            direction_A = '+' if fair_effect[(fair_method, metric_A)] > 0 else '-'
            direction_B = '+' if fair_effect[(fair_method, metric_B)] > 0 else '-'
            improve_A = True if rules[metric_A] == direction_A else False
            improve_B = True if rules[metric_B] == direction_B else False
            if improve_A != improve_B:
                causes = []
                if metric_B in nx.descendants(dg, metric_A):
                    T0 = ref_df[metric_A].mean()
                    T1 = data_df[data_df[fair_method] == 1][metric_A].mean()
                    ate = compute_ate(metric_A, metric_B, data_df, ref_df, dg, T0, T1)
                    direction = '+' if ate > 0 else '-'
                    if_improve = True if direction == rules[metric_B] else False
                    if if_improve != improve_A:
                        causes.append(metric_A)

                elif metric_A in nx.descendants(dg, metric_B):
                    T0 = ref_df[metric_B].mean()
                    T1 = data_df[data_df[fair_method] == 1][metric_B].mean()
                    ate = compute_ate(metric_B, metric_A, data_df, ref_df, dg, T0, T1)
                    direction = '+' if ate > 0 else '-'
                    if_improve = True if direction == rules[metric_A] else False
                    if if_improve != improve_B:
                        causes.append(metric_B)
                # else:
                #     print(f"{metric_A} and {metric_B}: No causal effect")

                ancestor_A = set(list(nx.ancestors(dg, metric_A)))
                ancestor_B = set(list(nx.ancestors(dg, metric_B)))
                common_ancestor = list(ancestor_A.intersection(ancestor_B))
                common_ancestor = [x for x in common_ancestor if x not in (METHOD_COLUMNS+['EGR'])]
                # print(f"No common ancestor for {metric_A} and {metric_B}")
                if len(common_ancestor) > 0:
                    # print(f"Common ancestor: {common_ancestor}")
                    ca_last_step = dict()
                    for ca in common_ancestor:
                        toX_paths = nx.all_simple_paths(dg, source=ca, target=metric_A)
                        toX_last_step = set([x[-2] for x in toX_paths])
                        toY_paths = nx.all_simple_paths(dg, source=ca, target=metric_B)
                        toY_last_step = set([x[-2] for x in toY_paths])
                        ca_last_step[ca] = (toX_last_step, toY_last_step)
                    ca_last_step = dict(
                        sorted(
                            ca_last_step.items(),
                            key=lambda x: sorted_nodes.index(x[0]),
                            reverse=True))
                    explored_step = set()
                    potential_causes = common_ancestor.copy()
                    for ca, (toX_last_step, toY_last_step) in ca_last_step.items():
                        if toX_last_step.issubset(explored_step) and toY_last_step.issubset(
                                explored_step):
                            potential_causes.remove(ca)
                        else:
                            explored_step.update(toX_last_step)
                            explored_step.update(toY_last_step)
                    # print(f"Potential causes: {potential_causes}\n")

                    for pc in potential_causes:
                        if "other" in pc:
                            continue
                        T0 = ref_df[pc].mean()
                        T1 = data_df[data_df[fair_method] == 1][pc].mean()
                        ate_A = compute_ate(pc, metric_A, data_df, ref_df, dg, T0, T1)
                        ate_B = compute_ate(pc, metric_B, data_df, ref_df, dg, T0, T1)
                        cf_direction_A = '+' if ate_A > 0 else '-'
                        cf_direction_B = '+' if ate_B > 0 else '-'
                        cf_improve_A = True if cf_direction_A == rules[metric_A] else False
                        cf_improve_B = True if cf_direction_B == rules[metric_B] else False
                        if cf_improve_A != cf_improve_B:
                            causes.append(pc)

                for ci in range(len(causes)):
                    if "_sens_" in causes[ci]:
                        new_cause = causes[ci].replace("_sens_", "_")
                        causes[ci] = new_cause
                    if "_DP" in causes[ci]:
                        new_cause = causes[ci].replace("_DP", "_DI")
                        causes[ci] = new_cause
                print(f"{fair_method}: {causes}")
                record = {'trade-off': f"{metric_A}--{metric_B}",
                          'fair_method': fair_method,
                          'causes': "+".join(causes)}
                record_df = pd.DataFrame([record])
                record_df.to_csv(out_path, mode='a', header=not os.path.exists(out_path), index=False)

                # print(pc, end=': ')
                # print(f"{metric_A}: {'+' if ate_A > 0 else '-'}", end=' || ')
                # print(f"{metric_B}: {'+' if ate_B > 0 else '-'}")
                # print('-'*20)


# %%



