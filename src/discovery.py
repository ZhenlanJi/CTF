# %%
import os
import argparse
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import *
from dibs.utils import visualize_ground_truth
from dibs.models import ErdosReniDAGDistribution, ScaleFreeDAGDistribution, BGe
from dibs.inference import JointDiBS, MarginalDiBS
from dibs.graph_utils import elwise_acyclic_constr_nograd
from jax.scipy.special import logsumexp
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]=".5"

import utils
from utils import *

METHOD_COLUMNS = ["FairMask", "Fairway", "FairSmote", "LTDD",
                  "DIR", "RW", "AdDebias","EGR", "PR", "EO", "CEO", "ROC"]


# %%
def read_data(folder: str, dataset_name: str, sensitive_name: str, 
              if_aggregate: bool = False, if_drop_mediate:bool = False) -> pd.DataFrame:
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
                        if if_aggregate:
                            df = df.groupby(["Model_Width"]).mean().reset_index()
                    else:
                        if if_aggregate:
                            df = df.groupby(["Ratio", "Model_Width"]).mean().reset_index()
                            # print(f"Aggregate {file_name} to {df.shape[0]} rows")
                            if if_drop_mediate:
                                # drop rows with ratio < 1
                                df = df[df["Ratio"] >= 1]
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
    return combined_df


def matrix_to_dgraph(matrix: np.ndarray, columns: List[str], threshold: float = 1.0) -> List[str]:
    dgraph = []
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i, j] >= threshold:
                dgraph.append(f"{columns[i]} -> {columns[j]}")
    return dgraph


def compute_expected_graph(*, dist):
    """
    Computes expected graph 

    Args:
        dist (:class:`dibs.metrics.ParticleDistribution`): particle distribution
    Returns: 
        expected Graph 
    """
    n_vars = dist.g.shape[1]

    # select acyclic graphs
    is_dag = elwise_acyclic_constr_nograd(dist.g, n_vars) == 0
    assert is_dag.sum() > 0,  "No acyclic graphs found"

    particles = dist.g[is_dag, :, :]
    log_weights = dist.logp[is_dag] - logsumexp(dist.logp[is_dag])

    # compute expected graph
    expected_g = jnp.zeros_like(particles[0])
    for i in range(particles.shape[0]):
        expected_g += jnp.exp(log_weights[i]) * particles[i, :, :]

    return expected_g



# %%
save_path= ""
dataset = "german"
sensitive_name = "sex"
rand_key = jax.random.PRNGKey(0)

# %%
# collected_data_path = os.path.join(save_path, f"{dataset}_{sensitive_name}.csv")
# if not os.path.exists(collected_data_path):
#     collected_df = read_data(save_path, dataset, sensitive_name)
# else:
#     collected_df = pd.read_csv(collected_data_path)
collected_df = read_data(save_path, dataset, sensitive_name, if_aggregate=True, if_drop_mediate=True)
# for EGR
# collected_df.drop(columns=["AE_FGSM", "AE_PGD", "MI_BlackBox"], inplace=True)
# for EGR
collected_df.replace([np.inf, -np.inf], np.nan, inplace=True)
collected_df.dropna(inplace=True)
collected_df = collected_df.sample(frac=1).reset_index(drop=True)
print(f"Collected data shape: {collected_df.shape}")
print(f"Collected data columns: {collected_df.columns}")
for m in ["FairMask", "Fairway", "FairSmote", "LTDD",
          "DIR", "RW", "AdDebias", "PR", "EO", "CEO", "ROC"]:
    if m not in collected_df.columns:
        collected_df[m] = 0

interv_df = collected_df.copy()
for col in interv_df.columns:
    if col not in METHOD_COLUMNS:
        interv_df[col] = 0
interv_mask = interv_df.values
interv_mask[interv_mask > 0] = 1
interv_mask = interv_mask.astype(int)
interv_mask = jnp.array(interv_mask)
scaler = StandardScaler()
collected_data = scaler.fit_transform(collected_df)

# model_graph = ScaleFreeDAGDistribution(collected_data.shape[1], n_edges_per_node=2)
model_graph = ErdosReniDAGDistribution(collected_data.shape[1], n_edges_per_node=2)
model = BGe(graph_dist=model_graph)
dibs = MarginalDiBS(x=collected_data, interv_mask=interv_mask, inference_model=model)


# %%
rand_key, subk = jax.random.split(rand_key)

gs = dibs.sample(key=subk, n_particles=50, steps=13000, callback_every=1000, callback=dibs.visualize_callback())
# gs = dibs.sample(key=subk, n_particles=20, steps=1500, callback_every=5000, callback=None)

# %%
dibs_output = dibs.get_mixture(gs)
# dibs_output = dibs.get_empirical(gs)
expected_g = compute_expected_graph(dist=dibs_output)


# %%
model.log_marginal_likelihood(g=expected_g, x=collected_data, interv_targets=interv_mask)

# %%
METHOD_COLUMNS = ["FairMask", "Fairway", "FairSmote", "LTDD",
                  "DIR", "RW", "AdDebias", "PR", "EO", "CEO", "ROC"]

ground_df = read_data(save_path, dataset, sensitive_name, if_aggregate=True)
ground_df.replace([np.inf, -np.inf], np.nan, inplace=True)
ground_df.dropna(inplace=True)
ground_df = ground_df.sample(frac=1).reset_index(drop=True)
print(f"Collected data shape: {ground_df.shape}")
print(f"Collected data columns: {ground_df.columns}")

ground_interv_df = ground_df.copy()
for col in ground_interv_df.columns:
    if col not in METHOD_COLUMNS:
        ground_interv_df[col] = 0
ground_interv_mask = ground_interv_df.values
ground_interv_mask[ground_interv_mask > 0] = 1
ground_interv_mask = ground_interv_mask.astype(int)
ground_interv_mask = jnp.array(ground_interv_mask)
scaler = StandardScaler()
# ground_df.drop(columns=METHOD_COLUMNS, inplace=True)
ground_data = scaler.fit_transform(ground_df)

score=model.log_marginal_likelihood(g=expected_g, x=ground_data, interv_targets=ground_interv_mask)
print("true score", score)

# %%
dibs_output.logp

# %%
visualize_ground_truth(jnp.array(expected_g), )

# %%
dgraph = matrix_to_dgraph(expected_g, collected_df.columns, threshold=0.99)
print(len(dgraph))
# dgraph = process_edges(dgraph)
for line in dgraph:
    print(line)

# %%
graph_path = os.path.join(
    '/home/zjiae/Project/Fairness-Causal-Code/data/causal_graph/', dataset+'_'+sensitive_name+'.txt')
with open(graph_path, 'r') as fin:
    origin_gfile = fin.read().splitlines()


