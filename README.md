# CTF

This repository belongs to our submitted manuscript:
> **CTF**: **C**ausality-Aided **T**rade-off Analysis for Machine Learning **F**airness

## Introduction

It has been seen a surge of interest in improving fairness in machine
learning (ML) pipeline. Despite the growing number of fairness-improving
methods, there is a lack of holistic understanding of the trade-offs among
multiple factors in the ML pipeline, especially when many of them are
mutually conflicting. Such an understanding is critical for developers to
make informed decisions in the ML pipeline. Nevertheless, it is also
challenging to analyze the trade-offs with many factors involved and
coupled. 

In this paper, we propose a novel approach to introduce causality analysis
as a principled approach to analyzing trade-offs between fairness and other
critical metrics in ML pipelines. To practically and effectively
instantiate causality analysis framework in the context of
fairness-improving methods in ML pipelines, we deliver a set of
domain-specific optimizations to enable more accurate causal discovery and
design a unified interface for trade-off analysis on the basis of standard
causal inference techniques. We conduct an extensive empirical study on a
collection of widely- used fairness-improving methods with three real-world
datasets. Our study obtains actionable suggestions for users and developers
of fair ML models. As a natural extension, we further demonstrate the
versatile usage of our method in the optimal fairness-improving method
selection, paving the way for more ethical and socially responsible AI
technologies 


## Dependency

We list the main dependencies for running the code in `./requirements.txt`.

## Causal Graph

We provide the causal graph for all six scenarios in `./causal-graph/`. The
causal graph is used for causal discovery.

## Human Evaluation

Documents related to the human evaluation are in `./human-eval/`. We
present the survey questions and the results.
