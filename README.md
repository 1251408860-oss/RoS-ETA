# RoS-ETA: Physics-Informed Robustness for NIDS

This repository contains the official implementation and artifacts for the paper **"RoS-ETA: Physics-Informed Robustness for NIDS"**. 

To facilitate reproducibility and provide a clear structure for reviewers, we have decoupled the project into three core modules. Please navigate to the respective directories below based on the experiments you wish to reproduce.

## 📂 Repository Navigation & Artifact Mapping

| Paper Section | Experiment Description | Directory Link |
| :--- | :--- | :--- |
| **Section 4.2** | Large-scale physical manifold empirical analysis on MAWI backbone traffic |  [`mawi_empirical`](./mawi_empirical) |
| **Section 4.3 & 4.4** | Core dual-view robust regression algorithm and white-box evasion simulation |  [`core_experiments`](./core_experiments) |
| **Section 4.5** | Live protocol stack defense and rate-blocking testbed based on Mininet |  [`mininet_testbed`](./mininet_testbed) |

## 🛠️ Global Environment Overview
The core modules of this framework are built upon Python 3.9. Specific dependencies and Mininet topology configurations are meticulously documented within the `README.md` of each respective subdirectory.
