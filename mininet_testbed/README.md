# Mininet Live Protocol Stack Adversarial Testbed

## Module Overview and Experimental Goals

This module is explicitly engineered to bridge the gap between theoretical models and real-world network deployments. The primary objective is to validate the engineering translation capability of the RoS-ETA framework within a live Linux protocol stack. By deploying a programmable Open vSwitch edge gateway equipped with our physical-informed detection algorithm we construct a high-fidelity attack-defense prototype. This setup evaluates system robustness under actual network jitter and hardware processing constraints proving that physical attribution holds true beyond static offline datasets.

## Environment and Dependency Preparation

Executing this live testbed requires a Linux environment with root privileges due to the strict kernel-level network namespace manipulations performed by the Mininet framework. You must first ensure the installation of the core network simulation architecture along with the necessary traffic generation tools.

A Python 3 environment is required to orchestrate the software-defined topology and execute the real-time stream processing algorithms. Please install the following fundamental system packages via your package manager

* **mininet** utilized to orchestrate the virtualized network endpoints and the centralized OVS backbone
* **iperf3** deployed to generate high-entropy benign TCP business flows simulating normal human-computer interactions
* **hping3** required to simulate high-throughput volumetric DoS attacks as well as precisely throttled adaptive evasions

You can install all necessary system dependencies by executing the following terminal command

**Bash**

```
sudo apt-get update
sudo apt-get install mininet iperf3 hping3
```

## Step 1 Live Topology Construction

To initiate the evaluation we establish a streamlined yet highly representative network topology. The Python script autonomously configures two isolated endpoints acting as the vulnerable victim server and the malicious attacker client respectively. Both nodes are bridged via a central OVS switch.

This switch functions as our intelligent edge gateway intercepting traffic and continuously computing physical residuals at the millisecond level without requiring complex multi-step neural network inferences.

## Step 2 Dynamic Adversarial Scenarios

We engineered three distinct network scenarios to thoroughly assess the detection boundaries of the physical manifold under dynamic conditions.

**Bash**

```
sudo python3 run_live_demo.py
```

Upon execution the script sequentially launches a standard TCP connection utilizing iperf3 to establish a benign baseline. It is immediately followed by an aggressive SYN flood simulated by hping3 aiming to exhaust network resources. Finally an adaptive evasion attack is executed where the packet transmission interval is intentionally throttled by the attacker attempting to mathematically fit the benign temporal curves and bypass statistical thresholds.

## Scientific Discoveries and Result Evaluation

Running the automated script yields immediate physical verdicts at the millisecond level across all three adversarial scenarios.

During standard benign TCP interactions the system consistently maintains low physical residuals remaining strictly below the designated tolerance threshold. This seamlessly validates the real-time computational advantages of the logarithmic linearization operator in high-speed stream processing environments. When facing the open-loop volumetric flood the physical residual instantaneously surges and reliably triggers a decisive violation verdict proving the immediate interception capability of the algorithm against mechanical high-rate attacks.

Most importantly the adaptive evasion attempt perfectly reproduces the mandatory suppression effect of the rate paradox within a live protocol stack. Although the attacker successfully manipulates the physical residual to drop below the detection threshold this action forces their effective attack throughput to collapse catastrophically. This live demonstration conclusively proves that attackers are forced into an inescapable zero-sum game between maintaining feature stealthiness and preserving tactical destructive power.
