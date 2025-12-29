# Formal Problem Formulation: Multi-Agent Smart Grid Optimization

## Mathematical Framework

This document provides a rigorous mathematical formulation of the Smart Grid Multi-Agent Reinforcement Learning (MARL) problem.

---

## 1. Problem Definition

We model the smart grid energy management problem as a **Decentralized Partially Observable Markov Decision Process (Dec-POMDP)**.

### 1.1 Dec-POMDP Tuple

The problem is defined by the tuple:

$$\langle \mathcal{N}, \mathcal{S}, \{\mathcal{A}_i\}_{i \in \mathcal{N}}, \mathcal{T}, \{R_i\}_{i \in \mathcal{N}}, \{\mathcal{O}_i\}_{i \in \mathcal{N}}, \mathcal{Z}, \gamma \rangle$$

Where:
- $\mathcal{N} = \{1, 2, ..., n\}$ — Set of **n** agents (neighborhoods)
- $\mathcal{S}$ — Global state space
- $\mathcal{A}_i$ — Action space for agent $i$
- $\mathcal{T}: \mathcal{S} \times \mathcal{A} \rightarrow \Delta(\mathcal{S})$ — State transition function
- $R_i: \mathcal{S} \times \mathcal{A} \rightarrow \mathbb{R}$ — Reward function for agent $i$
- $\mathcal{O}_i$ — Observation space for agent $i$
- $\mathcal{Z}: \mathcal{S} \times \mathcal{A} \rightarrow \Delta(\mathcal{O})$ — Observation function
- $\gamma \in [0, 1)$ — Discount factor

---

## 2. State Space

### 2.1 Global State

The global state at time $t$ is:

$$s_t = (E_t, D_t, G_t, P_t, \tau_t)$$

Where:
- $E_t = (e_1^t, e_2^t, ..., e_n^t)$ — Energy levels for all agents
- $D_t = (d_1^t, d_2^t, ..., d_n^t)$ — Current demand for all agents
- $G_t = (g_1^t, g_2^t, ..., g_n^t)$ — Solar generation for all agents
- $P_t \in \mathbb{R}^+$ — Current electricity price
- $\tau_t \in [0, 1]$ — Normalized time of day

### 2.2 Local Observation

Each agent $i$ observes a partial view of the state:

$$o_i^t = (e_i^t, d_i^t, g_i^t, \sigma_t, P_t, \tau_t, \{e_j^t\}_{j \in \mathcal{N}})$$

Where:
- $e_i^t \in [0, C_{max}]$ — Own battery level (kWh)
- $d_i^t \in \mathbb{R}^+$ — Current demand (kWh)
- $g_i^t \in \mathbb{R}^+$ — Solar generation (kWh)
- $\sigma_t \in [0, 1]$ — Grid stability indicator
- $P_t \in [0, 1]$ — Normalized electricity price
- $\tau_t \in [0, 1]$ — Time of day (normalized)
- $\{e_j^t\}$ — Neighbor energy levels

**Observation dimension:** $|\mathcal{O}_i| = 6 + n$

---

## 3. Action Space

Each agent $i$ takes continuous actions:

$$a_i^t = (f_1^t, f_2^t, ..., f_n^t, g^t) \in [-1, 1]^{n+1}$$

Where:
- $f_j^t \in [-1, 1]$ — Energy transfer intention to agent $j$
  - Negative: Send energy to agent $j$
  - Positive: Request energy from agent $j$
- $g^t \in [-1, 1]$ — Grid interaction
  - Positive: Buy from grid
  - Negative: Sell to grid

**Action dimension:** $|\mathcal{A}_i| = n + 1$

### 3.1 Action Constraints

Transfers are bounded by:
- Maximum transfer rate: $T_{max}$ kWh per timestep
- Available energy: $e_i^t$
- Receiver capacity: $C_{max} - e_j^t$

Actual transfer amount:
$$\text{transfer}_{i \rightarrow j} = \min(|f_j^t| \cdot T_{max}, e_i^t, C_{max} - e_j^t)$$

---

## 4. State Transition Dynamics

### 4.1 Energy Level Update

For each agent $i$, the energy level evolves as:

$$e_i^{t+1} = \text{clip}\left(e_i^t + g_i^t - d_i^t + \sum_{j \neq i} \text{net\_transfer}_{j \rightarrow i} + \text{grid}\_\text{trade}_i^t, 0, C_{max}\right)$$

Where:
- $g_i^t$ — Solar generation
- $d_i^t$ — Energy demand
- $\text{net\_transfer}_{j \rightarrow i}$ — Net energy received from agent $j$
- $\text{grid}\_\text{trade}_i^t$ — Energy bought/sold from grid

### 4.2 Demand and Generation Patterns

**Demand model** (24-hour cycle):
$$d_i^t = d_{base} + d_{amp} \cdot \sin\left(\frac{2\pi(\tau_t \cdot 24 - 6)}{24}\right) + d_{evening} \cdot \exp\left(-\frac{(\tau_t \cdot 24 - 19)^2}{8}\right) + \epsilon_d$$

**Solar generation model:**
$$g_i^t = \begin{cases}
G_{max} \cdot \sin\left(\frac{\pi(\tau_t \cdot 24 - 6)}{12}\right) \cdot \eta_{cloud} & \text{if } 6 \leq \tau_t \cdot 24 \leq 18 \\
0 & \text{otherwise}
\end{cases}$$

Where $\eta_{cloud} \sim \mathcal{U}(0.7, 1.0)$ models cloud cover.

---

## 5. Reward Function

Each agent receives a reward composed of multiple objectives:

$$R_i(s_t, a_t) = w_1 \cdot R_{stability} + w_2 \cdot R_{efficiency} + w_3 \cdot R_{cost} + w_4 \cdot R_{cooperation} + w_5 \cdot R_{shortage}$$

### 5.1 Component Breakdown

**Grid Stability Reward (Global):**
$$R_{stability} = \sigma_t \cdot 10.0$$

Where stability $\sigma_t$ is computed as:
$$\sigma_t = 0.3 \cdot \text{mean\_score} + 0.5 \cdot \text{variance\_score} + 0.2 \cdot (1 - \text{critical\_penalty})$$

**Efficiency Reward:**
$$R_{efficiency} = -\frac{|e_i^t - 0.5 \cdot C_{max}|}{10}$$

Penalizes deviation from 50% battery capacity.

**Cost Reward:**
$$R_{cost} = \begin{cases}
-g^t \cdot P_{base} \cdot \lambda_{import} & \text{if } g^t > 0 \text{ (buying)} \\
|g^t| \cdot P_{base} \cdot 0.8 & \text{if } g^t < 0 \text{ (selling)}
\end{cases}$$

**Cooperation Reward:**
$$R_{cooperation} = 0.5 \cdot |\text{net\_transfer}_i^t|$$

**Shortage Penalty:**
$$R_{shortage} = \begin{cases}
2.0 \cdot (e_i^t + g_i^t - d_i^t) & \text{if balance} < 0 \\
0 & \text{otherwise}
\end{cases}$$

### 5.2 Reward Weights

| Component | Weight | Rationale |
|-----------|--------|-----------|
| Stability | 10.0 | Primary objective: grid reliability |
| Efficiency | -0.1 | Maintain optimal storage levels |
| Cost | -1.5 × price | Minimize expensive grid imports |
| Cooperation | 0.5 | Encourage energy sharing |
| Shortage | -2.0 | Heavy penalty for blackouts |

---

## 6. Grid Stability Metric

The stability metric aggregates system health:

$$\sigma_t = \alpha_1 \cdot f_{mean}(E_t) + \alpha_2 \cdot f_{var}(E_t) + \alpha_3 \cdot f_{critical}(E_t)$$

Where:

**Mean score** (energy should be ~50%):
$$f_{mean}(E_t) = 1 - 2 \cdot \left|\frac{\bar{E}_t}{C_{max}} - 0.5\right|$$

**Variance score** (lower variance is better):
$$f_{var}(E_t) = \frac{1}{1 + 10 \cdot \text{Var}(E_t) / C_{max}^2}$$

**Critical penalty** (agents below 20%):
$$f_{critical}(E_t) = 1 - 0.1 \cdot \sum_{i=1}^{n} \mathbb{1}[e_i^t < 0.2 \cdot C_{max}]$$

---

## 7. Learning Algorithm: PPO for Multi-Agent

We use **Proximal Policy Optimization (PPO)** adapted for multi-agent settings.

### 7.1 Policy Representation

Each agent $i$ has a policy $\pi_{\theta_i}$ parameterized by neural network weights $\theta_i$:

$$\pi_{\theta_i}: \mathcal{O}_i \rightarrow \Delta(\mathcal{A}_i)$$

### 7.2 PPO Objective

For each agent, we maximize the clipped surrogate objective:

$$L^{CLIP}(\theta) = \mathbb{E}_t \left[ \min\left( r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]$$

Where:
- $r_t(\theta) = \frac{\pi_\theta(a_t | o_t)}{\pi_{\theta_{old}}(a_t | o_t)}$ — Probability ratio
- $\hat{A}_t$ — Advantage estimate (GAE)
- $\epsilon = 0.2$ — Clipping parameter

### 7.3 Value Function

Each agent learns a value function $V_{\phi_i}(o_i)$ to estimate expected returns:

$$V_{\phi_i}(o_i^t) \approx \mathbb{E}\left[\sum_{k=0}^{\infty} \gamma^k r_i^{t+k} \mid o_i^t\right]$$

### 7.4 Advantage Estimation (GAE)

Generalized Advantage Estimation with $\lambda$:

$$\hat{A}_t^{GAE} = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_t^V$$

Where $\delta_t^V = r_t + \gamma V(o_{t+1}) - V(o_t)$ is the TD residual.

---

## 8. Environment Configuration

| Parameter | Symbol | Value | Units |
|-----------|--------|-------|-------|
| Number of agents | $n$ | 5 | - |
| Max battery capacity | $C_{max}$ | 100 | kWh |
| Max transfer rate | $T_{max}$ | 20 | kWh/step |
| Base electricity price | $P_{base}$ | 0.15 | $/kWh |
| Grid import penalty | $\lambda_{import}$ | 1.5 | - |
| Episode length | $T$ | 288 | steps |
| Timestep duration | - | 5 | minutes |
| Discount factor | $\gamma$ | 0.99 | - |
| GAE lambda | $\lambda$ | 0.95 | - |
| Learning rate | $\alpha$ | 3e-4 | - |

---

## 9. Convergence Criteria

Training is considered converged when:
1. Episode reward variance $< 10\%$ of mean over last 10 iterations
2. Policy entropy stabilizes
3. Value function loss plateaus

---

## References

1. Schulman, J., et al. "Proximal Policy Optimization Algorithms." arXiv:1707.06347 (2017).
2. Lowe, R., et al. "Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments." NeurIPS (2017).
3. Terry, J., et al. "PettingZoo: A Standard API for Multi-Agent Reinforcement Learning." NeurIPS (2021).

