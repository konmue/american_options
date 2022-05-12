---
marp: true
theme: default
pagination: true
---

# Deep Optimal Stopping & Pricing of American-Style Options
### Applied Quantitative Finance Seminar

Sabina Georgescu, Konrad Müller


---

# Overview

1. Introduction to Bermudan Options
2. Least-Squares Monte Carlo
3. Deep LSM
4. Deep Optimal Stopping
5. Dual LSM
6. Results

---

# 1. Introduction to Bermudan Options

---

# Context

* European Option: Exercisbale on a **single** date
* Bermudan Option: Exercisable on **several** dates
* American Option: Exercisable on **any** date before expiration

$\therefore\quad$ Bermudan contracts lie between European & American.

---

# Bermudan Max-Call Option

* Call Option on the maximum of $d$ underlying assets $S^{1,...,d}$
* Finitely many exercise dates $0 = t_0 < t_1 < ...< t_N = T$
* Discounted payoff at exercise time $t_n$:

$$ G_{t_n} =e^{-r t_n}\max_{i = 1, ..., d} \left(S_{t_n}^i - K\right)^+ $$


---
# Pricing & Dynamics

* Multi-dimensional Black-Scholes model under risk-neutral dynamics

* $d$-dimensional Brownian Motion $W$ with uncorrelated instantaneous components:

$$    dS^i_t = S_0^i \exp{\left(\left[r - \delta - \frac{\sigma^2}{2}\right]dt + \sigma dW^i_t\right)}$$

* Pricing formula with $\sup$ attained at $\tau_n$:

$$ V_{t_n} = \sup_{\tau\in\{t_n,...,t_N\}} \mathbb{E}[G_\tau \mid \mathcal{F}_{t_n} ] $$

---
# 2. Least-Squares Monte Carlo

**Longstaff \& Schwartz** (2001), *Valuing American Options by Simulation: A Simple Least-Squares Approach*

Estimate **continuation values** by regressing simulated discounted cash flows onto basis functions $L_0,...,L_B$ of ITM paths:

$$ F(\omega; t_n) \approx \sum_{i=0}^B w^*_i \cdot L_i(X(\omega; t_n)) \quad\text{for}\quad \textbf{w}^* = \min_{w\in\mathbb{R}^{B+1}} || \textbf{Y}_{t_n} - B(t_n)\cdot\textbf{w}||_{L_2}  $$

where the discounted cash flows vector and basis functions matrix are given by:
$$\textbf{Y}_{t_n} = \left[Y(\omega_0; t_n),...,Y(\omega_m; t_n)\right]^T$$
$$ \textbf{B}_m(t_n) = \left[L_0(X(\omega_m;t_n)),...,L_B(X(\omega_m;t_n))\right]^T, \forall m\in\{1,...,M\}$$

---
**Optimal Stopping Strategy** via **Snell Envelope**, work backwards in time:

$$
 V_{t_n} =
 \begin{dcases}
 G_T \quad\text{if } n=N\\
 \max \{G_{t_n}, F_{t_n}\} \quad\text{if } n<N
 \end{dcases}
$$

**Dynamic Programming Equation**:
 $$
 \tau_n =
 \begin{dcases}
 t_n \quad\text{if } G_{t_n} \ge F_{t_n}\\
 \tau_{n+1} \quad\text{if } G_{t_n} < F_{t_n}
 \end{dcases}
$$

Apply to each path $\omega_{1,...,M}$ to obtain Bermudan option valuation:
$$ V_{LSM} := \sum_{i=1}^{m}V_{t_0}(\omega)$$




---

# Recursive Formulation (Dynamic Programming Principle) / Continuation Values / ... SABINA


---
# Approximating the Conditional Expectation SABINA



* Cond expecatation minimizes MSE over all Borel measureble functions
* Cannot optimize over that function space -> instead need set of basis functions ...

---

# Choice of Basis Functions KONRAD

* What features to use as a regression basis? 
* For Bermudan Max-Call ($d = 5$):
    * Longstaff-Schwartz (2001):
        * first five Hermite polynomials in the value of the most expensive asset
        * the value and its square of the other 4 assets
        * selected products between the individual asset prices
    * Broadie, Cao (2008): 12 - 18 polynonmial basis functions up to $5^{th}$ order

---

# LSM Algorithm Pseudo-Code KONRAD

```Py

def lsm(paths):

    payoff_at_stop = payoff_fn(N, paths[:, -1])

    for n in np.arange(start=N, stop=0, step=-1):

        x_n = paths[:, n]
        payoff_now = payoff_fn(n, x_n)

        features = feature_map(x_n, payoff_now)
        model = LinearRegression()
        model.fit(features, payoff_at_stop)
        continuation_values = model.predict(features)

        idx = payoff_now >= continuation_values
        payoff_at_stop[idx] = payoff_now[idx]
    
    biased_price = payoff_at_stop.mean()

```

---
# Issues with Feature Engineering KONRAD

* Feature maps not expressive enough for precise pricing
* Potentially over-engineered to the specifics of the payoff / market dynamics
* Feature maps grow "only" linearly in $d$, but quickly enough to make them infeasible for large $d$ or $N$

**Solution**: Learn the feature maps $\rightarrow$ Deep LSM

---

# 3. Deep LSM
---

# Lower Bound, Upper Bound, Point Estimate ... SABINA

* I would say we only talk here about the upper bound and leave it out in the LSM part

---
# 4. Deep Optimal Stopping
---

# Motivation

* DLSM: Uses neural networks to approximate continuation values that are then compared to the immediate payoff of the option
* The immediate payoff is already a feature for the neural network

$\rightarrow$ Let the neural network make this comparison step inherently, i.e., parametrize the stopping decision directly (**Deep Optimal Stopping**)

---

# Parametrizing the stopping decision

$$    \tau_{n+1} = \sum_{m=n+1}^{N} \left[m \cdot f^{\theta_m}(X_m) \cdot \prod_{j=n+1}^{m-1}(1-f^{\theta_j}(X_j))\right]$$

---

# Parameter Optimization

* Reward (Loss) calculated using the **soft** stopping decision

$$    r^k_n(\theta) = g(n,x^k_n)\cdot F^\theta(x^k_n) + g(l^k_{n+1},x^k_{l^k_{n+1}})\cdot (1-F^\theta(x^k_n))$$

---
# 6. Results

---

# Results LSM

<center>

| Features | d  | $S_0$ | $P_{\text{biased}}$ | s.e. | L     | U     | Point Est. | 95 \% CI       |
|----------|----|-------|---------------------|------|-------|-------|------------|----------------|
| base     | 5  | 90    | 16.33               | 0.03 | 16.54 | 16.89 | 16.72      | [16.30, 16.93] |
| ls       | 5  | 90    | 16.56               | 0.04 | 16.41 | 16.57 | 16.49      | [16.17, 16.59] |
| r-NN     | 5  | 90    | 16.63               | 0.04 | 16.56 | 16.78 | 16.67      | [16.33, 16.81] |
| base     | 5  | 100   | 25.71               | 0.06 | 25.72 | 26.28 | 26.00      | [25.43, 26.32] |
| ls       | 5  | 100   | 26.01               | 0.02 | 26.03 | 26.21 | 26.12      | [25.75, 26.24] |
| r-NN     | 5  | 100   | 25.94               | 0.04 | 26.00 | 26.32 | 26.16      | [25.72, 26.35] |

</center>

---


# Results DLSM
<center>

| d  | $S_0$ | L     | U     | Point Est. | 95 \% CI       |
|----|-------|-------|-------|------------|----------------|
| 5  | 90    | 16.62 | 16.66 | 16.64      | [16.60, 16.66] |
| 5  | 100   | 26.10 | 26.18 | 26.14      | [26.09, 26.20] |
| 5  | 110   | 36.69 | 36.79 | 36.74      | [36.67, 36.81] |
| 10 | 90    | 26.05 | 26.33 | 26.19      | [26.03, 26.37] |
| 10 | 100   | 38.09 | 38.40 | 38.24      | [38.06, 38.43] |
| 10 | 110   | 50.60 | 50.99 | 50.80      | [50.57, 51.03] |

</center>

---
# Results DOS

<center>

| d  | $S_0$ | L     | U     | Point Est. | 95 \% CI       |
|----|-------|-------|-------|------------|----------------|
| 5  | 90    | 16.60 | 16.65 | 16.62      | [16.59, 16.66] |
| 5  | 100   | 26.11 | 26.18 | 26.15      | [26.10, 26.19] |
| 5  | 110   | 37.78 | 37.86 | 37.82      | [37.76, 37.88] |
| 10 | 90    | 26.14 | 26.30 | 26.22      | [26.13, 26.33] |
| 10 | 100   | 38.23 | 38.45 | 38.34      | [38.21, 38.49] |
| 10 | 110   | 50.95 | 51.17 | 51.06      | [50.93, 51.22] |

</center>

---

# Comparison to the Literature
<center>

| d  | $S_0$ | DOS   | DLSM  | Literature |
|----|-------|-------|-------|------------|
| 5  | 90    | 16.62 | 16.64 | 16.64      |
| 5  | 100   | 26.15 | 26.14 | 26.15      |
| 5  | 110   | 37.82 | 36.74 | 36.78      |
| 10 | 90    | 26.22 | 26.19 | 26.28      |
| 10 | 100   | 38.34 | 38.24 | 38.37      |
| 10 | 110   | 51.06 | 50.80 | 50.90      |

</center>

---

# References

#### Literature

* [Valuing American Options by Simulation:
A Simple Least-Squares Approach](https://people.math.ethz.ch/~hjfurrer/teaching/LongstaffSchwartzAmericanOptionsLeastSquareMonteCarlo.pdf)
* [Deep optimal stopping](https://arxiv.org/pdf/1804.05394.pdf)
* [ Pricing and Hedging American-Style Options with Deep Learning](https://www.mdpi.com/1911-8074/13/7/158/htm)
* [Optimal Stopping via Randomized Neural Networks](https://arxiv.org/abs/2104.13669)


#### Code

https://github.com/konmue/american_options


---

# Q&A