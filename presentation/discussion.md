---
marp: true
theme: default
pagination: true
---

# CAN NEURAL NETWORKS SMILE? 
### EXPLORING IMPLIED VOLATILITY MOVEMENTS USING DEEP LEARNING
### *Applied Quantitative Finance Seminar*

Authors: Jonathan Baker and Feodor Doval

Discussed by: Sabina Georgescu, Konrad Müller

---
# Minimum Variance Hedging

* Hedging an option based on the Black-Scholes delta does not generally minimize the variance of the hedged portfolio

* The hedging position in the underlying that achieves minimum variance is called the **minimum variance** (MV) delta

$$\delta_{MV} := \argmin_{\delta} Var(\Delta f - \delta \Delta S)$$

* It can be approximated (see Hull and White 2017, Appendix A) as 
$$\delta_{MV} \approx \delta_{BS} + \nu_{BS} \frac{\partial E[\sigma_{imp}(S, K)]}{\partial S}$$


* This reduces the problem of finding $\delta_{MV}$ to estimating $E[\Delta \sigma_{imp}]$

---

# Estimating the expected implied volatility change

* To estimate $E[\Delta \sigma_{imp}]$
    * Hull and White (2017) propose an analytical form with 3-parameters
    * Cao et al. (2018) propose using **feed-forward neural networks** with 3 (4) features: 
    $$\frac{\Delta S}{S}, \delta_{\mathrm{BS}}, T, (V)$$
* The forecasts of the different models are compared relatively with each other based on their impact on the **hedging performance**

---
# Probably not relevant enough for the presentation

* The forecasts from a model $A$ is evaluated on its impact on the **hedging performance**, via relative comparison with a benchmark model $B$
$$\text { Gain }=1-\frac{S S E[\text { Model } A ]}{S S E[\text { Model } B] }$$

&emsp; &emsp; where SSE denotes the sum of squared hedging errors

---

# IV Surface approach

[Contribution part]

---
# Feedback

---
# Critisism

* independence assumption / cross validation

---

# Questions & Suggestions
