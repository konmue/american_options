---
marp: true
theme: default
pagination: true
math: katex
---

# CAN NEURAL NETWORKS SMILE? 
### EXPLORING IMPLIED VOLATILITY MOVEMENTS USING DEEP LEARNING
### *Applied Quantitative Finance Seminar*

Authors: Jonathan Baker and Feodor Doval

Discussed by: Sabina Georgescu, Konrad Müller

---
# Motivation: Minimum-Variance Hedging

* Delta-hedging options within the BS paradigm generally fails to minimise the hedging portfolio's variance, due to a non-zero correlation between the underlying's price and volatility movements

* **MV $\delta$**: Accounts for both changes to achieve min. variance in the hedging position:

$$\delta_{MV} := \argmin_{\delta} Var(\Delta f - \delta \Delta S)$$

* Hull & White (2017, appx. A) approximate it as: 
$$\delta_{MV} \approx \delta_{BS} + \nu_{BS} \frac{\partial E[\sigma_{imp}]}{\partial S}$$


$\therefore$ Finding $\delta_{MV}$ reduces to estimating the **expected change in IV**, $E[\Delta \sigma_{imp}].$

---

# Estimation: Expected Changes in IV
Proposed estimation models for $E[\Delta \sigma_{imp}]$ :
  * **Hull and White** (2017): analytical form with 3-parameters
  * **Cao et al.** (2018): **feed-forward NNs** with 3 (4) features: 
    $$\frac{\Delta S}{S}, \delta_{\mathrm{BS}}, \tau, (\text{VIX}).$$

Replication:

* Theoretical methodology comparison};
* Results align with the literature: NN models outperform the parsimonious HW};
* Model forecast comparison via simple \textbf{hedging performance} metric};
* Highlights bad prediction behaviour for extreme $\delta_{BS}$.


---
# IV surface approach

* Issue: Option data often sparse - does not include every ($K$, $\tau$) combination that might be of interest (here few ATM options; unusual)
* Idea:
    * Instead of estimating $\frac{\Delta S}{S}, \delta_{\mathrm{BS}}, \tau, \text{VIX} \mapsto \Delta IV$ for each traded ($K$, $\tau$) combination separately
        * First, estimate the IV surface using the SVI parametrization
        * Then, estimate 
        $$\frac{\Delta S}{S}, \frac{K}{S}, \tau, \text{VIX} \mapsto\Delta IV^{SVI}$$ 
        for any of the fitted gridpoints ($K$, $\tau$)
    

---
# Feedback

* We like your approach of rephrasing the problem and predicting the change in the entire implied volatility surface
    * The sparsity of the option data is such a fundamental problem that it makes sense to inherently incorporate it into problem formulation and modelling

---
# Lookahed bias from different options at same time

*"It is assumed that each observation is independent of the other observation"*


<center>

![width:20cm](table_1.png)

</center>

* You are modelling $\Delta IV$ as a **function** of $\frac{\Delta S}{S}, \delta_{\mathrm{BS}}, \tau, (\text{VIX})$
* Two of these features are **identical** for these observations $\rightarrow$ clearly not independent
* Random splitting of observations into train and test set leads to lookahead bias 

---

# Target Encoding & Lookahead bias

* Used two closest observations: $IV_t$, $IV_{t+n}$ for some $n \in \{1, 2, 3, 4\}$ 
* to calculate $n$ targets $\Delta IV_{t+n} := \frac{IV_{t+n} - IV_t}{n}$
    * These targets might be significantly different to the true (unobserved) values:
        * e.g., $IV_t = IV_{t+2}$ does not imply $IV_t = IV_{t+1}$
* Further, the corresponding features: $\delta_{BS}, \tau, \text{VIX}$ will be quite similar for these $n$ days
* These created observations are then randomly split into train or test sets
    * So in the worst case ($n = 4$), you might be training on $\{\Delta IV_t, \Delta IV_{t+4}\}$ and testing on $\{\Delta IV_{t+1}, \Delta IV_{t+3}\}$ $\rightarrow$ **Lookahead bias** for the same option

---

# Suggestion

* Follow Hull and White (2017) and instead of doing that target encoding just look for data points where there are prices for the same option available on successive days

* With respect to CV and lookahead bias: 
    * *At least*: split the data into train, validation, and test *periods* (no shuffling; avoiding any leakage)
    * *Better*: do this multiple times using rolling windows with or without a fixed origin
    * *Alternative*: use CV techniques that are not just *Walk-Forward* methods, but are very careful in avoiding biases e.g., by deleting overlapping periods (purging, embargo, ...); see e.g., De Prado (2018)

---

# Questions 


* Hull and White (2017) filter out options with $\delta_{BS} \leq 0.05$ and  $\delta_{BS} \geq 0.95$. Did you consider this? Might have avoided the reported "explosions".

* Usually: ITM options illiquid, as one might as well trade in the underlying. Do you have an idea why you mainly find deep OTM and deep ITM options?
