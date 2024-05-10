
Measures for ordinal classification

# 1. Ranked probability score

- Is a generalization of the Brier score to multiple categories.

- The *predicted cumulative distribution function* can be computed from class probabilities that are predicted by a model, that is the estimated probabilities of an observation belonging to classes ${k = 1, \cdots, K}$.

- The *true cumulative distribution function* simplifies to a step function with a step from 0 to 1 at the true value $Y_i$ for observation $i$.

- The RPS uses solely the ordering of the categories and does not require information on the distances between categories.

- A **smaller** RPS score represents a **better** prediction.

$$
RPS = \frac{1}{N} \sum_{i=1}^{N} \sum_{k=1}^{K}
\left[ p_k({x_n}) - I(Y_n \leq k) \right]^2
$$

- where $k=1,2,\cdots, K$ denotes the class labels

- $K$ denotes the number of response classes

- $p_k(x_i)$ denotes the predicted probability of sample $i$ belonging to class $k$

- $I(Y_i \leq k)$ is a indicator function indicating whether $Y_i$ is less than or equals to $k$

- $\sum_{k=1}^{K}
  \left| p_k({x_n}) - I(Y_n \leq k) \right|$ represent the area between the predicted and the true cumulative distribution functions


# 2. Concordance index (C-index)

- The C-index is computed as the proportion of concordant pairs among the comparable pairs.

- A higher C-index indicated a better performance.

- C-index varies between 0 and 1.

- When there are only two different class labels, the C-index reduces to the area under the Receiver Operating Characteristic (ROC) curve

$$
\text{C-index} 
$$


# 3. Spearman's rank correlation coefficient


# Reference

Epstein, E.S., 1969. A scoring system for probability forecasts of ranked categories. J. Appl. Meteorol. 8 (6), 985–987, doi: [10.1175/1520-0450(1969)008<0985:ASSFPF>2.0.CO;2](https://doi.org/10.1175/1520-0450(1969)008<0985:ASSFPF>2.0.CO;2)

Janitza, S., Tutz, G. and Boulesteix, A.L., 2016. Random forest for ordinal responses: prediction and variable selection. *Computational Statistics & Data Analysis*, 96, pp.57-73, doi: [10.1016/j.csda.2015.10.005](https://doi.org/10.1016/j.csda.2015.10.005)

Tang, M., Perez-Fernandez, R. and De Baets, B., 2021. A comparative study of machine learning methods for ordinal classification with absolute and relative information. Knowledge-Based Systems, 230, p.107358. doi: [10.1016/j.knosys.2021.107358](https://doi.org/10.1016/j.knosys.2021.107358)

Waegeman, W., De Baets, B. and Boullart, L., 2008. ROC analysis in ordinal regression learning. Pattern Recognition Letters, 29(1), pp.1-9, doi: [10.1016/j.patrec.2007.07.019](https://doi.org/10.1016/j.patrec.2007.07.019)