# Metrics

- Entropy H(X): Monte Carlo estimate via model evaluation.
- Mutual Information I(X;Y): MI between subsets using shared API.
- Conditional Mutual Information I(X;Y|Z): extend API by conditioning indices.
- Edge MI: pairwise MI for edges in first tree as proxy for direct interactions.
- Change points: CUSUM on time series of MI/entropy.
- Edge persistence: average Jaccard similarity across consecutive adjacency estimates.
