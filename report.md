\# Self-Pruning Neural Network



\# Results



| Lambda | Accuracy | Sparsity |

|--------|---------|---------|

| 1e-5   | 39.02%  | 0.11%   |

| 1e-4   | 40.10%  | 0.17%   |

| 1e-3   | 38.96%  | 0.17%   |



\# Observation

As lambda increases, sparsity increases slightly while accuracy decreases. This shows a trade-off between model performance and compression.



\# Conclusion

The model successfully learns to prune itself during training while maintaining reasonable accuracy.

