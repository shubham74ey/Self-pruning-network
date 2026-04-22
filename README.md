\# Self-Pruning Neural Network



This project implements a neural network that learns to prune itself during training using learnable gates.



\#Overview

In traditional pruning, weights are removed after training.

In this project, the model learns which weights to remove during training itself.



Each weight is associated with a gate (value between 0 and 1):

\- Gate ≈ 1 → weight active

\- Gate ≈ 0 → weight pruned



\#Key Idea

We add a sparsity penalty to the loss function:



Total Loss = Classification Loss + λ × Sparsity Loss



\- Sparsity Loss = sum of all gate values (L1 penalty)

\- Encourages many gates to become zero



\#Tech Used

\- Python

\- PyTorch

\- CIFAR-10 Dataset

\- Matplotlib



\#Results



| Lambda | Accuracy | Sparsity |

|--------|---------|---------|

| 1e-5   | 39.02%  | 0.11%   |

| 1e-4   | 40.10%  | 0.17%   |

| 1e-3   | 38.96%  | 0.17%   |



\#Observations

\- Increasing λ increases sparsity

\- Higher sparsity slightly reduces accuracy

\- Shows trade-off between efficiency and performance



\#Gate Distribution



!\[Graph](graph.png)





\#Conclusion

The model successfully learns to prune unnecessary weights during training, creating a sparse and efficient network.



