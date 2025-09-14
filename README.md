# sparse-lminimax

Here you can find the codes for our algorithm MIDCA to find candidate solutions to k-sparse strategies in bimatrix games and k-sparse minimum margin maximizing classifier problems.


## SparseGameMIDCA.py

This is the code we used in the paper for game theory experiments. One part of the code generates random matrices with desired sizes. User sets the following parameters:

alpha (Corrseponds to $\alpha$ in the paper): Scale parameter of CPFP  
beta  (Corrseponds to $\gamma$ in the paper): Scale parameter of CPFP  
lamb (Corrseponds to $\beta$ in the paper): Concave regularization term coefficient  
k: Sparsity limit, should be less than $n$  

## BoostingMIDCA.py

This takes 7 datasets mentioned in the paper as well as a random 1000x1000 instance. You can find the datasets in the references, and they are available online.  
The code and the algorithm are essentially the same, but here we do not assume prior knowledge about $\beta$, and implement the search procedure discussed in the paper.


## Replication

To replicate our results, simply set numpy seeds to 78. Note that as gurobipy version changes, the comparison results might be different due to the change in gurobipy performance. However, MIDCA yields the same result independent of the optimizer. So you might want to test different optimization software if you believe they can perform better than gurobipy. In our experiments, gurobipy was the best one for optimization at that time.
