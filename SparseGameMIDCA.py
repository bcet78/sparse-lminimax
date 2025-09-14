# -*- coding: utf-8 -*-
"""
Created on Fri May 23 09:08:13 2025

@author: baybo
"""


import numpy as np
import gurobipy as gp
from gurobipy import GRB
import time
import cvxpy as cp
import pandas as pd


def f(x, A):

    return np.max(A @ x)


def g(x, alpha, beta, lamb):

    return -lamb*np.sum((x**alpha) * ((1 - x)**beta))


def nabla_g(x, alpha, beta, lamb):
    x = np.clip(x, 1e-8, 1 - 1e-8)
    term1 = alpha * (1 - x)**beta * x**(alpha - 1)
    term2 = beta * (1 - x)**(beta - 1) * x**alpha
    return -lamb*(term1 - term2)


def zero_norm(x):
    return np.count_nonzero(x)


def Subproblem_Solver(A, C):

    A_restricted = A[:, C]

    x = cp.Variable(len(C))
    objective = cp.Minimize(cp.max(A_restricted @ x))
    constraints = [x >= 0, cp.sum(x) == 1]
    problem = cp.Problem(objective, constraints)
    problem.solve()

    return x.value, problem.value


def DCA_Step(A, xk, alpha, beta, lamb):

    n = A.shape[1]
    m = A.shape[0]

    gradg = nabla_g(xk, alpha, beta, lamb)

    g_val = g(xk, alpha, beta, lamb)

    model = gp.Model()

    model.setParam('OutputFlag', 0)

    x = model.addVars(n, lb=0, ub=1, name="x")
    t = model.addVar(name="t", lb=-GRB.INFINITY)

    for i in range(m):
        model.addConstr(gp.quicksum(A[i, j] * x[j] for j in range(n)) <= t)

    model.addConstr(gp.quicksum(x[j] for j in range(n)) == 1)

    linear_term = g_val + \
        gp.quicksum(gradg[j] * (x[j] - xk[j]) for j in range(n))

    model.setObjective(t - linear_term, GRB.MINIMIZE)
    model.optimize()

    vars = model.getVars()
    return np.array([v.X for v in vars[:n]])

# %% MILP

def solve_sparse_minimax(A, k, time_limit):
    
    n = A.shape[1]

    I = np.arange(n)
    
    LPrelax = Subproblem_Solver(A,I)
    
    Initial_indices = np.argsort(-LPrelax[0])[:k]
    
    soln = Subproblem_Solver(A,Initial_indices)
    
    new_x = np.zeros(n)
    new_x[Initial_indices] = soln[0]
    
    "---------------------------------"
    m, n = A.shape
    model = gp.Model()

    model.setParam('OutputFlag', 0) 
    model.setParam('LogToConsole', 0) 
    model.setParam("TimeLimit", time_limit)
    
    x = model.addVars(n, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS)
    y = model.addVars(n, vtype=GRB.BINARY)
    t = model.addVar(lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS)

    model.addConstr(gp.quicksum(x[j] for j in range(n)) == 1)
    model.addConstr(gp.quicksum(y[j] for j in range(n)) <= k)

    for j in range(n):
        model.addConstr(x[j] <= y[j])

    for i in range(m):
        model.addConstr(gp.quicksum(A[i, j] * x[j] for j in range(n)) <= t)
        

    model.setObjective(t, GRB.MINIMIZE)
    
    "----------"
    for i in range(n):
        x[i].Start = new_x[i]
    "----------"
    
    model.optimize()

    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        x_sol = np.array([x[j].X for j in range(n)])
        return x_sol, t.X
    else:
        return None, None


#%%
def generate_uniform_mnk_triplets(F):
    triplets = []
    m_values = list(range(60, 700, 100))  
    n_values = list(range(12, 150, 20))   

    for m in m_values:
        for n in n_values:
            if m <= 2*n:
                continue  

            k_min = max(1, int(n / 10))
            k_max = max(k_min + 1, int(n / 3))
            if k_max >= n:
                continue


            k = (k_min + k_max) // 2

            triplets.append((m, n, k))

            if len(triplets) == F:
                return triplets

    return triplets  


F = 40
triplets = generate_uniform_mnk_triplets(F)

#%%

def augment_nonzero_indices(I: np.ndarray, nonzero_indices: np.ndarray, m: int) -> np.ndarray:

    mask = ~np.isin(I, nonzero_indices)
    additional_indices = I[mask][:m]
    combined = np.concatenate([nonzero_indices, additional_indices])
    return np.unique(combined)  





#%%


np.random.seed(78)


alpha = 0.5
beta = 0.5
lamb = 10
tolerence = 10**(-5)
Alb = -100
Aub = 100
num_matrices = 20


summary_stats = []

for i,triplet in enumerate(triplets):
    
    print("Experiment",i)

    m = triplet[0]
    n = triplet[1]
    k = triplet[2]
    
    print(f"{m}-{n}-{k}")
    
    matrices = [np.random.randint(Alb, Aub + 1, size=(m, n)) for _ in range(num_matrices)]
    
    results = []
    
    for i,A in enumerate(matrices):
        
        start = time.time()
        record = {"matrix_index": i}
        
        x_new = np.ones(n)/n
        
        
        OPTIMUM = False
        
        while OPTIMUM == False:
            
            support = np.nonzero(x_new)[0]
            x_old = np.copy(x_new)
            x_i1 = x_new[support]

            x_DCA = DCA_Step(A[:, support], x_i1, alpha, beta, lamb)
            
            x_new = np.zeros(n)
            x_new[support] = x_DCA

        
            if np.linalg.norm(x_new - x_old) < tolerence:
                x_small_new = np.copy(x_new)
                x_new = np.copy(x_old)
        
                OPTIMUM = True
        
            if zero_norm(x_new) < k:
        
                x_small_new = np.copy(x_new)
                x_new = np.copy(x_old)
                OPTIMUM = True
        
            else:
                continue
        
        
        indices = np.argsort(-x_new)[:k]
        
        
        nonzero_small = np.array([i for i, val in enumerate(x_small_new) if val != 0])
        

    
        if zero_norm(x_small_new) < k:
            extended_set =  augment_nonzero_indices(indices, nonzero_small, k-zero_norm(x_small_new))
            
            extended_soln = Subproblem_Solver(A, extended_set)
            
            extended_x = np.zeros(n)
            extended_x[extended_set] = extended_soln[0]
            
            val = f(extended_x, A)
        
        else:
            val = 999999
        
        soln = Subproblem_Solver(A, indices)
        thresholded_x = np.zeros(n)
        thresholded_x[indices] = soln[0]
        
    
        end = time.time()
        
        MILP_Soln = solve_sparse_minimax(A, k, end-start)
        
        record["Runtime"] = end-start
        record["DC Result"] = min(val,f(thresholded_x, A))

        record["MILP Result"] = MILP_Soln[-1]
        results.append(record)
        
    df = pd.DataFrame(results)
    percentage = (df["DC Result"] < df["MILP Result"]).mean() * 100
    
    summary_stats.append({
        "m": m,
        "n": n,
        "k": k,
        "Samples": num_matrices,
        "Avg Runtime": df["Runtime"].mean(),
        "Avg Rel Impr": round(((-df['DC Result'] + df['MILP Result']) / df['MILP Result']).mean()*100,3),
        "DC_better_percent": percentage
    })
    
    import os
    from pathlib import Path

    script_dir = Path(os.path.dirname(os.path.abspath(__file__)))

    output_filename = script_dir / f"{m}-{n}-{k}.xlsx"

    df.to_excel(output_filename, index=False)
    
    print(f"Results saved to {output_filename}")
    npz_filename = script_dir / f"{m}-{n}-{k}.npz"
    np.savez(npz_filename, *matrices)

#%%
summary_df = pd.DataFrame(summary_stats)
summary_df.to_excel(script_dir / "Summary of Experiments.xlsx", index=False)




