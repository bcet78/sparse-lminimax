# -*- coding: utf-8 -*-
"""
Created on Sun May 25 21:01:52 2025

@author: baybo
"""


import numpy as np
import gurobipy as gp
from gurobipy import GRB
import time
import cvxpy as cp
import pandas as pd
import os

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

    linear_term = g_val + gp.quicksum(gradg[j] * (x[j] - xk[j]) for j in range(n))

    model.setObjective(t - linear_term, GRB.MINIMIZE)
    model.optimize()

    vars = model.getVars()
    return np.array([v.X for v in vars[:n]])

def augment_nonzero_indices(I: np.ndarray, nonzero_indices: np.ndarray, m: int) -> np.ndarray:

    mask = ~np.isin(I, nonzero_indices)
    additional_indices = I[mask][:m]
    combined = np.concatenate([nonzero_indices, additional_indices])
    return np.unique(combined)  


# %% MILP

def solve_sparse_minimax(A, k, time_limit):
    
    start = time.time()
    n = A.shape[1]

    I = np.arange(n)
    
    LPrelax = Subproblem_Solver(A,I)
    
    Initial_indices = np.argsort(-LPrelax[0])[:k]
    
    soln = Subproblem_Solver(A,Initial_indices)
    
    new_x = np.zeros(n)
    new_x[Initial_indices] = soln[0]
    
    end = time.time()
    
    "---------------------------------"
    m, n = A.shape
    model = gp.Model()

    model.setParam('OutputFlag', 0) 
    #model.setParam('LogToConsole', 0) 
    model.setParam("TimeLimit", time_limit-(end-start))
    
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
        print(f"Model status: {model.status}")
        return None, None

os.chdir('C:\\Users\\baybo\\Desktop\\Final Codes\\DC Data')

MATRICES = []

#%% Read Prostate Data

df = pd.read_excel('prostata.xlsx')

medians = df.median()

# Apply the rule: 1 if value > median, else -1
df_binary = df.apply(lambda col: np.where(col > medians[col.name], 1, -1))
df_binary = pd.DataFrame(df_binary, columns=df.columns)

labels = df_binary[['class']]

df_binary = df_binary.drop(columns=['class'])

Z = df_binary.mul(labels['class'], axis=0)
Z = Z.to_numpy()
Z = -1 * Z

MATRICES.append(Z)

#%% Read Arryhtmia

df = pd.read_csv('arrhythmia.csv')

df["binaryClass"] = df["binaryClass"].replace({"N": -1, "P": 1})

def is_number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)

# Keep only columns where all values are int or float
df_clean = df.loc[:, df.applymap(is_number).all()]

medians = df_clean.median()

# Apply the rule: 1 if value > median, else -1
df_binary = df_clean.apply(lambda col: np.where(col > medians[col.name], 1, -1))
df_binary = pd.DataFrame(df_binary, columns=df_clean.columns)

labels = df_binary[['binaryClass']]

df_binary = df_binary.drop(columns=['binaryClass'])
df_binary = df_binary.drop(columns=['id'])

Z = df_binary.mul(labels['binaryClass'], axis=0)
Z = Z.to_numpy()
Z = -1 * Z
MATRICES.append(Z)

#%% Read Kidney

df = pd.read_csv('Kidney.csv')

df["Tissue"] = df["Tissue"].replace({"Colon": -1, "Kidney": 1})

medians = df.median()

# Apply the rule: 1 if value > median, else -1
df_binary = df.apply(lambda col: np.where(col > medians[col.name], 1, -1))
df_binary = pd.DataFrame(df_binary, columns=df.columns)

labels = df_binary[['Tissue']]

df_binary = df_binary.drop(columns=['Tissue'])
df_binary = df_binary.drop(columns=['ID_REF'])
df_binary = df_binary.drop(columns=['id'])

Z = df_binary.mul(labels['Tissue'], axis=0)
Z = Z.to_numpy()
Z = -1 * Z
MATRICES.append(Z)


#%% Read OVA Colon

df = pd.read_csv('OVA_Colon.csv')

df["Tissue"] = df["Tissue"].replace({"Other": -1, "Colon": 1})

medians = df.median()

# Apply the rule: 1 if value > median, else -1
df_binary = df.apply(lambda col: np.where(col > medians[col.name], 1, -1))
df_binary = pd.DataFrame(df_binary, columns=df.columns)

labels = df_binary[['Tissue']]

df_binary = df_binary.drop(columns=['Tissue'])
df_binary = df_binary.drop(columns=['ID_REF'])
df_binary = df_binary.drop(columns=['id'])

Z = df_binary.mul(labels['Tissue'], axis=0)
Z = Z.to_numpy()
Z = -1 * Z

MATRICES.append(Z)


#%% Read Reuters



df = pd.read_csv('reuters.csv')

df["'label1'"] = df["'label1'"].replace({False: -1, True: 1})
df["'label2'"] = df["'label2'"].replace({False: -1, True: 1})
df["'label3'"] = df["'label3'"].replace({False: -1, True: 1})
df["'label4'"] = df["'label4'"].replace({False: -1, True: 1})
df["'label5'"] = df["'label5'"].replace({False: -1, True: 1})
df["'label6'"] = df["'label6'"].replace({False: -1, True: 1})
df["'label7'"] = df["'label7'"].replace({False: -1, True: 1})

medians = df.median()

# Apply the rule: 1 if value > median, else -1
df_binary = df.apply(lambda col: np.where(col > medians[col.name], 1, -1))
df_binary = pd.DataFrame(df_binary, columns=df.columns)

labels = df_binary["'label1'"]

df_binary = df_binary.drop(columns=["'label1'"])
df_binary = df_binary.drop(columns=["'label2'"])
df_binary = df_binary.drop(columns=["'label3'"])
df_binary = df_binary.drop(columns=["'label4'"])
df_binary = df_binary.drop(columns=["'label5'"])
df_binary = df_binary.drop(columns=["'label6'"])
df_binary = df_binary.drop(columns=["'label7'"])

Z = df_binary.mul(labels, axis=0)
Z = Z.to_numpy()
Z = -1 * Z

MATRICES.append(Z)


#%% Read Scene

df = pd.read_csv('scene.csv')
medians = df.median()

# Binary transformation: 1 if value > median, else -1
df_binary = (df > medians).astype(int) * 2 - 1  # Converts True to 1, False to -1

# Separate label and features
labels = df_binary[['Urban']]
df_binary = df_binary.drop(columns=['Urban', 'id'])

# Elementwise multiplication
Z = df_binary.mul(labels.squeeze(), axis=0).to_numpy()
Z = -1 * Z

MATRICES.append(Z)

#%% Read Bioresponse

df = pd.read_csv('bioresponse.csv')
medians = df.median()

# Binary transformation: 1 if value > median, else -1
df_binary = (df > medians).astype(int) * 2 - 1  # Converts True to 1, False to -1
labels = df_binary[['target']]
df_binary = df_binary.drop(columns=['target', 'id'])

Z = df_binary.mul(labels.squeeze(), axis=0).to_numpy()
Z = -1 * Z

MATRICES.append(Z)

#%% Internet Advertisements

df = pd.read_csv('Internet-Advertisements.csv')


medians = df.median()

# Binary transformation: 1 if value > median, else -1
df_binary = (df > medians).astype(int) * 2 - 1  # Converts True to 1, False to -1
labels = df_binary[["'class'"]]
df_binary = df_binary.drop(columns=["'class'", 'id'])

Z = df_binary.mul(labels.squeeze(), axis=0).to_numpy()
Z = -1 * Z

MATRICES.append(Z)


#%% madelon
df = pd.read_csv('madelon.csv')
medians = df.median()

# Binary transformation: 1 if value > median, else -1
df_binary = (df > medians).astype(int) * 2 - 1  # Converts True to 1, False to -1

labels = df_binary[['Class']]
df_binary = df_binary.drop(columns=['Class', 'id'])

Z = df_binary.mul(labels.squeeze(), axis=0).to_numpy()
Z = -1 * Z

MATRICES.append(Z)


#%% Random Z


Z = np.random.choice([-1, 1], size=(1000, 1000))

MATRICES.append(Z)





#%% Main Code

alpha = 0.5
beta = 0.5
k = 40
tolerence = 10**(-5)

emptylist = []

#%%


for Z in MATRICES:
    start = time.time()
    m = Z.shape[0]
    n = Z.shape[1]
    
    lambmax = 5
    lambmin = 0
    termination_sparsity = n
    iters = 0
    
    x_0 = np.ones(n)/n
     
    
    support = np.nonzero(x_0)[0]
    x_LP = DCA_Step(Z[:, support], x_0, alpha, beta, 0.1)
    x_1 = np.zeros(n)
    x_1[support] = x_LP
      
    #x_1 = np.copy(LPrelax)
    print(zero_norm(x_1))
    
    ht_indices = np.argsort(-x_LP[0])[:k]
    
    relsoln = Subproblem_Solver(Z,ht_indices)
    
    Relaxation_Result = relsoln[1]
    
    x_1 = np.round(x_1,5)
    
    
    while abs(termination_sparsity-k) >= int(0.1*k) and iters < 20:
        
        terminated_down = False
        terminated_up = False
        
        iters += 1
        print(lambmin,lambmax)
        
        lamb = (lambmax+lambmin)/2
        
        x_old = np.copy(x_0)
        x_new = np.copy(x_1)
        
        OPTIMUM = False
        
        while OPTIMUM == False:
            print("-----")
        
            if np.linalg.norm(x_new - x_old) < tolerence:
                x_small_new = np.copy(x_new)
                x_new = np.copy(x_old)
                terminated_up = True
                OPTIMUM = True
        
            if zero_norm(x_new) < k:
        
                x_small_new = np.copy(x_new)
                x_new = np.copy(x_old)
                terminated_down = True
                OPTIMUM = True
        
            else:
                support = np.nonzero(x_new)[0]
                x_old = np.copy(x_new)
                x_i1 = x_new[support]

                x_DCA = DCA_Step(Z[:, support], x_i1, alpha, beta, lamb)
                
                x_new = np.zeros(n)
                x_new[support] = x_DCA
            
            
        
        termination_sparsity = zero_norm(x_small_new)
        indices = np.argsort(-x_new)[:k]
        print(termination_sparsity)
        
        nonzero_small = np.array([i for i, val in enumerate(x_small_new) if val != 0])
        
        
        
        if zero_norm(x_small_new) < k:
            extended_set =  augment_nonzero_indices(indices, nonzero_small, k-zero_norm(x_small_new))
            
            extended_soln = Subproblem_Solver(Z, extended_set)
            
            extended_x = np.zeros(n)
            extended_x[extended_set] = extended_soln[0]
            
            val = f(extended_x, Z)
        
            
        else:
            val = 999999
            
        
        soln = Subproblem_Solver(Z, indices)
        thresholded_x = np.zeros(n)
        thresholded_x[indices] = soln[0]
        
        
        DC_Result = min(val,f(thresholded_x, Z))
        print(DC_Result)
        if terminated_up:
            lambmin = lamb
            
        if terminated_down:
            lambmax = lamb
    
    end = time.time()
    Runtime = end-start
    
    MILP_Soln = solve_sparse_minimax(Z, k, 1*(end-start))
    MILP_Result = MILP_Soln[-1]
    
    emptylist.append((m,n,k,round(DC_Result,5),round(Relaxation_Result,5), round(MILP_Result,5), Runtime,lamb))

#%%

#selected_column_names = df_binary.columns[extended_set]


import pandas as pd

df = pd.DataFrame(np.array(emptylist),columns=["m","n","k","DC","Prel","MILP","Runtime","Beta" ])


df.to_excel("Beta Selection MIDCA Results.xlsx")