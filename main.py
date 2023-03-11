import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import argparse



def get_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--year', type=int, default=1997)
    parser.add_argument('--year_min', type=int, default=1997)
    parser.add_argument('--year_max', type=int, default=2021)

    return parser.parse_args()

def draw(G, pos, measures, measure_name, ticks, labels):
    node_size = np.array([v for v in measures.values()])
    node_size = node_size / node_size.max() * 100


    nodes = nx.draw_networkx_nodes(G, pos, cmap=plt.cm.plasma, 
                                   node_color=list(measures.values()),
                                   nodelist=measures.keys(),
                                   node_size=node_size)
    nodes.set_norm(mcolors.SymLogNorm(linthresh=0.01, linscale=1, base=10))
    # labels = nx.draw_networkx_labels(G, pos)
    edges = nx.draw_networkx_edges(G, pos, alpha=0.1, width=0.5)

    plt.title(measure_name)

    cbar = plt.colorbar(nodes, ticks=ticks)
    cbar.ax.set_yticklabels(labels)
    cbar.set_label('Katz Centrality in $G^R$', rotation=270)
    cbar.ax.set_title('Raw Material', fontsize=10)
    cbar.ax.set_xlabel('Complex Product', fontsize=10)
    plt.axis('off')
    plt.savefig('visualization.pdf')

def estimate_resilience(A, n, eps, T, intervention_idx=np.array([])):
    K = A.shape[0]

    x = np.linspace(0, 1, 100)
    lo = 0
    hi = x.shape[0] - 1

    lb = 1 - 1 / K

    while lo < hi:
        mid = (lo + hi) // 2
        p_est = estimate_probability(A, n, eps, T, x[mid], intervention_idx)
        p_std = np.sqrt(p_est * (1 - p_est) / T)
        print(f'x = {x[mid]:.3f}, Pr[S ≥ {1 - eps:.3f} * K] = {p_est:.3f} ± {p_std:.3f}')
        if p_est < lb: 
            hi = mid - 1
        else:
            lo = mid + 1

    return x[mid]

def estimate_probability(A, n, eps, T, x, intervention_idx):
    correct = 0
    K = A.shape[0]

    for t in range(T):
        U = np.random.uniform(size=(K))
        W = (U <= 1 - x**n).astype(np.int64)
        W[intervention_idx] = 1

        Z = W
        for _ in range(100):
            Z_old = Z
            for i in range(K):
                Z[i] = np.prod(Z[A[i, :].nonzero()[0]]) * W[i]
            
            if np.all(np.isclose(Z, Z_old)):
                break

        S = Z.sum()

        if S >= (1 - eps) * K:
            correct += 1

    return correct / T


sns.set_theme()

args = get_argparser()

# Plot lower bound on resilience vs intervention budget for a year range for the same y 
year_range = np.arange(args.year_min, args.year_max + 1)

n = 1

A = {}
y = {}

eps = 0.01

labels = {}

for year in year_range:
    df = pd.read_excel('import_matrices.xlsx', sheet_name=str(year), skiprows=5)
    values = df.values[2:67, 2:67]
    values[np.where(values == '...')] = 0
    values = values.astype(np.int64)
    labels[year] = df.values[0, 2:67]
    A[year] = (values > 0).astype(np.float64)
    y[year] = 1 / (1e-5 + A[year].sum(0).max())
    K = A[year].shape[0]

plt.figure(figsize=(10, 10))
plt.title('Lower bound on $R_G(\\varepsilon)$')
plt.xlabel('Intervention Budget $T$')
plt.ylabel('Resilience Lower Bound')

T_range = 1 + np.arange(K)
I = np.eye(K, dtype=np.float64)

for year in year_range:
    beta_katz_inverse = np.linalg.inv(I - y[year] * A[year]).sum(-1)
    beta_katz_inverse_ordered_cumsum = np.cumsum(np.sort(beta_katz_inverse))[::-1]
    resilience_lb = (eps / beta_katz_inverse_ordered_cumsum)**(1/n)
    plt.plot(T_range, resilience_lb, label=year)


plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.savefig('resilience_lb_vs_year.pdf')



# Plot lower bound on resilience vs intervention budget for a year for a range of y 
year = args.year

y = 1 / (1e-5 + A[year].sum(0).max())
y_range = np.linspace(1e-5, y, 10)

plt.figure(figsize=(10, 10))
plt.title('Lower bound on $R_G(\\varepsilon)$')
plt.xlabel('Intervention Budget $T$')
plt.ylabel('Resilience Lower Bound')

for yy in y_range:
    beta_katz_inverse = np.linalg.inv(I - yy * A[year]).sum(-1)
    beta_katz_inverse_ordered_cumsum = np.cumsum(np.sort(beta_katz_inverse))[::-1]
    resilience_lb = (eps / beta_katz_inverse_ordered_cumsum)**(1/n)
    plt.plot(T_range, resilience_lb, label=f'y = {yy:.4f}')

plt.yscale('log')
plt.xscale('log')
plt.legend()
plt.savefig('resilience_lb_vs_y.pdf')

# Plot graph and visualize Katz centralities
G = nx.from_numpy_array(A[year], create_using=nx.DiGraph)
pos = nx.spring_layout(G, seed=675)

y = 1 / (1e-5 + A[year].sum(0).max())

beta_katz_inverse = np.linalg.inv(I - y * A[year]).sum(-1)
ordering = np.argsort(-beta_katz_inverse)
num_ticks = 3
ticks_linspace = np.linspace(0, len(ordering) - 1, num_ticks).astype(np.int64)
ticks = beta_katz_inverse[ordering][ticks_linspace]
ticks_labels = labels[year][ordering][ticks_linspace].tolist()

beta_katz_inverse_dict = dict([(i, x) for i, x in enumerate(beta_katz_inverse)])
plt.figure()
draw(G, pos, beta_katz_inverse_dict, f'U.S. Economy in {year}', ticks, ticks_labels)
plt.savefig('visualization.pdf')

# plt.figure()
# plt.title('Resilience (Monte-Carlo Estimate)')
# plt.ylabel('Resilience')
# plt.xlabel('$\\epsilon$')

# for year in [2000, 2010, 2020]:
#     eps_range = np.linspace(0, 1, 20)
#     R_mc = np.zeros_like(eps_range)

#     for i, eps in enumerate(eps_range):
#         print('eps = ', eps)
#         R_mc[i] = estimate_resilience(A[year], n=1, eps=eps, T=1000)
#         print('\n')

#     plt.plot(eps_range, R_mc, label=f'{year}')

# plt.legend()
# plt.savefig('resilience_monte_carlo.pdf')

plt.figure()
plt.title('Resilience (Monte-Carlo Estimate) vs. Interventions')
plt.xlabel('Intervention Budget $T$')
plt.ylabel('Resilience')
# plt.xscale('log')
# plt.yscale('log')

year = 1997
ordering = np.argsort(-beta_katz_inverse)
R_mc_intervention = np.zeros(len(ordering), dtype=np.float64)
eps = 0.8

for i in range(len(ordering)):
    print(f'T = {i + 1}')
    R_mc_intervention[i] = estimate_resilience(A[year], n, eps=eps, T=1000, intervention_idx=ordering[:i+1])
    print('\n')

plt.plot(T_range, R_mc_intervention)

plt.savefig('resilience_intervention.pdf')
