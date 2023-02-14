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

def draw(G, pos, measures, measure_name):
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
    plt.colorbar(nodes)
    plt.axis('off')
    plt.savefig('visualization.pdf')


sns.set_theme()

args = get_argparser()

# Plot lower bound on resilience vs intervention budget for a year range for the same y 
year_range = np.arange(args.year_min, args.year_max + 1)

n = 1

A = {}
y = {}

eps = 0.01

for year in year_range:
    df = pd.read_excel('import_matrices.xlsx', sheet_name=str(year), skiprows=5)
    values = df.values[2:72, 2:72]
    values[np.where(values == '...')] = 0
    values = values.astype(np.int64)
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
beta_katz_inverse_dict = dict([(i, x) for i, x in enumerate(beta_katz_inverse)])
plt.figure()
draw(G, pos, beta_katz_inverse_dict, f'Visualization of Katz Centralities for the U.S. Economy in {year}')



plt.show()



