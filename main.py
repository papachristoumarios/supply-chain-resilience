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

    parser.add_argument('--name', type=str, default='us_econ', choices=['us_econ', 'er', 'ba', 'msom_willems'])
    parser.add_argument('--n', type=int, default=1)
    parser.add_argument('--eps', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--us_econ_year', type=int, default=2020)
    parser.add_argument('--us_econ_year_min', type=int, default=2000)
    parser.add_argument('--us_econ_year_max', type=int, default=2020)
    parser.add_argument('--us_econ_year_step', type=int, default=5)

    parser.add_argument('--msom_idx', type=int, default=10)
    parser.add_argument('--msom_idx_min', type=int, default=1)
    parser.add_argument('--msom_idx_max', type=int, default=38)
    parser.add_argument('--msom_idx_step', type=int, default=10)

    parser.add_argument('--er_p', type=float, default=0.5)
    parser.add_argument('--er_K', type=int, default=100)
    parser.add_argument('--er_p_min', type=float, default=0.1)
    parser.add_argument('--er_p_max', type=float, default=1)
    parser.add_argument('--er_p_linspace', type=int, default=10)

    parser.add_argument('--ba_K', type=int, default=100)
    parser.add_argument('--ba_m', type=int, default=1)
    parser.add_argument('--ba_m_min', type=int, default=1)
    parser.add_argument('--ba_m_max', type=int, default=5)
    parser.add_argument('--ba_m_step', type=int, default=1)

    return parser.parse_args()

def get_extra_title(args):
    if args.name == 'us_econ':
        return f'US Economy in {args.us_econ_year}'
    elif args.name == 'er':
        return f'ER graph with $K = {args.er_K}, p = {args.er_p}$'
    elif args.name == 'ba':
        return f'BA graph with $K = {args.ba_K}, m = {args.ba_m}$'
    elif args.name == 'msom_willems':
        return f'Supply-chain network {args.msom_idx} from Willems (2008)'

def get_extra_suptitle(args):
    if args.name == 'us_econ':
        return f'US Economy'
    elif args.name == 'er':
        return f'ER graphs with $K = {args.er_K}$'
    elif args.name == 'ba':
        return f'BA graphs with $K = {args.ba_K}$'
    elif args.name == 'msom_willems':
        return f'Supply-chain networks from Willems (2008)'
    
def get_label(args, key):
    if args.name == 'us_econ':
        return f'Year: {key}'
    elif args.name == 'er':
        return f'$p = {key:.2f}$'
    elif args.name == 'ba':
        return f'$m = {key}$'
    elif args.name == 'msom_willems':
        return f'Network #{key}'

def draw(G, pos, measures, measure_name, ticks=None, labels=None):
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

    if ticks is not None:
        cbar = plt.colorbar(nodes, ticks=ticks)
        cbar.ax.set_yticklabels(labels)
    else:
        cbar = plt.colorbar(nodes)

    cbar.set_label('Katz Centrality in $G^R$', rotation=270)
    cbar.ax.set_title('Raw Material', fontsize=10)
    cbar.ax.set_xlabel('Complex Product', fontsize=10)
    plt.axis('off')
    plt.savefig(f'visualization_{args.name}.pdf')

def estimate_resilience(A, n, eps, T, intervention_idx=[]):
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
        if len(intervention_idx) > 0:
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

def load_us_economy(args):
    A = {}
    y = {}
    labels = {}
    year_range = np.arange(args.us_econ_year_min, args.us_econ_year_max + 1, args.us_econ_year_step)

    for year in year_range:
        df = pd.read_excel('import_matrices.xlsx', sheet_name=str(year), skiprows=5)
        values = df.values[2:67, 2:67]
        values[np.where(values == '...')] = 0
        values = values.astype(np.int64)
        labels[year] = df.values[0, 2:67]
        A[year] = (values > 0).astype(np.float64)
        y[year] = 1 / (1e-5 + A[year].sum(0).max())

    return A, y, labels

def load_msom_willems(args):

    A = {}
    y = {}
    labels = {}

    id_range = np.arange(args.msom_idx_min, args.msom_idx_max + 1, args.msom_idx_step)

    for idx in id_range:
        df = pd.read_excel('msom-willems.xls', sheet_name=f"{'0' if idx < 10 else ''}{idx}_LL")
        G = nx.from_pandas_edgelist(df, 'sourceStage', 'destinationStage', create_using=nx.DiGraph)
        A[idx] = nx.to_numpy_array(G).astype(np.float64)
        y[idx] = 1 / (1e-5 + A[idx].sum(0).max())

    return A, y, labels

def load_random(args):

    A = {}
    y = {}
    labels = {} 

    if args.name == 'er':
        rng = np.linspace(args.er_p_min, args.er_p_max, args.er_p_linspace)
    elif args.name == 'ba':
        rng = np.arange(args.ba_m_min, args.ba_m_max + 1, args.ba_m_step)

    for r in rng:
        if args.name == 'er':
            G = nx.erdos_renyi_graph(args.er_K, r, seed=args.seed, directed=True)
        elif args.name == 'ba':
            G = nx.barabasi_albert_graph(args.ba_K, r, seed=args.seed)

        A[r] = nx.to_numpy_array(G)
        y[r] = 1 / (1e-5 + A[r].sum(0).max())
        labels[r] = []

    return A, y, labels

def get_key(args):
    if args.name == 'us_econ':
        key = args.us_econ_year
    elif args.name == 'msom_willems':
        key = args.msom_idx
    elif args.name == 'er':
        key = args.er_p
    elif args.name == 'ba':
        key = args.ba_m
    else:
        key = ''

    return key

def resilience_lb_vs_key(args, A, y, labels):
    plt.figure(figsize=(10, 10))
    plt.title(f'Lower bound on $R_G(\\varepsilon)$ for {get_extra_suptitle(args)}')
    plt.xlabel('Intervention Budget $T$')
    plt.ylabel('Resilience Lower Bound')

    for key in sorted(A.keys()):
        K = A[key].shape[0]
        T_range = 1 + np.arange(K)
        I = np.eye(K, dtype=np.float64)
        beta_katz_inverse = np.linalg.inv(I - y[key] * A[key]).sum(-1)
        beta_katz_inverse_ordered_cumsum = np.cumsum(np.sort(beta_katz_inverse))[::-1]
        resilience_lb = (args.eps / beta_katz_inverse_ordered_cumsum)**(1/args.n)
        if key == '':
            plt.plot(T_range, resilience_lb)
        else:
            plt.plot(T_range, resilience_lb, label=get_label(args, key))

    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.savefig(f'resilience_lb_vs_key_{args.name}.pdf')

def resilience_lb_vs_y(args, A, y, labels):
    key = get_key(args)
    
    K = A[key].shape[0]
    y_range = np.linspace(1e-5, y[key], 10)

    T_range = 1 + np.arange(K)
    I = np.eye(K, dtype=np.float64)

    plt.figure(figsize=(10, 10))
    plt.title(f'Lower bound on $R_G(\\varepsilon)$ for {get_extra_suptitle(args)}')
    plt.xlabel('Intervention Budget $T$')
    plt.ylabel('Resilience Lower Bound')

    for yy in y_range:
        beta_katz_inverse = np.linalg.inv(I - yy * A[key]).sum(-1)
        beta_katz_inverse_ordered_cumsum = np.cumsum(np.sort(beta_katz_inverse))[::-1]
        resilience_lb = (args.eps / beta_katz_inverse_ordered_cumsum)**(1/args.n)
        plt.plot(T_range, resilience_lb, label=f'y = {yy:.4f}')

    plt.yscale('log')
    plt.xscale('log')
    plt.legend()
    plt.savefig(f'resilience_lb_vs_y_{args.name}.pdf')

def visualize(args, A, y, labels, num_ticks=2):
    key = get_key(args)

    K = A[key].shape[0]
    # Plot graph and visualize Katz centralities
    G = nx.from_numpy_array(A[key], create_using=nx.DiGraph)
    pos = nx.spring_layout(G, seed=args.seed)

    I = np.eye(K)

    beta_katz_inverse = np.linalg.inv(I - y[key] * A[key]).sum(-1)
    
    if args.name == 'us_econ':
        ordering = np.argsort(-beta_katz_inverse)
        ticks_linspace = np.linspace(0, len(ordering) - 1, num_ticks).astype(np.int64)
        ticks = beta_katz_inverse[ordering][ticks_linspace]
        ticks_labels = labels[key][ordering][ticks_linspace].tolist()
    else:
        ticks = None
        ticks_labels = None
       
    beta_katz_inverse_dict = dict([(i, x) for i, x in enumerate(beta_katz_inverse)])
    plt.figure()
    draw(G, pos, beta_katz_inverse_dict, f'Visualization for {get_extra_suptitle(args)}', ticks, ticks_labels)
    plt.savefig(f'visualization_{args.name}.pdf')

def resilience_monte_carlo_vs_eps(args, A):

    plt.figure(figsize=(10, 10))
    plt.title(f'$\\hat R_G(\\varepsilon)$ (Monte-Carlo Estimate) for {get_extra_suptitle(args)}')
    plt.ylabel('$\\hat R_G(\\varepsilon)$')
    plt.xlabel('$\\epsilon$')

    eps_range = np.linspace(0, 1, 20)
    R_mc = np.zeros_like(eps_range)

    for key in A.keys():
        K = A[key].shape[0]
        for i, eps in enumerate(eps_range):
            print('eps = ', eps)
            R_mc[i] = estimate_resilience(A[key], n=args.n, eps=eps, T=1000)
            print('\n')

        plt.plot(eps_range, R_mc, label=get_label(args, key))
    
    plt.legend()
    plt.savefig(f'resilience_monte_carlo_vs_eps_{args.name}.pdf')

def resilience_monte_carlo_vs_intervention(args, A):
    
    plt.figure(figsize=(10, 10))
    plt.title(f'$\\hat R_G(\\varepsilon)$ (Monte-Carlo Estimate) vs. Interventions for {get_extra_suptitle(args)}')
    plt.xlabel('Intervention Budget $T$')
    plt.ylabel('$\\hat R_G(\\varepsilon)$')
    
    for key in A.keys():
        K = A[key].shape[0] 
        T_range = 1 + np.arange(K)
        I = np.eye(K)

        beta_katz_inverse = np.linalg.inv(I - y[key] * A[key]).sum(-1)

        ordering = np.argsort(-beta_katz_inverse)
        R_mc_intervention = np.zeros(len(ordering), dtype=np.float64)

        for i in range(len(ordering)):
            print(f'key = {key}, T = {i + 1}')
            R_mc_intervention[i] = estimate_resilience(A[key], args.n, eps=args.eps, T=1000, intervention_idx=ordering[:i+1])
            print()

        plt.plot(T_range, R_mc_intervention, label=get_label(args, key))

    plt.legend()
    plt.savefig(f'resilience_monte_carlo_vs_intervention_{args.name}.pdf')

if __name__ == '__main__':
    sns.set_theme()
    args = get_argparser()

    # Load/generate data
    if args.name == 'us_econ':
        A, y, labels = load_us_economy(args)
    elif args.name in ['er', 'ba']:
        A, y, labels = load_random(args)
    elif args.name == 'msom_willems':
        A, y, labels = load_msom_willems(args)

    resilience_lb_vs_key(args, A, y, labels)
    # resilience_lb_vs_y(args, A, y, labels)
    visualize(args, A, y, labels)
    resilience_monte_carlo_vs_eps(args, A)
    resilience_monte_carlo_vs_intervention(args, A)







