import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cvxpy as cp
import argparse
import itertools
import powerlaw

FONTSIZE = 20

def get_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', type=str, default='us_econ', choices=['us_econ', 'er', 'ba', 'msom_willems', 'wiot', 'sf'])
    parser.add_argument('--n', type=int, default=1)
    parser.add_argument('--eps', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('-y', type=float, default=1.0)

    parser.add_argument('--us_econ_year', type=int, default=2020)
    parser.add_argument('--us_econ_year_min', type=int, default=2000)
    parser.add_argument('--us_econ_year_max', type=int, default=2020)
    parser.add_argument('--us_econ_year_step', type=int, default=5)

    parser.add_argument('--msom_idx', type=int, default=10)
    parser.add_argument('--msom_idx_min', type=int, default=10)
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

    parser.add_argument('--wiot_countries', default='USA,JPN,GBR,CHN,IDN,IND', type=str)
    parser.add_argument('--wiot_country', default='CHN', type=str)

    parser.add_argument('--sf_K', type=int, default=100)

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
    elif args.name == 'wiot':
        return f'I-O Table for {args.wiot_country}'

def get_extra_suptitle(args):
    if args.name == 'us_econ':
        return f'US Economy'
    elif args.name == 'er':
        return f'ER graphs with $K = {args.er_K}$'
    elif args.name == 'ba':
        return f'BA graphs with $K = {args.ba_K}$'
    elif args.name == 'msom_willems':
        return f'Supply-chain networks from Willems (2008)'
    elif args.name == 'wiot':
        return f'World I-O Tables'
    elif args.name == 'sf':
        return f'Scale-free Graph'
    
def get_label(args, key):
    if args.name == 'us_econ':
        return f'Year: {key}'
    elif args.name == 'er':
        return f'$p = {key:.2f}$'
    elif args.name == 'ba':
        return f'$m = {key}$'
    elif args.name == 'msom_willems':
        return f'Network #{key}'
    elif args.name == 'wiot':
        return f'Country: {key}'
    else:
        return key

def draw(G, pos, measures, measure_name, ticks=None, labels=None):
    node_size = np.array([v for v in measures.values()])
    node_size = node_size / node_size.max() * 100 * 2

    nodes = nx.draw_networkx_nodes(G, pos, cmap=plt.cm.plasma, 
                                   node_color=list(measures.values()),
                                   nodelist=measures.keys(),
                                   node_size=node_size)
    nodes.set_norm(mcolors.SymLogNorm(linthresh=0.01, linscale=1, base=10))


    edges = nx.draw_networkx_edges(G, pos, alpha=0.1, width=0.5)

    plt.title(measure_name, fontsize=FONTSIZE)

    if ticks is not None:
        cbar = plt.colorbar(nodes, ticks=ticks)
        cbar.ax.set_yticklabels(labels, fontsize=9, rotation=270)
    else:
        cbar = plt.colorbar(nodes)
        # cbar.ax.set_yticklabels([' '])


    # cbar.set_label('Katz Centrality in $G^R$', rotation=270, labelpad=1)
    # cbar.ax.set_title('More Raw', fontsize=FONTSIZE)
    # cbar.ax.set_xlabel('Complex', fontsize=10)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'visualization_{args.name}.pdf')

def estimate_resilience(A, n, eps, T, intervention_idx=[]):
    K = A.shape[0]

    x = np.linspace(0, 1, 100)
    lo = 0
    hi = x.shape[0] - 1

    lb = 1 - 1 / K

    while lo < hi:
        mid = (lo + hi) // 2
        p_est, _ = estimate(A, n, eps, T, x[mid], intervention_idx, mode='survival_probability')
        p_std = np.sqrt(p_est * (1 - p_est) / T)
        print(f'x = {x[mid]:.3f}, Pr[S ≥ {1 - eps:.3f} * K] = {p_est:.3f} ± {p_std:.3f}')
        if p_est < lb: 
            hi = mid - 1
        else:
            lo = mid + 1

    return x[mid]

def estimate(A, n, eps, T, x, intervention_idx, mode='survival_probability', y=1):
    correct = np.zeros(T)
    K = A.shape[0]

    for t in range(T):
        Y = (np.random.uniform(size=(K, K)) <= y).astype(np.float64)
        AY = A * Y 
        U = np.random.uniform(size=(K))
        W = (U <= 1 - x**n).astype(np.int64)
        if len(intervention_idx) > 0:
            W[intervention_idx] = 1

        Z = W
        for _ in range(100):
            Z_old = Z
            for i in range(K):
                Z[i] = np.prod(Z[AY[i, :].nonzero()[0]]) * W[i]
            
            if np.all(np.isclose(Z, Z_old)):
                break

        S = Z.sum()

        if S >= (1 - eps) * K and mode == 'survival_probability':
            correct[t] = 1
        elif mode == 'survivals':
            correct[t] = S
        elif mode == 'failures':
            correct[t] = K - S

    return correct.mean(), correct.std()

def number_of_failures_lp(A, n, x, y, intervention_idx):
    K = A.shape[0]
    ones = np.ones((K, 1))
   
    u = (x**n) * ones
    u[intervention_idx] = 0

    beta = cp.Variable((K, 1))
    objective = cp.Maximize(cp.sum(beta))

    constraints = [beta >= 0, beta <= ones, beta <= y * (A.T @ beta) + u]

    prob = cp.Problem(objective, constraints)
    result = prob.solve()

    return result

def degree_distribution(A, out=True):

    if out:
        degrees = A.sum(1)
    else:
        degrees = A.sum(0)

    degrees += 1

    values, counts = np.unique(degrees, return_counts=True)
    counts = counts.astype(np.float64)
    values = values.astype(np.float64)
    counts /= counts.sum()

    return degrees, values, counts

def powerlaw_fit(A, out=True):
    if out:
        degrees = A.sum(1)
    else:
        degrees = A.sum(0)

    r

def fit_degree_distribution(A, args, out=True):

    # plt.figure(figsize=(10, 10))
    # plt.title(f"{'Outdegree' if out else 'Indegree'} Distribution for US {get_extra_suptitle(args)}", fontsize=FONTSIZE)
    # plt.xlabel('Degree (log)', fontsize=FONTSIZE)
    # plt.ylabel('Frequency (log)', fontsize=FONTSIZE)
    # plt.xscale('log')
    # plt.yscale('log')

    for key in A.keys():
        degrees, values, counts = degree_distribution(A[key], out=out)
        results = powerlaw.Fit(degrees, xmin=1.0)
        print(f'{get_label(args, key)}: alpha = {results.power_law.alpha}')

    # plt.legend(fontsize=0.75*FONTSIZE)
    # plt.savefig(f"{'outdegree' if out else 'indegree'}_{args.name}.pdf")

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
    depths = {}

    id_range = np.arange(args.msom_idx_min, args.msom_idx_max + 1, args.msom_idx_step)

    plt.figure(figsize=(10, 10))
    plt.title('Tier Distribution for Supply-networks from Willems (2008)', fontsize=FONTSIZE)
    plt.ylabel('Number of Nodes', fontsize=FONTSIZE)
    plt.xlabel('Tier', fontsize=FONTSIZE)

    max_val = -1
    for i, idx in enumerate(id_range):
        df = pd.read_excel('msom-willems.xls', sheet_name=f"{'0' if idx < 10 else ''}{idx}_LL")
        G = nx.from_pandas_edgelist(df, 'sourceStage', 'destinationStage', create_using=nx.DiGraph)
        A[idx] = nx.to_numpy_array(G).astype(np.float64)
        y[idx] = 1 / (1e-5 + A[idx].sum(0).max())

        df_stat = pd.read_excel('msom-willems.xls', sheet_name=f"{'0' if idx < 10 else ''}{idx}_SD")
        
        values, counts = np.unique(df_stat['relDepth'], return_counts=True)
        values = 1 + values.astype(np.int64)
        max_val = max(max_val, values.max())

        df_rd = df_stat[['Stage Name', 'relDepth']]
        df_rd.set_index('Stage Name', inplace=True)

        depths[idx] = df_rd.to_dict()['relDepth']
        
        plt.bar(values + 0.2 * i, counts, 0.2, label=f'{get_label(args, idx)}, Num Tiers: {values.max() + 1}, Num Edges: {int(A[idx].sum())}')

    plt.xticks(np.arange(1, 1 + max_val), np.arange(1, 1 + max_val))

    plt.xlim(1, max_val)
    plt.legend(fontsize=0.75 * FONTSIZE)
    plt.savefig('statistics.pdf')


    return A, y, labels, depths

def load_random(args):

    A = {}
    y = {}
    labels = {} 

    if args.name == 'er':
        rng = np.linspace(args.er_p_min, args.er_p_max, args.er_p_linspace)
    elif args.name == 'ba':
        rng = np.arange(args.ba_m_min, args.ba_m_max + 1, args.ba_m_step)
    elif args.name == 'sf':
        rng = [(0.41, 0.54, 0.05, 0.2, 0)]

    for r in rng:
        if args.name == 'er':
            G = nx.erdos_renyi_graph(args.er_K, r, seed=args.seed, directed=True)
        elif args.name == 'ba':
            G = nx.barabasi_albert_graph(args.ba_K, r, seed=args.seed)
        elif args.name == 'sf':
            alpha, beta, gamma, delta_in, delta_out = r
            G = nx.scale_free_graph(args.sf_K, alpha=alpha, beta=beta, gamma=gamma, delta_in=delta_in, delta_out=delta_out)

        A[r] = nx.to_numpy_array(G)
        y[r] = 1 / (1e-5 + A[r].sum(0).max())
        labels[r] = []

    return A, y, labels

def load_wiot(args):

    df = pd.read_excel('wiot.xlsb', sheet_name='2014', skiprows=2, nrows=2410)

    countries = args.wiot_countries.split(',')

    indices = {}

    for country in countries:
        indices[country] = (+float('inf'), -float('inf'))

    for i in range(df.values[:, 2].shape[0]):
        if df.values[i, 2] in countries:
            start, end = indices[df.values[i, 2]]
            start = min(start, i)
            end = max(end, i)
            indices[df.values[i, 2]] = (start, end)
    
    A = {}
    y = {}
    labels = {}

    for country in countries:
        start, end = indices[country]
        values = df.values[start:end+1, start+1:end+2].astype(np.float64)
        A[country] = (values > 0).astype(np.float64)
        idx = np.arange(A[country].shape[0])
        A[country][idx, idx] = 0
        y[country] = 1 / (1e-5 + A[country].sum(0).max())
        labels[country] = df.values[start:end+1, 1]
    # import pdb; pdb.set_trace()

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
    elif args.name == 'wiot':
        key = args.wiot_country
    elif args.name == 'sf':
        key = (0.41, 0.54, 0.05, 0.2, 0)
    else:
        key = ''

    return key

def resilience_lb_vs_key(args, A, y, labels):
    plt.figure(figsize=(10, 10))
    plt.title(f'Lower bound on $R_G(\\varepsilon)$ for {get_extra_suptitle(args)}', fontsize=FONTSIZE)
    plt.xlabel('Intervention Budget $T$', fontsize=FONTSIZE)
    plt.ylabel('Resilience Lower Bound', fontsize=FONTSIZE)

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
    plt.legend(fontsize=0.75*FONTSIZE)
    plt.savefig(f'resilience_lb_vs_key_{args.name}.pdf')

def resilience_lb_vs_y(args, A, y, labels):
    key = get_key(args)
    
    K = A[key].shape[0]
    y_range = np.linspace(1e-5, y[key], 10)

    T_range = 1 + np.arange(K)
    I = np.eye(K, dtype=np.float64)

    plt.figure(figsize=(10, 10))
    plt.title(f'Lower bound on $R_G(\\varepsilon)$ for {get_extra_suptitle(args)}', fontsize=FONTSIZE)
    plt.xlabel('Intervention Budget $T$', fontsize=FONTSIZE)
    plt.ylabel('Resilience Lower Bound', fontsize=FONTSIZE)

    for yy in y_range:
        beta_katz_inverse = np.linalg.inv(I - yy * A[key]).sum(-1)
        beta_katz_inverse_ordered_cumsum = np.cumsum(np.sort(beta_katz_inverse))[::-1]
        resilience_lb = (args.eps / beta_katz_inverse_ordered_cumsum)**(1/args.n)
        plt.plot(T_range, resilience_lb, label=f'y = {yy:.4f}')

    plt.yscale('log')
    plt.xscale('log')
    plt.legend(fontsize=0.75*FONTSIZE)
    plt.savefig(f'resilience_lb_vs_y_{args.name}.pdf')

def visualize(args, A, y, labels, num_ticks=2, depths=None):
    key = get_key(args)

    K = A[key].shape[0]
    # Plot graph and visualize Katz centralities
    G = nx.from_numpy_array(A[key], create_using=nx.DiGraph)

    if args.name == 'wiot':
        pos = nx.spring_layout(G, seed=args.seed, k=10 / np.sqrt(K))
    else:
        pos = nx.spring_layout(G, seed=args.seed)
    
    I = np.eye(K)

    beta_katz_inverse = np.linalg.inv(I - y[key] * A[key]).sum(-1)
    
    # import pdb; pdb.set_trace()

    if args.name == 'us_econ':
        ordering = np.argsort(-beta_katz_inverse)
        ticks_linspace = np.linspace(0, len(ordering) - 1, num_ticks).astype(np.int64)
        ticks = beta_katz_inverse[ordering][ticks_linspace]
        ticks_labels = labels[key][ordering][ticks_linspace].tolist()
    else:
        ticks = None
        ticks_labels = None
       
    beta_katz_inverse_dict = dict([(i, x) for i, x in enumerate(beta_katz_inverse)])
    plt.figure(figsize=(10, 10))
    draw(G, pos, beta_katz_inverse_dict, f'Visualization for {get_extra_title(args)}', ticks=ticks, labels=ticks_labels)
    plt.savefig(f'visualization_{args.name}.pdf')

def resilience_monte_carlo_vs_eps(args, A):

    plt.figure(figsize=(10, 10))
    plt.title(f'{get_extra_suptitle(args)}', fontsize=FONTSIZE)
    plt.ylabel('$\\hat R_G(\\varepsilon)$ (MC Estimate)', fontsize=FONTSIZE)
    plt.xlabel('$\\epsilon$', fontsize=FONTSIZE)

    eps_range = np.linspace(0, 1, 20)

    for key in A.keys():
        R_mc = np.zeros_like(eps_range)
        area_under_curve = 0

        K = A[key].shape[0]
        for i, eps in enumerate(eps_range):
            print('eps = ', eps)
            R_mc[i] = estimate_resilience(A[key], n=args.n, eps=eps, T=1000)
            
            if i >= 1:
                area_under_curve += R_mc[i] * (eps_range[i] - eps_range[i - 1])
        
        print(f'AUC: {area_under_curve}')
        plt.plot(eps_range, R_mc, label=f'{get_label(args, key)} (AUC: {area_under_curve:.3f})')
    
    plt.legend(fontsize=0.75*FONTSIZE)
    plt.savefig(f'resilience_monte_carlo_vs_eps_{args.name}.pdf')

def expected_number_of_failures_vs_lp(args, A, y):
    plt.figure(figsize=(10, 10))
    plt.title(f'Number of Failures {get_extra_suptitle(args)}', fontsize=FONTSIZE)
    plt.ylabel('Number of Failures')
    plt.xlabel('$x$', fontsize=FONTSIZE)

    x_range = np.linspace(0, 1, 20)
    palette = itertools.cycle(sns.color_palette())

    for key in A.keys():
        color = next(palette)

        F_mc_mean = np.zeros_like(x_range)
        F_mc_std = np.zeros_like(x_range)
        F_lp = np.zeros_like(x_range)

        for i, x in enumerate(x_range):
            F_lp[i] = number_of_failures_lp(A[key], args.n, x=x, y=y[key], intervention_idx=[])
            F_mc_mean[i], F_mc_std[i] = estimate(A[key], args.n, T=1000, x=x, mode='failures', y=y[key], eps=0, intervention_idx=[])    
            
            print(f'x = {x}, LP = {F_lp[i]}, MC = {F_mc_mean[i]}')

        plt.plot(x_range, F_mc_mean, label=f'{get_label(args, key)} (MC)', color=color)
        plt.fill_between(x_range, F_mc_mean - F_mc_std, F_mc_mean + F_mc_std, color=color, alpha=0.2)
        plt.plot(x_range, F_lp, label=f'{get_label(args, key)} (LP)', color=color, linestyle='dotted')

    plt.legend(fontsize=0.75*FONTSIZE)
    plt.savefig(f'failures_vs_lp_{args.name}.pdf')
    

def resilience_monte_carlo_vs_intervention(args, A):
    
    plt.figure(figsize=(10, 10))
    plt.title(f'$\\hat R_G(\\varepsilon)$ (Monte-Carlo Estimate) vs. Interventions for {get_extra_suptitle(args)}', fontsize=FONTSIZE)
    plt.xlabel('Intervention Budget $T$', fontsize=FONTSIZE)
    plt.ylabel('$\\hat R_G(\\varepsilon)$', fontsize=FONTSIZE)
    
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

    plt.legend(fontsize=0.75*FONTSIZE)
    plt.savefig(f'resilience_monte_carlo_vs_intervention_{args.name}.pdf')

def print_statistics(args, A):
    print('DATASET STATISTICS')

    for key in A.keys():
        print(get_label(args, key))
        print(f'K {A[key].shape[0]}')
        print(f'Min/Max In-degree {A[key].sum(0).min()} / {A[key].sum(0).max()}')
        print(f'Min/Max Our-degree {A[key].sum(1).min()} / {A[key].sum(1).max()}')
        print(f'Average degree {A[key].sum(0).mean()}')
        print(f'outdegree distribution')
        print(f'indegree distribution')
        K = A[key].shape[0]

        print(f'Density: {A[key].sum() / (K**2 - K)} ')

        print()

    print('Outdegree powerlaw fit')
    fit_degree_distribution(A, args, out=True)
    print('Indegree powerlaw fit')
    fit_degree_distribution(A, args, out=False)

    exit()
    

if __name__ == '__main__':
    sns.set_theme()
    args = get_argparser()

    # Load/generate data
    if args.name == 'us_econ':
        A, y, labels = load_us_economy(args)
        depths = None
    elif args.name in ['er', 'ba', 'sf']:
        A, y, labels = load_random(args)
        depths = None
    elif args.name == 'msom_willems':
        A, y, labels, depths = load_msom_willems(args)
    elif args.name == 'wiot':
        A, y, labels = load_wiot(args)
        depths = None
    print_statistics(args, A)

    exit()

    visualize(args, A, y, labels, depths=depths)

    expected_number_of_failures_vs_lp(args, A, y)
    resilience_lb_vs_key(args, A, y, labels)
    # resilience_lb_vs_y(args, A, y, labels)
    resilience_monte_carlo_vs_eps(args, A)
    # resilience_monte_carlo_vs_intervention(args, A)







