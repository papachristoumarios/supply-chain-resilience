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
from sklearn.linear_model import LinearRegression, HuberRegressor
import statsmodels.api as sm
from stargazer.stargazer import Stargazer
import os
import collections
import bindata

FONTSIZE = 20
FIGSIZE = 5

MEDIUM_SIZE = FONTSIZE
SMALL_SIZE = 0.75 * MEDIUM_SIZE
BIGGER_SIZE = 1.5 * MEDIUM_SIZE

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

def get_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', type=str, default='us_econ', choices=['us_econ', 'er', 'ba', 'msom_willems', 'wiot'])
    parser.add_argument('--num_suppliers', type=str, choices=['constant', 'power_law', 'exponential'], default='constant')
    parser.add_argument('--n', type=str, default='1')
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

def get_n(arg, A):

    if arg.isnumeric():
        result = int(arg) * np.ones(A.shape[0])
    elif arg == 'deg':
        result = A.sum(1) + 1
    elif arg.startswith('PL'):
        alpha = float(arg.split(':')[-1])
        result = np.random.zipf(alpha, A.shape[0])
    elif arg.startswith('U'):
        a, b = map(int, arg.split(':')[1:])
        result = np.random.randint(a, b + 1, A.shape[0])
    else:
        raise ValueError(f'Unknown n: {arg}')
    return result.astype(int)


def calculate_binomial_coefficients(n):
    coefficients = np.zeros((n+1, n+1))
    for i in range(n+1):
        for j in range(i+1):
            if j == 0 or j == i:
                coefficients[i][j] = 1
            else:
                coefficients[i][j] = coefficients[i-1][j-1] + coefficients[i-1][j]
    return coefficients

def calculate_failure_probability_bahadur(x, n, K, rho, mode='correlated_suppliers'):
    z = 0
    q = x**n
    if mode == 'correlated_suppliers':
        C = calculate_binomial_coefficients(n)
        z = q
        for j in range(2, n + 1):
            z += C[n, j] * rho * (1 - x)**(j / 2) * x**(n - j / 2)
    elif mode == 'correlated_products':
        C = calculate_binomial_coefficients(K)
        for b in range(0, K):
            temp = 1
            for j in range(2, K + 1):
                for r in range(0, min(j, b + 1) + 1):
                    temp += C[j, r] * C[K - j, j - r] * rho * (1 - q)**(j / 2) * q**(j / 2 - r)

            z += temp * C[K - 1, b] * q**(b + 1) * (1 - q)**(K - b - 1)

    elif mode == 'correlated_all':
        N = n * K

        C = calculate_binomial_coefficients(N)

        for b in range(0, N - n + 1):
            temp = 1
            for j in range(2, N + 1):
                for r in range(0, min(j, b + n) + 1):
                    temp += C[j, r] * C[N - j, j - r] * rho * (1 - x)**(j / 2) * x**(j / 2 - r)

            z += temp * C[N - n, b] * x**(b + n) * (1 - x)**(N - n - b)

    return z 

def brute_force_ols(endog, exog, df):

    best_adjusted_R2 = -1
    best_model = None
    best_subset = None

    for i in range(1, len(exog)):
        for subset in itertools.combinations(exog, i):
            model = sm.OLS(df[endog], sm.add_constant(df[list(subset)])).fit()
        
            if model.rsquared_adj > best_adjusted_R2:
                best_adjusted_R2 = model.rsquared_adj
                best_model = model
                best_subset = subset

    return best_model, best_subset

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
    elif args.name == 'rdag':
        return f'Random DAG with p = {args.er_p}'

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
    elif args.name == 'rdag':
        return f'Random DAG'
    
def get_label(args, key):
    if args.name == 'us_econ':
        return f'Year: {key}'
    elif args.name in ['er', 'rdag']:
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

    plt.title(measure_name)

    if ticks is not None:
        cbar = plt.colorbar(nodes, ticks=ticks)
        cbar.ax.set_yticklabels(labels, fontsize=9, rotation=270)
    else:
        cbar = plt.colorbar(nodes)

    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'visualization_{args.name}.pdf')

def estimate_resilience(A, n, eps, T, intervention_idx=[], y=1, x_lo=0, x_hi=1, rho=0, mode='correlated_suppliers'):
    K = A.shape[0]

    lo = 0
    x = np.linspace(x_lo, 1, 100)
    hi = np.searchsorted(x, x_hi)

    lb = 1 - 1 / K

    while lo < hi:
        mid = (lo + hi) // 2
        p_est, _ = estimate(A, n, eps, T, x[mid], intervention_idx, mode='survival_probability', y=y, rho=rho, correlation_mode=mode)
        p_std = np.sqrt(p_est * (1 - p_est) / T)
        print(f'x = {x[mid]:.3f}, Pr[S ≥ {1 - eps:.3f} * K] = {p_est:.3f} ± {p_std:.3f}')
        if p_est < lb: 
            hi = mid - 1
        else:
            lo = mid + 1

    return x[mid]

def estimate_resilience_auc(A, n, T, intervention_idx=[], y=1, num_linspace=10, rho=0, eps_min=0, eps_max=1, x_lo=0, x_hi=1):

    eps_range = np.linspace(eps_min, eps_max, num_linspace)[::-1]

    auc = 0

    for i in range(1, len(eps_range)):
        R_mc = estimate_resilience(A, n, eps_range[i], T, intervention_idx, y=y, x_lo=x_lo, x_hi=x_hi, rho=rho)
        auc += R_mc * (eps_range[i-1] - eps_range[i])
        x_hi = R_mc

        if R_mc == 0:
            break

    return auc / (eps_max - eps_min)

def sample_survivals(x, K, n_arr, rho, mode='correlated_all'):

    if rho == 0:
        U = np.random.uniform(size=(K))
        W = (U <= 1 - x**n_arr).astype(np.int64)
    elif mode == 'correlated_suppliers':
        W = np.zeros(K)
        for i, n in enumerate(n_arr):
            I = np.eye(n)
            margprob = x * np.ones(n)
            bincorr = rho * (1 - I) + I
            commonprob = bindata.bincorr2commonprob(bincorr=bincorr, margprob=margprob)
            W[i] = 1 - bindata.rmvbin(N=1,commonprob=commonprob, margprob=margprob).prod()
    elif mode == 'correlated_products':
        I = np.eye(K)
        margprob = x**n_arr * np.ones(K)
        bincorr = rho * (1 - I) + I
        commonprob = bindata.bincorr2commonprob(bincorr=bincorr, margprob=margprob)
        W = bindata.rmvbin(N=1,commonprob=commonprob)
    elif mode == 'correlated_all':
        N = np.sum(n_arr)
        I = np.eye(N)
        margprob = x * np.ones(N)
        bincorr = rho * (1 - I) + I
        commonprob = bindata.bincorr2commonprob(bincorr=bincorr, margprob=margprob)
        U = bindata.rmvbin(N=1,commonprob=commonprob)
        W = np.zeros(K)
        j = 0

        for i in range(K):
            W[i] = U[j:j + n_arr[i]].prod()
            j += n_arr[i]

    return W

def estimate(A, n, eps, T, x, intervention_idx, mode='survival_probability', y=1, rho=0, correlation_mode='correlated_suppliers'):
    
    n_arr = get_n(n, A)

    correct = np.zeros(T)
    K = A.shape[0]

    for t in range(T):
        Y = (np.random.uniform(size=(K, K)) <= y).astype(np.float64)
        AY = A * Y 

        W = sample_survivals(x, K, n_arr, rho, mode=correlation_mode).astype(np.int64)
        
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

def number_of_failures_lp(A, n, x, y, intervention_idx, intervention_value=0):
    K = A.shape[0]
    n_arr = get_n(n, A)
    if isinstance(x, float):
        ones = np.ones((K, 1))
        u = (x**n_arr) * ones
    else:
        u = x**n_arr

    u[intervention_idx] = intervention_value**n

    beta = cp.Variable((K, 1))
    objective = cp.Maximize(cp.sum(beta))

    constraints = [beta >= 0, beta <= ones, beta <= y * (A.T @ beta) + u]

    prob = cp.Problem(objective, constraints)
    result = prob.solve()

    return result

def resilience_correlation(args, A, y, labels):

    if os.path.exists(f'results_correlation_{args.name}.csv'):
        df = pd.read_csv(f'results_correlation_{args.name}.csv') 
        
    else:
        rho_range = np.linspace(0, 1, 4)

        results = []
        for key in sorted(A.keys()):
            # for mode in ['correlated_suppliers', 'correlated_products', 'correlated_all']:
            for mode in ['correlated_all']:
                x_hi = 1

                for rho in rho_range:
                    if x_hi > 0:
                        resilience_mc = estimate_resilience(A[key], args.n, T=100, intervention_idx=[], y=1, rho=rho, x_hi=x_hi, eps=args.eps, mode=mode)
                    else:
                        resilience_mc = 0

                    results.append({
                        'Network ID': key,
                        'rho' : rho,
                        'Resilience (MC)': resilience_mc,     
                        'mode' : mode               
                    })

                    x_hi = resilience_mc

        df = pd.DataFrame(results)

        df.to_csv(f'results_correlation_{args.name}.csv')

    fig, ax = plt.subplots(1, len(df['mode'].unique()), figsize=(len(df['mode'].unique()) * FIGSIZE, FIGSIZE), squeeze=False)

    ylim_min = df['Resilience (MC)'].min()
    ylim_max = df['Resilience (MC)'].max()

    for i, mode in enumerate(df['mode'].unique()):
        df_mode = df[df['mode'] == mode]
        sns.lineplot(data=df_mode, x='rho', y='Resilience (MC)', hue='Network ID', ax=ax[0, i], markers=True, style='Network ID', linewidth=2)

        if mode == 'correlated_all':
            ax[0, i].set_title('All Correlated')
        else:
            ax[0, i].set_title(mode.replace('_', ' ').title() + ' Only')
        
        ax[0, i].set_xlabel('$\\rho$')
        ax[0, i].set_ylabel('Resilience (MC)')
        ax[0, i].set_ylim(ylim_min, ylim_max)

    plt.tight_layout()


    plt.savefig(f'results_correlation_{args.name}.pdf', bbox_inches='tight')

def resilience_lb_lp(A, eps, y, n):
    
    if not n.isnumeric():
        raise ValueError(f'Only numeric values of n are supported, got {n}')

    K = A.shape[0]

    gamma = cp.Variable((K, 1))

    objective = cp.Minimize(cp.sum(gamma))

    constraints = [gamma >= 0, (np.eye(K) - y * A) @ gamma >= 1]

    prob = cp.Problem(objective, constraints)
    result = ((1 - eps) / prob.solve())**1/n

    gamma_dict = {i: gamma.value[i] for i in range(K)}

    return result, gamma_dict

def resilience_ub(A, eps, y, n):
    if not n.isnumeric():
        raise ValueError(f'Only numeric values of n are supported, got {n}')    

    K = A.shape[0]
    r = np.sum(np.sum(A, 1) == 0)

    print(f'K = {K}, r = {r}')

    num = (1 - eps) * K
    den = np.sqrt(2) * r**(3/2) + np.sqrt(r * np.log(K))

    return (num / den)**(1/n), r

def rei(A, eps, y, n, x, x_new):

    K = A.shape[0]
    PI = np.zeros(K)

    p_start = number_of_failures_lp(A, n, x, y, [], intervention_value=0)

    for i in range(K):
        p_end = number_of_failures_lp(A, n, x, y, [i], intervention_value=x_new)

        PI[i] = p_start - p_end / (x_new - x)

    REI = np.max(np.abs(PI))

    return REI

def rei_ttr_katz(A, eps, y, n, x):
    n_arr = get_n(n, A)

    katz_reverse = np.linalg.inv(np.eye(A.shape[0]) - y * A) @ np.ones((A.shape[0], 1))

    max_katz = np.max(katz_reverse)

    rei_ttr = n_arr * x**(n_arr - 1) * max_katz

    return rei_ttr

def rei_ttr_katz_auc(A, eps, y):

    katz_reverse = np.linalg.inv(np.eye(A.shape[0]) - y * A) @ np.ones((A.shape[0], 1))

    max_katz = np.max(katz_reverse)

    return max_katz

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

    plt.figure(figsize=(FIGSIZE, FIGSIZE))
    plt.title(f"{'Outdegree' if out else 'Indegree'} Distribution for US {get_extra_suptitle(args)}")
    plt.xlabel('Degree (log)')
    plt.ylabel('Frequency (log)')
    plt.xscale('log')
    plt.yscale('log')

    for key in A.keys():
        degrees, values, counts = degree_distribution(A[key], out=out)
        results = powerlaw.Fit(degrees, xmin=1.0)
        print(f'{get_label(args, key)}: alpha = {results.power_law.alpha}')

    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{'outdegree' if out else 'indegree'}_{args.name}.pdf", bbox_inches='tight')

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

    plt.figure(figsize=(FIGSIZE, FIGSIZE))
    plt.title('Tier Distribution for Supply-networks from Willems (2008)')
    plt.ylabel('Number of Nodes')
    plt.xlabel('Tier')

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

        labels[idx] = f'{idx}/{values.max() + 1}'

        plt.bar(values + 0.2 * i, counts, 0.2, label=f'{get_label(args, idx)}, Num Tiers: {values.max() + 1}, Num Edges: {int(A[idx].sum())}')

    plt.xticks(np.arange(1, 1 + max_val), np.arange(1, 1 + max_val))

    plt.xlim(1, max_val)
    plt.legend()
    plt.tight_layout()
    plt.savefig('statistics.pdf', bbox_inches='tight')


    return A, y, labels, depths

def load_random(args):

    A = {}
    y = {}
    labels = {} 

    if args.name in ['er', 'rdag']:
        rng = np.linspace(args.er_p_min, args.er_p_max, args.er_p_linspace)
    elif args.name == 'ba':
        rng = np.arange(args.ba_m_min, args.ba_m_max + 1, args.ba_m_step)
    elif args.name == 'sf':
        rng = [(0.41, 0.54, 0.05, 0.2, 0)]

    for r in rng:
        if args.name in ['er', 'rdag']:
            G = nx.erdos_renyi_graph(args.er_K, r, seed=args.seed, directed=True)
        elif args.name == 'ba':
            G = nx.barabasi_albert_graph(args.ba_K, r, seed=args.seed)
        elif args.name == 'sf':
            alpha, beta, gamma, delta_in, delta_out = r
            G = nx.scale_free_graph(args.sf_K, alpha=alpha, beta=beta, gamma=gamma, delta_in=delta_in, delta_out=delta_out)

        A[r] = nx.to_numpy_array(G)

        if args.name == 'rdag':
            A[r] = np.triu(A[r])

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
    elif args.name in ['er', 'rdag']:
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
    plt.figure(figsize=(FIGSIZE, FIGSIZE))
    plt.title(f'Lower bound on $R_G(\\varepsilon)$ for {get_extra_suptitle(args)}')
    plt.xlabel('Intervention Budget $T/K$')
    plt.ylabel('Resilience Lower Bound')

    for key in sorted(A.keys()):
        K = A[key].shape[0]
        T_range = np.arange(K + 1)
        I = np.eye(K, dtype=np.float64)
        beta_katz_inverse = np.linalg.inv(I - y[key] * A[key]).sum(-1)
        beta_katz_inverse_ordered_cumsum = np.cumsum(np.sort(beta_katz_inverse))[::-1]
        resilience_lb = np.zeros(K + 1)
        resilience_lb[1:] = (args.eps / beta_katz_inverse_ordered_cumsum)**(1/args.n)
        resilience_lb[0] = (args.eps / beta_katz_inverse.sum())**(1/args.n)
        if key == '':
            plt.plot(T_range / K, resilience_lb)
        else:
            plt.plot(T_range / K, resilience_lb, label=get_label(args, key))

    # plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.savefig(f'resilience_lb_vs_key_{args.name}.pdf')

def resilience_lb_vs_y(args, A, y, labels):
    if not n.isnumeric():
        raise ValueError(f'Only numeric values of n are supported, got {args.n}')
    
    key = get_key(args)
    
    K = A[key].shape[0]
    y_range = np.linspace(1e-5, y[key], 10)

    T_range = 1 + np.arange(K)
    I = np.eye(K, dtype=np.float64)

    plt.figure(figsize=(FIGSIZE, FIGSIZE))
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

def resilience_lb_heterogeneous(args, A, y):
    key = get_key(args)

    result, gamma = resilience_lb_lp(A[key], args.eps, y[key], args.n)

    G = nx.from_numpy_array(A[key], create_using=nx.DiGraph)

    print('gamma', gamma)

    pos = nx.random_layout(G, seed=args.seed)

    plt.figure()
    draw(G, pos, gamma, measure_name=f'Lower bound: {result:.3f}')

    plt.savefig(f'vulnerability_{args.name}.pdf')

def resilience_scatter(args, A, y, labels):
    results = []

    if os.path.exists(f'results_{args.name}.csv'):
        df = pd.read_csv(f'results_{args.name}.csv')
        
    else:
        for key in sorted(A.keys()):
            lower_bound, _ = resilience_lb_lp(A[key], args.eps, y[key], args.n)
            upper_bound, num_raw_products = resilience_ub(A[key], args.eps, y[key], args.n)
            resilience_mc = estimate_resilience(A[key], n=args.n, eps=args.eps, T=1000, y=y[key])

            rei_ttr = rei_ttr_katz(A[key], args.eps, y[key], args.n, x=resilience_mc)
            rei_tts = rei(A[key], args.eps, y[key], args.n, x=resilience_mc, x_new=1)

            results.append({
                'Network ID': key,
                'Lower Bound' : lower_bound,
                'Upper Bound': upper_bound,
                'Resilience (MC)' : resilience_mc,
                'REI (TTR)': rei_ttr,
                'REI (TTS)': rei_tts,
                'Raw Products Ratio': num_raw_products / A[key].shape[0],
                'Network Size' : A[key].shape[0],
                'REI (TTR) AUC': rei_ttr_katz_auc(A[key], args.eps, y[key]),
                'Resilience (MC) AUC': estimate_resilience_auc(A[key], args.n, T=1000, intervention_idx=[], y=y[key], num_linspace=10),
                'Number of Raw Products': num_raw_products,
                'Number of Tiers': int(labels[key].split('/')[1]) if args.name == 'msom_willems' else -1,
                'Max out-degree': A[key].sum(1).max()
            })

        df = pd.DataFrame(results)

        df.to_csv(f'results_{args.name}.csv')

    df['Number of Raw Products'] = df['Raw Products Ratio'] * df['Network Size']

    if args.name == 'msom_willems':
        df['Number of Tiers'] = df['Network ID'].apply(lambda x: int(labels[x].split('/')[1]))

    df['Max out-degree'] = df['Network ID'].apply(lambda x: A[x].sum(1).max())

    for col in df.columns:
        df["log-" + col] = np.log(df[col])

    # # Set color palette to viridis
    # sns.set_palette('viridis')

    # fig, ax = plt.subplots(1, 3, figsize=(3/2 *FIGSIZE, FIGSIZE / 2))

    # sns.scatterplot(data=df, x='Raw Products Ratio', y='REI (TTR)', ax=ax[0])

    # x_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)

    # ax[0].plot(x_range, reg.predict(x_range), color='black', label=f'$R^2 = {R2:.2f}$')

    # ax[0].set_xlabel('Raw Products Ratio')
    # ax[0].set_ylabel('REI (TTR)')

    # ax[0].legend()

    # sns.scatterplot(data=df, x='REI (TTR)', y='REI (TTS)', hue='Raw Products Ratio', size='Raw Products Ratio', ax=ax[1])

    # ax[1].get_legend().remove()


    # plt.tight_layout()    

    
    # plt.savefig(f'results_{args.name}.pdf')

    # Stata-like table

    # TTR
    est = brute_force_ols('log-REI (TTR) AUC', ['log-Resilience (MC) AUC', 'log-Network Size', 'log-Number of Raw Products', 'log-Max out-degree'], df)

    stargazer = Stargazer([est])

    with open(f'results_{args.name}_ttr.tex', 'w') as f:
        f.write(stargazer.render_latex())

    est = brute_force_ols('log-Resilience (MC) AUC', ['log-Network Size', 'log-Number of Raw Products', 'log-Max out-degree'], df)

    stargazer = Stargazer([est])

    with open(f'results_{args.name}_resilience.tex', 'w') as f:
        f.write(stargazer.render_latex())


def visualize(args, A, y, labels, num_ticks=2):
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
    plt.figure(figsize=(FIGSIZE, FIGSIZE))
    draw(G, pos, beta_katz_inverse_dict, f'Visualization for {get_extra_title(args)}', ticks=ticks, labels=ticks_labels)
    plt.savefig(f'visualization_{args.name}.pdf')

def resilience_monte_carlo_vs_eps(args, A):

    plt.figure(figsize=(FIGSIZE, FIGSIZE))
    plt.title(f'$\\hat R_G(\\varepsilon)$ (Monte-Carlo Estimate) for {get_extra_suptitle(args)}')
    plt.ylabel('$\\hat R_G(\\varepsilon)$')
    plt.xlabel('$\\epsilon$')

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
    
    plt.legend()
    plt.savefig(f'resilience_monte_carlo_vs_eps_{args.name}.pdf')

def expected_number_of_failures_vs_lp(args, A, y):
    plt.figure(figsize=(FIGSIZE, FIGSIZE))
    plt.title(f'Expected Number of Failures (Monte-Carlo) vs LP Upper Bound for {get_extra_suptitle(args)}')
    plt.xlabel('$x$')

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
    

def failures_distribution(args, A, y):
    fig, ax = plt.subplots()
    plt.ylabel('Number of Failures')
    plt.xlabel('Frequency', fontsize=FONTSIZE)
    T = 2000

    for key in A.keys():
        F = np.zeros(T)
        for i in range(T):
            F[i], _ = estimate(A[key], args.n, T=1, x=0.01, mode='failures', y=1, eps=0, intervention_idx=[])    
        sns.histplot(F, ax=ax, )
    
        break

    plt.legend()

    plt.legend(fontsize=0.75*FONTSIZE)
    plt.savefig(f'failures_distribution_{args.name}.pdf')

def resilience_monte_carlo_vs_intervention(args, A):
    
    plt.figure(figsize=(FIGSIZE, FIGSIZE))
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

    plt.legend(fontsize=0.75*FONTSIZE)
    plt.savefig(f'resilience_monte_carlo_vs_intervention_{args.name}.pdf')

if __name__ == '__main__':
    sns.set_theme()
    args = get_argparser()

    # Load/generate data
    if args.name == 'us_econ':
        A, y, labels = load_us_economy(args)
        depths = None
    elif args.name in ['er', 'ba', 'sf', 'rdag']:
        A, y, labels = load_random(args)
        depths = None
    elif args.name == 'msom_willems':
        A, y, labels, depths = load_msom_willems(args)
    elif args.name == 'wiot':
        A, y, labels = load_wiot(args)

   
    expected_number_of_failures_vs_lp(args, A, y)
    resilience_lb_vs_key(args, A, y, labels)
    resilience_lb_vs_y(args, A, y, labels)
    resilience_monte_carlo_vs_eps(args, A)
    resilience_monte_carlo_vs_intervention(args, A)
    resilience_correlation(args, A, y, labels)
    resilience_scatter(args, A, y, labels)





