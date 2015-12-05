import re
import collections
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KernelDensity

COUNTER = r"Counter: (\d+) Training: (\w+)\(x=([-.\d]+)\). Time left: ([.\d]+)"
RESULT = r"...Result: perf=([-.\d]+), time=([.\d]+)"


def load_rui(path):
    with open(path, 'r') as f:
        data = collections.defaultdict(list)

        for line in f:
            m = re.match(COUNTER, line)
            if m is not None:
                i, model, hp, time_left = m.groups()
                data['iter'].append(i)
                data['model'].append(model)
                data['hp'].append(hp)
                data['time_left'].append(time_left)

            m = re.match(RESULT, line)
            if m is not None:
                perf, runtime = m.groups()
                data['perf'].append(perf)
                data['runtime'].append(runtime)

    return data


def load_csv(path):
    with open(path, 'r') as f:
        data = collections.defaultdict(list)
        for line in f:
            i, time_left, model, hp, perf, runtime = line.split(', ')
            data['iter'].append(i)
            data['model'].append(model)
            data['hp'].append(hp)
            data['time_left'].append(time_left)
            data['perf'].append(perf)
            data['runtime'].append(runtime)

    return data


def acc_max_perf(data):
    max_perf = float('-inf')
    for perf in data['perf']:
        max_perf = max(max_perf, perf)
        data['max_perf'].append(max_perf)


def acc_model_counts(data):
    model_counts = collections.defaultdict(list)
    for i, model in zip(data['iter'], data['model']):
        model_counts[model].append(i)

    data['counts'] = model_counts


def plot_density(data, name):
    # n_queries = len(data['iter'])

    iters_plot = np.arange(0, len(data['iter']))[:, np.newaxis]
    plt.figure()
    for model, counts in data['counts'].iteritems():
        counts = np.int32(counts)[:, np.newaxis]
        kde = KernelDensity(kernel='gaussian', bandwidth=2.5).fit(counts)
        density = np.exp(kde.score_samples(iters_plot))
        plt.plot(iters_plot[:, 0], density, label=model)
        # plt.fill(iters_plot[:, 0], np.exp(log_dens), fc='#AAAAFF')
        # plt.text(-3.5, 0.31, "Gaussian Kernel Density")
    plt.title("Model Selection Frequency Density for " + name)
    plt.xlabel("iterations")
    plt.ylabel("frequency of selection per iteration")
    plt.legend()
    plt.show()


thompson_run = load_rui('alt_autostat.out')
random_run = load_rui('alt_random.out')
depth_autostat_run = load_csv('depth_autostat.out')

datas = (thompson_run, random_run, depth_autostat_run)

for d in datas:
    acc_max_perf(d)
    acc_model_counts(d)



# Plot against queries
n_queries = min(len(d['iter']) for d in datas)

plt.axis([0,27,0.55,1])
d = datas[0]
plt.plot(d['iter'][:n_queries], d['max_perf'][:n_queries], label='Thompson Sampling')
d = datas[2]
plt.plot(d['iter'][:n_queries], d['max_perf'][:n_queries], label='Lookahead')
d = datas[1]
plt.plot(d['iter'][:n_queries], d['max_perf'][:n_queries], label='Random Search')
plt.legend()
plt.xlabel('Iterations')
plt.ylabel('Best Observed Performance')

    
plt.savefig('iteration_based.eps', format='eps', dpi=1000)

# Plot frequency of selecting certain models
plot_density(thompson_run, 'Thompson Sampling')
plot_density(depth_autostat_run, 'Lookahead')