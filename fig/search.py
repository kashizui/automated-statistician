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
    acc_model_counts(data)
    total_queries = len(data['iter'])
    iters_plot = np.arange(0, 60)[:, np.newaxis]

    # time_plot = np.linspace(0, 60., 1000)[:, np.newaxis]

    plt.figure()
    plt.axis([0, 60, 0, .012])
    for model, counts in data['counts'].iteritems():
        counts = np.float32(counts)[:, np.newaxis]
        kde = KernelDensity(kernel='gaussian', bandwidth=2.5).fit(counts)
        density = np.exp(kde.score_samples(iters_plot))
        weighted_density = density * len(counts) / total_queries
        plt.plot(iters_plot[:, 0], weighted_density, label=model)
        # plt.fill(iters_plot[:, 0], np.exp(log_dens), fc='#AAAAFF')
        # plt.text(-3.5, 0.31, "Gaussian Kernel Density")
    plt.xlabel("Iterations")
    plt.ylabel("Selection Density")
    plt.legend()
    plt.savefig("density_" + name.lower().replace(' ', '_') + ".eps", format='eps', dpi=1000)


thompson_run = load_rui('alt_autostat.out')
random_run = load_rui('alt_random.out')
depth_autostat_run = load_csv('depth_autostat.out')

datas = (thompson_run, random_run, depth_autostat_run)

for d in datas:
    acc_max_perf(d)

# Plot against queries
min_queries = min(len(d['iter']) for d in datas)
plt.axis([0, 27, 0.55, 1])
d = datas[0]
plt.plot(d['iter'][:min_queries], d['max_perf'][:min_queries], label='Thompson Sampling')
d = datas[2]
plt.plot(d['iter'][:min_queries], d['max_perf'][:min_queries], label='Lookahead')
d = datas[1]
plt.plot(d['iter'][:min_queries], d['max_perf'][:min_queries], label='Random Search')
plt.legend()
plt.xlabel('Iterations')
plt.ylabel('Best Observed Performance')
plt.savefig('iteration_based.eps', format='eps', dpi=1000)

# Plot against time
plt.figure()
# plt.axis([0, 27, 0.55, 1])
d = datas[0]
plt.plot(60. - np.float32(d['time_left']), d['max_perf'], label='Thompson Sampling')
d = datas[2]
plt.plot(60. - np.float32(d['time_left']), d['max_perf'], label='Lookahead')
d = datas[1]
plt.plot(60. - np.float32(d['time_left']), d['max_perf'], label='Random Search')
plt.legend(loc="lower right")
plt.xlabel('Time Elapsed (seconds)')
plt.ylabel('Best Observed Performance')
plt.savefig('time_based.eps', format='eps', dpi=1000)


# Plot frequency of selecting certain models
plot_density(thompson_run, 'Thompson Sampling')
plot_density(depth_autostat_run, 'Lookahead')
