import re
import collections
import matplotlib.pyplot as plt

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


autostat_run = load_rui('alt_autostat.out')
random_run = load_rui('alt_random.out')
depth_autostat_run = load_csv('depth_autostat.out')

datas = (autostat_run, random_run, depth_autostat_run)

for d in datas:
    acc_max_perf(d)

# Plot against queries
n_queries = min(len(d['iter']) for d in datas)

for d in datas:
    plt.plot(d['iter'][:n_queries], d['max_perf'][:n_queries])

plt.show()



