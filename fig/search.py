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


def acc_max_perf(data):
    max_perf = float('-inf')
    for perf in data['perf']:
        max_perf = max(max_perf, perf)
        data['max_perf'].append(max_perf)


autostat_run = load_rui('alt_autostat.out')
random_run = load_rui('alt_random.out')

for d in (autostat_run, random_run):
    acc_max_perf(d)


# Plot against queries
n_queries = max(len(d['iter']) for d in (autostat_run, random_run))

plt.plot(autostat_run['iter'][:n_queries], autostat_run['max_perf'][:n_queries])
plt.show()



