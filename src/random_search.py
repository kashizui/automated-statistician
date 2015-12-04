"""
random_search.py

Author: Stephen Koo

Randomly search the model-hyperparameter space and report the best performance.
"""
import random
import models
import sys


class RandomSearcher(object):
    def __init__(self):
        self.models = models.list_classification_models()

    @staticmethod
    def print_line(*tokens):
        print ', '.join(str(t) for t in tokens)
        sys.stdout.flush()

    def run(self, dataset, time_limit):
        random.seed(1337)
        self.print_line('seconds_left', 'model', 'hyperparameter', 'performance', 'runtime')
        try:
            time_left = time_limit
            while time_left > 0:
                name, hp, perf, runtime = self.run_random_model(dataset)
                self.print_line(time_left, name, hp, perf, runtime)
                time_left -= runtime
        except KeyboardInterrupt:
            print "Caught keyboard interrupt, finishing early..."

    def run_random_model(self, dataset):
        model = random.choice(self.models)
        hp = random.random()
        perf, runtime = model.fit(dataset, (hp,), verbose=False)
        return model.__name__, hp, perf, runtime
