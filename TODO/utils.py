import numpy as np


class CustomEarlyStopper:
    def __init__(self, no_improvement_rounds=3):
        self.no_improvement_rounds = no_improvement_rounds
        self.best_score = -np.inf
        self.rounds_without_improvement = 0

    def __call__(self, result):
        current_score = result.func_vals[-1]
        if current_score > self.best_score:
            self.best_score = current_score
            self.rounds_without_improvement = 0
        else:
            self.rounds_without_improvement += 1

        if self.rounds_without_improvement >= self.no_improvement_rounds:
            return True
        return False
