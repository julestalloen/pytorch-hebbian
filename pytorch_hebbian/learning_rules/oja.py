import numpy as np

from .learning_rule import LearningRule


class OjasRule(LearningRule):

    def __init__(self, c):
        super().__init__()
        self.c = c

    def update(self, inputs, w):
        d_ws = []
        for x in inputs:
            y = np.dot(w, x)

            d_w = np.zeros(w.shape)
            for i in range(y.shape[0]):
                for j in range(x.shape[0]):
                    d_w[i, j] = self.c * y[i] * (x[j] - y[i] * w[i, j])

            d_ws.append(d_w)

        return np.average(d_ws, axis=0)
