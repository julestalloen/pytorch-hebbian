from optimizers.optimizer import Optimizer


class Linear(Optimizer):

    def get_updates(self, d_w, epoch, epochs):
        learning_rate = self.learning_rate * (1 - epoch / epochs)
        updates = learning_rate * d_w
        self.updates.append(updates)

        return updates
