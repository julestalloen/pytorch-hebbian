from pytorch_hebbian.lr_schedulers.lr_scheduler import LRScheduler


class Linear(LRScheduler):

    def get_updates(self, d_w, epoch, epochs):
        learning_rate = self.learning_rate * (1 - epoch / epochs)
        updates = learning_rate * d_w
        self.updates.append(updates)

        return updates
