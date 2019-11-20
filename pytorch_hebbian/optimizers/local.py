from torch.optim.optimizer import Optimizer, required


class Local(Optimizer):

    def __init__(self, params, lr=required):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))

        defaults = dict(lr=lr)
        super(Local, self).__init__(params, defaults)

    def local_step(self, d_p, closure=None):
        """Performs a single local optimization step."""
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                p.data.add_(-group['lr'] * d_p)

        # TODO: dirty fix
        self._step_count += 1

        return loss
