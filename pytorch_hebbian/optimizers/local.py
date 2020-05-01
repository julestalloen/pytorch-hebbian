from torch.optim.optimizer import Optimizer, required


class Local(Optimizer):

    def __init__(self, named_params, lr=required):
        self.param_names, params = zip(*named_params)

        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))

        defaults = dict(lr=lr)
        super(Local, self).__init__(params, defaults)

    def local_step(self, d_p, layer_name, closure=None):
        """Performs a single local optimization step."""
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            layer_index = self.param_names.index(layer_name + '.weight')
            p = group['params'][layer_index]
            p.data.add_(group['lr'] * d_p)

        try:
            self._step_count += 1
        except AttributeError:
            pass

        return loss
