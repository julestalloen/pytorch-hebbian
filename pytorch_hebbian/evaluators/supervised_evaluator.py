from pytorch_hebbian.evaluators.evaluator import Evaluator


class SupervisedEvaluator(Evaluator):

    def __init__(self, model, data_loader, metrics='acc'):
        super().__init__(model, data_loader)
        self.metrics = metrics

    def run(self):
        # TODO
        return {metric: 0 for metric in self.metrics}
