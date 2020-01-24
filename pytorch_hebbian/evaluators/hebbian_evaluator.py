import torch


class HebbianEvaluator:

    def __init__(self, model, data_loader, epochs=100):
        self.model = model
        self.data_loader = data_loader
        self.epochs = epochs

    def run(self):
        optimizer = torch.optim.Adam(self.model.parameters())
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
        criterion = torch.nn.CrossEntropyLoss()
        # TODO
        # evaluator = SupervisedEvaluator(data_loader=self.data_loader, model=self.model, loss_criterion=criterion)
        # supervised_engine = SupervisedEngine(optimizer=optimizer,
        #                                      lr_scheduler=lr_scheduler,
        #                                      criterion=criterion,
        #                                      evaluator=evaluator)
        #
        # # Freeze all but final layer
        # for layer in list(self.model.children())[:-1]:
        #     for param in layer.parameters():
        #         param.requires_grad = False
        #
        # # Train with gradient descent and evaluate
        # supervised_engine.train(model=self.model, data_loader=self.data_loader, epochs=self.epochs, eval_every=10)
        #
        # return {'loss': min(evaluator.losses), 'acc': max(evaluator.accuracies)}
