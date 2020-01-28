import torchvision


def write_stats(visualizer, model, data_loader, params):
    """Visualize the model, some input samples and the hyperparameters"""
    images, labels = next(iter(data_loader))
    visualizer.writer.add_graph(model, images)
    visualizer.writer.add_image('input/samples', torchvision.utils.make_grid(images[:64]))
    num_project = 100
    visualizer.project(images[:num_project], labels[:num_project], params['input_size'])
    visualizer.writer.add_hparams(params, {})
