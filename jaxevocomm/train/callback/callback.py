class TrainerCallback:

    def on_train_begin(self, config):
        pass

    def on_train_end(self, training_state):
        pass

    def on_iteration_end(self, iteration, training_state, metrics):
        pass


class ChainedCallback(TrainerCallback):

    def __init__(self, *callbacks):
        assert all(isinstance(cb, TrainerCallback) for cb in callbacks)
        self.callbacks = callbacks

    def on_train_begin(self, config):
        for cb in self.callbacks:
            cb.on_train_begin(config)

    def on_train_end(self, training_state):
        for cb in self.callbacks:
            cb.on_train_end(training_state)

    def on_iteration_end(self, iteration, training_state, metrics):
        for cb in self.callbacks:
            cb.on_iteration_end(iteration, training_state, metrics)
