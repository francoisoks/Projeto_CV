class EarlyStop:
    def __init__(self, patience=3, delta=0.01):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def check(self, loss):

        if self.best_loss is None:
            self.best_loss = loss
        elif loss > self.best_loss - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = loss
            self.counter = 0
        print(f'count: {self.counter} best: {self.best_loss} delta: {self.delta} patience: {self.patience}')
        return self.early_stop
