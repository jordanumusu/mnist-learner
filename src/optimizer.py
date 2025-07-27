class BasicOptimiser:
    def __init__(self, params, lr): 
        self.params = params
        self.lr = lr

    def step(self):
        for p in self.params: p -= p.grad.data * self.lr

    def zero_grad():
        for p in self.params: p.grad = Nonein 
