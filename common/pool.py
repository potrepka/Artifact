import random
import torch

class Pool:
    def __init__(self, pool_size):
        assert pool_size >= 0
        self.pool_size = pool_size
        self.data = []

    def query(self, x):
        if self.pool_size == 0:
            return x
        result = []
        for item in x:
            item = torch.unsqueeze(item.data, 0)
            if len(self.data) < self.pool_size:
                self.data.append(item)
            else:
                if random.uniform(0, 1) < 0.5:
                    index = random.randint(0, self.pool_size - 1)
                    temp = self.data[index].clone().type_as(item)
                    self.data[index] = item
                    item = temp
            result.append(item)
        result = torch.cat(result, 0)
        return result
