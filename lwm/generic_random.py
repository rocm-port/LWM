import jax.random as random
import torch
import torch2jax.torch2jax as torch2jax

# generic, repeatable RNG
class generic_random:
    def __init__(self, seed:int=0):
        self.key = random.key(seed)

    def randint(self, size: tuple, max:int):
        return random.randint(self.key, size, 0, max)

    def normal(self, size: tuple, dtype):
        return random.normal(self.key, size, dtype)

    def randint_torch(self, size: tuple, max, device, dtype):
        return torch2jax.j2t(self.randint(size, max)).to(device=device, dtype=dtype)

    def normal_torch(self, size: tuple, dtype):
        dtypemap={
            torch.float32: "float32",
            torch.float16: "float16",
            torch.bfloat16: "bfloat16",
            torch.int32: "int32",
            torch.int32: "int32",
        }

        return torch2jax.j2t(self.normal(size, dtypemap[dtype]))
