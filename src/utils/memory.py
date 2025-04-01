import psutil
import torch


def print_memory_allocated():
    if torch.cuda.is_available():
        print("***")
        print(f"Memory allocated: {torch.cuda.memory_allocated() / 1024**2} MB")

    print(f"Memory usage: {psutil.Process().memory_info().rss / 1024 / 1024:.2f} MB")
    print("***")


def print_tensor_memory(tensor):
    print(
        f"Tensor memory: {tensor.element_size() * tensor.numel() / 1024 / 1024:.2f} MB"
    )
