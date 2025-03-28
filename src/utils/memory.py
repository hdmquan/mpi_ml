import psutil


def print_memory_usage():
    print(f"Memory usage: {psutil.Process().memory_info().rss / 1024 / 1024:.2f} MB")


def print_tensor_memory(tensor):
    print(
        f"Tensor memory: {tensor.element_size() * tensor.numel() / 1024 / 1024:.2f} MB"
    )
