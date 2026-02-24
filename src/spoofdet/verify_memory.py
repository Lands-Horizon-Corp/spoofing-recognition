# fmt: off
from __future__ import annotations

import os

import psutil


def print_memory_usage(step_name):
    process = psutil.Process(os.getpid())
    # rss (Resident Set Size) is the standard measure of RAM actually held in memory
    ram_usage_mb = process.memory_info().rss / (1024 * 1024)
    print(f"[{step_name}] RAM Usage: {ram_usage_mb:.2f} MB")
# def measure_time_and_memory():
#     # STEP 1: Baseline (Python interpreter only)
#     print_memory_usage('------Baseline (Python Only)')
#     # STEP 2: The "PyTorch Tax"
#     print('Importing PyTorch...')
#     import torch
#     print_memory_usage("-------After 'import torch'")
#     # STEP 4: Quantized Model (Approximate check)
#     # We can't easily quantize in one line here without your specific code,
#     # but this shows the relative impact of the weights vs the library.
#     model = torch.jit.load('src/spoofdet/effi
#  cient_net/quantized_model_scripted.pt') # flake8: noqa
#     print_memory_usage('-------After Loading Quantized Model')
# # fmt: on
