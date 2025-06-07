import math
import matplotlib.pyplot as plt
import numpy as np

# constants
NUM_NODES = 4
THROUGHPUT = 50e9       # 50 GOPS per node
NUM_LINKS = 12
COMM_BW = 2e9           # 2 GB/s per link
MAC_TO_OPS = 2          # 1 MAC = 2 OPS
BYTES_PER_FP16 = 2

# CNN models
model1 = [
    # number of filers, number of input channels, fx, fy, input activation dimension
    # kernel size is fx x fy
    {"filters": 128, "in_ch": 3, "fx": 3, "fy": 3, "in_dim": 32},
    {"filters": 128, "in_ch": 128, "fx": 3, "fy": 3, "in_dim": 30},
    {"filters": 128, "in_ch": 128, "fx": 3, "fy": 3, "in_dim": 28},
    {"filters": 128, "in_ch": 128, "fx": 3, "fy": 3, "in_dim": 26},
]

model2 = [
    {"filters": 128, "in_ch": 3, "fx": 3, "fy": 3, "in_dim": 32},
    {"filters": 128, "in_ch": 128, "fx": 3, "fy": 3, "in_dim": 30},
    {"filters": 128, "in_ch": 128, "fx": 3, "fy": 3, "in_dim": 14},
    {"filters": 128, "in_ch": 128, "fx": 3, "fy": 3, "in_dim": 12},
]

def compute_layer_latency_ops(layer):
    out_dim = layer["in_dim"] - 2   # no padding and the kernel size is 3x3
    ops = (
        layer["filters"]
        * layer["in_ch"]
        * layer["fx"]
        * layer["fy"]
        * out_dim           # output height * output width = the number of spacital locations
        * out_dim
        * MAC_TO_OPS
    )

    return ops, (layer["filters"], out_dim, out_dim)

"""
- Each layer is split across nodes
- Nodes share partial outputs after each layer

"""
def tensor_parallelism(model, num_inputs):
    total_ops = 0
    total_comm = 0

    for layer in model:
        ops, (ch, h, w) = compute_layer_latency_ops(layer)
        total_ops += ops
        activation_size = ch*h*w*BYTES_PER_FP16
        # need NUM_NODES-1 times to connect nodes and compute the whole outputs for a layer
        total_comm += activation_size * (NUM_NODES - 1)
    
    latency = total_ops / (THROUGHPUT * NUM_NODES)
    total_latency = latency * num_inputs
    total_comm *= num_inputs

    return total_latency*1000, total_comm/1024        # convert the units to ms and KB

"""
- Each node computes a layer
- Layers have dependencies, the output of one layer is the input of the next
    - the total latency = startup time + (N-1)*max layer time
- After each layer, the output is sent to the next node
- For one input: full pipeline time = sum of all layers
"""
def pipeline_parallelism(model, num_inputs):
    total_comm = 0
    layer_latencies = []

    for layer in model:
        ops, (ch, h, w) = compute_layer_latency_ops(layer)
        layer_latencies.append(ops/THROUGHPUT)
        if layer != model[-1]:
            activation_size = ch*h*w*BYTES_PER_FP16
            total_comm += activation_size
    
    total_latency = sum(layer_latencies) + (num_inputs - 1) * max(layer_latencies)
    total_comm *= num_inputs

    return total_latency*1000, total_comm / 1024    # convert the units to ms and KB

# Q1 - single input
tp_1_latency, tp_1_traffic = tensor_parallelism(model1, 1)
pp_1_latency, pp_1_traffic = pipeline_parallelism(model1, 1)

# Q2 - 32 inputs
tp_32_latency, tp_32_traffic = tensor_parallelism(model1, 32)
pp_32_latency, pp_32_traffic = pipeline_parallelism(model1, 32)

# Q3 - model with pooling
q3_tp_32_latency, q3_tp_32_traffic = tensor_parallelism(model2, 32)
q3_pp_32_latency, q3_pp_32_traffic = pipeline_parallelism(model2, 32)


# Print results in required format
print(f"Q1: TP total latency: {tp_1_latency:.2f} ms")
print(f"Q1: TP total network traffic: {tp_1_traffic:.2f} KB")
print(f"Q1: PP total latency: {pp_1_latency:.2f} ms")
print(f"Q1: PP total network traffic: {pp_1_traffic:.2f} KB")
print(f"Q2: TP total latency: {tp_32_latency:.2f} ms")
print(f"Q2: TP total network traffic: {tp_32_traffic:.2f} KB")
print(f"Q2: PP total latency: {pp_32_latency:.2f} ms")
print(f"Q2: PP total network traffic: {pp_32_traffic:.2f} KB")
print(f"Q3: TP total latency: {q3_tp_32_latency:.2f} ms")
print(f"Q3: TP total network traffic: {q3_tp_32_traffic:.2f} KB")
print(f"Q3: PP total latency: {q3_pp_32_latency:.2f} ms")
print(f"Q3: PP total network traffic: {q3_pp_32_traffic:.2f} KB")
