import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input", "-i", type=str, default="data/ProblemB_case4_20230816.txt")
parser.add_argument("--output", "-o", type=str, default="data/output.txt")
parser.add_argument("--prob", "-p", type=float, default=0.7)
args = parser.parse_args()

ori_data = open(args.input, "r")
output_data = open(args.output, "w")
prob = args.prob

def output_nets(ori_data: list, prob: float) -> list:
    cache = []
    out = []
    for line in ori_data:
        if line.startswith("Net"):
            if len(cache) > 0 and random.random() < prob:
                # output_data.writelines(cache)
                out.append(cache)
            cache = []
        cache.append(line)
    if len(cache) > 0 and random.random() < prob:
        # output_data.writelines(cache)
        out.append(cache)
    return out

ori_data = ori_data.readlines()
slice_index = 0
for i in range(len(ori_data)):
    if ori_data[i].startswith("NumNets"):
        slice_index = i
output_data.writelines(ori_data[:slice_index])

randomly_picked_nets = output_nets(ori_data[slice_index+1:], prob)

output_data.write(f"NumNets {len(randomly_picked_nets)}\n")
for net in randomly_picked_nets:
    output_data.writelines(net)