import os
import re
import csv
import numpy as np
from tools import CalcTopMap

# 获取当前脚本（evaluate_model.py）所在的真实目录
script_dir = os.path.dirname(os.path.abspath(__file__))
output_csv = os.path.join(script_dir, "evaluation_results.csv")

def find_best_result_files(folder):
    pattern = re.compile(r"^(.+)-(\d+\.\d{10})-trn_binary\.npy$")
    candidates = {}

    for fname in os.listdir(folder):
        match = pattern.match(fname)
        if match:
            score = float(match.group(2))
            key = match.group(1) + f"-{score:.10f}"
            prefix_path = os.path.join(folder, key)
            candidates[score] = {
                "trn_binary": prefix_path + "-trn_binary.npy",
                "trn_label": prefix_path + "-trn_label.npy",
                "tst_binary": prefix_path + "-tst_binary.npy",
                "tst_label": prefix_path + "-tst_label.npy",
                "model": prefix_path + "-model.pt" if os.path.exists(prefix_path + "-model.pt") else None
            }

    if not candidates:
        return None

    best_score = max(candidates.keys())
    return best_score, candidates[best_score]

# topk要与训练阶段的参数保持一致，在这里
#  if "cifar" in config["dataset"]:
#         config["topK"] = -1
# 表示全量，即59000
def evaluate_all_models(base_dir="save", output_csv=output_csv, topk=-1):
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["method", "bit", "MAP", "trn_binary", "tst_binary", "model_file"])
        writer.writeheader()

        for method in os.listdir(base_dir):
            method_path = os.path.join(base_dir, method)
            if not os.path.isdir(method_path):
                continue

            result = find_best_result_files(method_path)
            if not result:
                print(f"❌ No valid result found in {method}")
                continue

            best_score, files = result
            print(f"\n[📂] {method} — Best MAP: {best_score:.4f}")

            trn_binary = np.load(files["trn_binary"])
            trn_label = np.load(files["trn_label"])
            tst_binary = np.load(files["tst_binary"])
            tst_label = np.load(files["tst_label"])

            map_score = CalcTopMap(trn_binary, tst_binary, trn_label, tst_label, topk=topk)
            print(f"✅ Final Recalculated MAP@{topk} = {map_score:.4f}")

            writer.writerow({
                "method": method,
                "bit": trn_binary.shape[1],
                "MAP": f"{map_score:.4f}",
                "trn_binary": os.path.basename(files["trn_binary"]),
                "tst_binary": os.path.basename(files["tst_binary"]),
                "model_file": os.path.basename(files["model"]) if files["model"] else "None"
            })

if __name__ == "__main__":
    evaluate_all_models(base_dir="save", output_csv="utils/evaluation_results.csv")
