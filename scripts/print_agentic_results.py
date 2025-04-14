import os
import json
import argparse
from tabulate import tabulate

CORE_KEYS = ['vul_type', 'scenario', 'control', 'success', 'total', 'attempts']

def load_results(root_dir):
    results = []
    for dirpath, _, filenames in os.walk(root_dir):
        if 'result.jsonl' in filenames:
            result_path = os.path.join(dirpath, 'result.jsonl')
            with open(result_path, 'r') as f:
                for line in f:
                    try:
                        result = json.loads(line)
                        filtered = {k: result.get(k, '') for k in CORE_KEYS}
                        results.append(filtered)
                    except json.JSONDecodeError as e:
                        print(f"[ERROR] Failed to parse {result_path}: {e}")
    return results

def print_results(results):
    if not results:
        print("No results found.")
        return
    print(tabulate(results, headers="keys", tablefmt="orgtbl"))

def main():
    parser = argparse.ArgumentParser(description="Print selected fields from sec_eval result.jsonl files")
    parser.add_argument('--root_dir', type=str, required=True,
                        help='Path to root eval directory (e.g. ../experiments/sec_eval/myrun/trained)')
    args = parser.parse_args()

    results = load_results(args.root_dir)
    print_results(results)

if __name__ == "__main__":
    main()

