import os
import json
import argparse
from tabulate import tabulate

def load_results(root_dir):
    results = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if 'result.jsonl' in filenames:
            file_path = os.path.join(dirpath, 'result.jsonl')
            with open(file_path, 'r') as f:
                for line in f:
                    try:
                        results.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"Error parsing line in {file_path}: {e}")
    return results

def print_results(results):
    if not results:
        print("No results found.")
        return

    # Determine all unique keys across all records
    all_keys = sorted({key for r in results for key in r.keys()})
    
    table = []
    for r in results:
        row = [r.get(k, '') for k in all_keys]
        table.append(row)

    print(tabulate(table, headers=all_keys, tablefmt='orgtbl'))

def main():
    parser = argparse.ArgumentParser(description="Print all sec_eval result.jsonl entries")
    parser.add_argument('--root_dir', type=str, required=True,
                        help='Path to the root eval directory (e.g. ../experiments/sec_eval/myrun/trained)')
    args = parser.parse_args()

    results = load_results(args.root_dir)
    print_results(results)

if __name__ == "__main__":
    main()
