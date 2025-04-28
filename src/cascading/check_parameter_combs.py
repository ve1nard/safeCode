import json
import pandas as pd
import numpy as np
import random
import os
import argparse
from itertools import product, permutations

sol_number_set = [-1,0,1,3,5,10]
test_number_set = [0,2,4]
thresholds_set = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

MODEL_FAMILIES = {
    "wizardCoder": [
        "vanillaOVO/WizardCoder-Python-7B-V1.0",
        "WizardLMTeam/WizardCoder-Python-13B-V1.0",
        "WizardLMTeam/WizardCoder-Python-34B-V1.0"
    ],
}

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--family', type=str, default='wizardCoder')
    parser.add_argument('--num_loops', type=int, default=10)
    parser.add_argument('--outputs', type=str, default='../../data/cascading/outputs')
    parser.add_argument('--selected', type=str, default='../../data/cascading/selected')
    parser.add_argument('--costs', type=str, default='../../data/cascading/model_costs.json')
    return parser.parse_args()


# Initialize all combinations of k and testlines
combs = list(product(sol_number_set, repeat=3))
def is_valid_combination(comb):
    # Exclude combinations where all entries are -1
    if set(comb) == {-1}:
        return False
    # Exclude combinations where an early element is 0, and a later element has a non-negative value.
    after_zero = False
    for val in comb:
        if after_zero and val >= 0:
            return False
        if val == 0:
            after_zero = True
    return True

all_k_combos = [perm for comb in combs for perm in set(permutations(comb)) if is_valid_combination(perm)]
all_k_combos = list(set(all_k_combos))
all_k_combos = sorted(all_k_combos, key=lambda x: (x[0], x[1], x[2]))
all_t_combos = list(product(test_number_set, repeat=3))

def is_bad_combo(k, t):
    if k<=0 and t>0:
        return True
    if k>0 and t==0:
        return True
    return False

def main():
    args = get_args()
    cascade_results_path = os.path.join(args.outputs, "cascade_results")
    if not os.path.exists(cascade_results_path):
        os.mkdir(cascade_results_path)
    model_names = [checkpoint.split('/')[-1] for checkpoint in MODEL_FAMILIES[args.family]]
    with open(args.costs, "r") as f:
            costs_list = json.load(f)

    # Generation cost per 1k tokens
    cpt_1 = costs_list[model_names[0]]
    cpt_2 = costs_list[model_names[1]]
    cpt_3 = costs_list[model_names[2]]
    
    for threshold in thresholds_set:
        val_numbers_len = 79
        output_file_name_val = os.path.join(cascade_results_path, f"val_threshold{threshold}.csv")

        df_result = pd.DataFrame(columns=["k1", "k2", "k3", "t1", "t2", "t3", "loop", "cost", "accuracy"])

        for (k1, k2, k3) in all_k_combos:
            for (t1, t2, t3) in all_t_combos:
                this_num_loops = 1 if (k1 < 1 and k2 < 1 and k3 < 1) else args.num_loops
                for loop in range(this_num_loops):
                    if is_bad_combo(k1, t1) or is_bad_combo(k2, t2) or is_bad_combo(k3, t3):
                        continue
                    
                    total_cost = 0.0
                    total_correct = 0
                    all_numbers_left = list(range(val_numbers_len))
                    
                    if k1 > -1:
                        if k1 == 0:
                            selected_file_1 = os.path.join(args.selected, f"{model_names[0]}/{model_names[0]}_greedy.json")
                        else:
                            selected_file_1 = os.path.join(args.selected, f"{model_names[0]}/{model_names[0]}_p{k1}_t{t1}_l{loop}.json")
                        if os.path.exists(selected_file_1):
                            selected_answers_1 = json.load(open(selected_file_1, "r"))
                            for selected_dict in selected_answers_1:
                                number = selected_dict["problem"]
                                if number in all_numbers_left:
                                    if selected_dict["max_answer_num"] != 0 and selected_dict["max_test_num"] != 0:
                                        total_cost += selected_dict["total_length_in_tokens"] * cpt_1
                                        confidence = selected_dict["max_product"]
                                        total_product = selected_dict["total_product"]
                                        adopt = (confidence >= total_product*threshold)
                                        if k2 == -1 and k3 == -1:
                                            adopt = True
                                        if adopt:
                                            all_numbers_left.remove(number)
                                            if selected_dict["valid"]:
                                                total_correct += 1
                    
                    print(f"{k1}{k2}{k2}: {total_correct}")
                        
                    
                    if k2 > -1:
                        if k2 == 0:
                            selected_file_2 = os.path.join(args.selected, f"{model_names[1]}/{model_names[1]}_greedy.json")
                        else:
                            selected_file_2 = os.path.join(args.selected, f"{model_names[1]}/{model_names[1]}_p{k2}_t{t2}_l{loop}.json")
                        if os.path.exists(selected_file_2):
                            selected_answers_2 = json.load(open(selected_file_2, "r"))
                            for selected_dict in selected_answers_2:
                                number = selected_dict["problem"]
                                if number in all_numbers_left:
                                    if selected_dict["max_answer_num"] != 0 and selected_dict["max_test_num"] != 0:
                                        total_cost += selected_dict["total_length_in_tokens"] * cpt_2
                                        confidence = selected_dict["max_product"]
                                        total_product = selected_dict["total_product"]
                                        adopt = (confidence >= total_product*threshold)
                                        if k3 == -1:
                                            adopt = True
                                        if adopt:
                                            all_numbers_left.remove(number)
                                            if selected_dict["valid"]:
                                                total_correct += 1
                    print(f"{k1}{k2}{k2}: {total_correct}")
                    if k3 > -1:
                        if k3 == 0:
                            selected_file_3 = os.path.join(args.selected, f"{model_names[2]}/{model_names[2]}_greedy.json")
                        else:
                            selected_file_3 = os.path.join(args.selected, f"{model_names[2]}/{model_names[2]}_p{k3}_t{t3}_l{loop}.json")
                        if os.path.exists(selected_file_3):
                            selected_answers_3 = json.load(open(selected_file_3, "r"))
                            for selected_dict in selected_answers_3:
                                number = selected_dict["problem"]
                                if number in all_numbers_left:
                                    if selected_dict["max_answer_num"] != 0 and selected_dict["max_test_num"] != 0:
                                        total_cost += selected_dict["total_length_in_tokens"] * cpt_3
                                        all_numbers_left.remove(number)
                                        if selected_dict["valid"]:
                                            total_correct += 1
                                    
                    total_accuracy = total_correct / val_numbers_len
                    df_result.loc[len(df_result)] = [k1, k2, k3, t1, t2, t3, loop, total_cost, total_accuracy]
                    print(f"k1: {k1}, k2: {k2}, k3: {k3}, t1: {t1}, t2: {t2}, t3: {t3}, loop: {loop}, cost: {total_cost}, accuracy: {total_accuracy}")

        # Write df_result
        avg_df = df_result.groupby(['k1', 'k2', 'k3', 't1', 't2', 't3']).agg({
            'loop': 'last',
            'cost':'mean', 
            'accuracy':'mean'
        }).reset_index()
        # Convert ks and ts to integers
        avg_df['k1'] = avg_df['k1'].astype(int)
        avg_df['k2'] = avg_df['k2'].astype(int)
        avg_df['k3'] = avg_df['k3'].astype(int)
        avg_df['t1'] = avg_df['t1'].astype(int)
        avg_df['t2'] = avg_df['t2'].astype(int)
        avg_df['t3'] = avg_df['t3'].astype(int)
        avg_df['loop'] = avg_df['loop'].astype(int)
        avg_df['accuracy'] = avg_df['accuracy'] * 100
        # Divide cost by 1000, because cpt is in 1000 tokens; also divide by the number of questions
        avg_df['cost'] = avg_df['cost'] / (1000*val_numbers_len)
        
        avg_df.to_csv(output_file_name_val, index=False)

if __name__ == "__main__":
    main()