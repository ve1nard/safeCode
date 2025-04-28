import numpy as np
import multiprocessing
import os
import argparse
import json

MODEL_FAMILIES = {
    "wizardCoder": [
        "vanillaOVO/WizardCoder-Python-7B-V1.0"
        "WizardLMTeam/WizardCoder-Python-13B-V1.0"
        "WizardLMTeam/WizardCoder-Python-34B-V1.0"
    ],
}

# In one experiment, each model generated 10 solutions and 10 test sets,
# so that we can assess its pass@k security rate for k=[1-10]
MAX_PASS_K = 10

# The parameters are explored as combinations from two sets:
# ans_count_set = [1,3,5,10] and test_counts_set = [2, 4]
# For k=0, there is only one solution per problem. According to the design 
# of the pipeline, the solution is accepted without testing. However, in
# the process of hyperparameter exploration, for the sake of computing the
# accuracy of the model, we test the solution in the same way as for other
# of k 
ans_count_set = [1, 3, 5, 10]
test_count_set = [2, 4]

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--family', type=str, default='wizardCoder')
    parser.add_argument('--num_loops', type=int, default=10)
    parser.add_argument('--outputs', type=str, default='../../data/cascading/outputs')
    return parser.parse_args()

def find_max_product(matrix):
    max_product = 0
    max_indices = (-1, -1)
    max_answer_num = 0
    max_test_num = 0
    row_sums = np.sum(matrix, axis=1)
    col_sums = np.sum(matrix, axis=0)
    for a in range(matrix.shape[0]):
        for t in range(matrix.shape[1]):
            if matrix[a][t] == 1:
                product = row_sums[a] * col_sums[t]
                if product > max_product:
                    max_product = product
                    max_indices = (a, t)
                    max_answer_num = row_sums[a]
                    max_test_num = col_sums[t]
    return max_product, max_indices, max_answer_num, max_test_num

def main():
    args = get_args()
    model_names = [checkpoint.split('/')[-1] for checkpoint in MODEL_FAMILIES[args.family]]
    if not os.path.exists(f"../data/selected"):
        os.mkdir(f"../data/selected")
    
    # Iterate over the space of parameters for each model and each question
    for model_name in model_names:
        if not os.path.exists(f"../data/selected/{model_name}"):
            os.mkdir(f"../data/selected/{model_name}")

        # For k=0, for each problem there is only one solution generated
        greedy_answers_file = os.path.join(args.outputs, f"completed_code/{model_name}/greedy_completion_{model_name}.json")
        greedy_testcases_file = os.path.join(args.outputs, f"tests/{model_name}/greedy_tests_{model_name}.json")
        greedy_selected_file = f"../data/selected/{model_name}/{model_name}_greedy.json"
        greedy_all_selected = []
        with open(greedy_answers_file, 'r') as f:
            greedy_answer_data = json.load(f)
        with open(greedy_testcases_file, 'r') as f:
            greedy_testcase_data = json.load(f)
        for number in range(79):                        
            # Get the answer for this question  
            answer = None
            for answer_dict in greedy_answer_data:
                if answer_dict["problem"]==number:
                    answer = answer_dict["answer"]
                    total_length_in_tokens = answer_dict["answer_length_in_tokens"]
                    break     
            if not answer:
                selected_dict = {
                    "problem": number,
                    "max_answer_num": 0,
                    "max_test_num": 0,
                    "total_product": 0,
                    "answer": None,
                    "test": "",
                    "total_length_in_tokens": 0,
                    "valid": False
                }
                greedy_all_selected.append(selected_dict)
                continue
            
            # Collect all tests for this question
            all_greedy_generated_tests = []
            for testcase_dict in greedy_testcase_data:
                if testcase_dict["problem"] == number:
                    this_test = testcase_dict["test"]
                    testlines = this_test.split("\n")
                    for i in range(len(testlines)):
                        if testlines[i].startswith("assert"):
                            all_greedy_generated_tests.append(testlines[i])

            # Check the correctness for each answer-testline
            correct_stats = np.zeros([1, len(all_greedy_generated_tests)], np.int32)
            def code_to_run(t, answer, generated_test, result_queue):
                full_code = answer + "\n" + generated_test
                try:
                    exec(full_code, globals())
                    result_queue.put((t, True, 0))
                except Exception as e:
                    result_queue.put((t, False, e))
            processes = []
            result_queue = multiprocessing.Queue()
            for t in range(len(all_greedy_generated_tests)):
                generated_test = all_greedy_generated_tests[t]
                process = multiprocessing.Process(target=code_to_run, args=(t, answer, generated_test, result_queue))
                process.start()
                processes.append(process)
            for process in processes:
                process.join(1)  # Kill infinite loops in 1 second
                if process.is_alive():
                    process.terminate()
                    process.join()
            while not result_queue.empty():
                t, correct, excp = result_queue.get()
                if correct:
                    correct_stats[0][t] += 1
                else:
                    print(f"-----------------\nFailed {number}:\n\tanswer:\n\t\t{answer}\n\ttest:\n\t\t{all_greedy_generated_tests[t]}\nexception:\n{excp}\n")
            for process in processes:
                process.close()
            
            max_product, indices, max_answer_num, max_test_num = find_max_product(correct_stats)
            selected_greedy_answer = answer if indices[0] != -1 else None
            selected_greedy_test = all_greedy_generated_tests[indices[1]] if indices[1] != -1 else None
            valid = True if indices[0] != -1 else None

            selected_greedy_dict = {
                "problem": number,
                "max_answer_num": int(max_answer_num),
                "max_test_num": int(max_test_num),
                "max_product": int(max_product),
                "total_product": int(len(all_greedy_generated_tests)),
                "answer": selected_greedy_answer,
                "test": selected_greedy_test,
                "total_length_in_tokens": total_length_in_tokens,
                "valid": valid,
            }
            greedy_all_selected.append(selected_greedy_dict)
        with open(greedy_selected_file, 'w', encoding="utf-8") as f:
            json.dump(greedy_all_selected, f, indent=4)

        # Solution selection for other combinations of hyperparameters
        for ans_count in ans_count_set:
            for test_count in test_count_set:
                for loop in range(args.num_loops):
                    answer_file = os.path.join(args.outputs, f"completed_code/{model_name}/completion_{model_name}_{loop}.json")
                    testcase_file = os.path.join(args.outputs, f"tests/{model_name}/tests_{model_name}_{loop}.json")
                    selected_file = f"../data/selected/{model_name}/{model_name}_p{ans_count}_t{test_count}_l{loop}.json"
                    all_selected = []
                    
                    # Load the answer and testcase files
                    with open(answer_file, 'r') as f:
                        answer_data = json.load(f)
                    with open(testcase_file, 'r') as f:
                        testcase_data = json.load(f)

                    for number in range(79):  
                        total_length_in_tokens = 0                      
                        # Collect all answers for this question
                        all_answers = []
                        remain_ans_count = max(ans_count, 1)   
                        for answer_dict in answer_data:
                            if answer_dict["problem"]==number and remain_ans_count>0:
                                answer = answer_dict["answer"]
                                total_length_in_tokens += answer_dict["answer_length_in_tokens"]
                                all_answers.append(answer)
                                remain_ans_count -= 1

                        if not all_answers:
                            selected_dict = {
                                "number": number,
                                "max_answer_num": 0,
                                "max_test_num": 0,
                                "total_product": 0,
                                "answer": None,
                                "test": "",
                                "valid": False
                            }
                            all_selected.append(selected_dict)
                            continue
                        
                        # Collect all tests for this question
                        all_generated_tests = []
                        remain_test_count =  max(ans_count, 1) 
                        for testcase_dict in testcase_data:
                            if testcase_dict["problem"] == number and remain_test_count>0:
                                this_test = testcase_dict["test"]
                                testlines = this_test.split("\n")
                                for j in range(min(test_count, len(testlines))):
                                    if testlines[j].startswith("assert"):
                                        all_generated_tests.append(testlines[j])
                                total_length_in_tokens += testcase_dict[f"{test_count}_tests_length_in_tokens"]
                                remain_test_count -= 1

                        # Check the correctness for each answer-testline
                        correct_stats = np.zeros([len(all_answers),len(all_generated_tests)], np.int32)
                        def code_to_run(a, t, answer, generated_test, result_queue):
                            full_code = answer + "\n" + generated_test
                            try:
                                exec(full_code, globals())
                                result_queue.put((a, t, True, 0))
                            except Exception as e:
                                result_queue.put((a, t, False, e))
                        processes = []
                        result_queue = multiprocessing.Queue()
                        for a in range(len(all_answers)):
                            for t in range(len(all_generated_tests)):
                                answer = all_answers[a]
                                generated_test = all_generated_tests[t]
                                process = multiprocessing.Process(target=code_to_run, args=(a, t, answer, generated_test, result_queue))
                                process.start()
                                processes.append(process)
                        for process in processes:
                            process.join(1)  # Kill infinite loops in 1 second
                            if process.is_alive():
                                process.terminate()
                                process.join()
                        while not result_queue.empty():
                            a, t, correct, excp = result_queue.get()
                            if correct:
                                correct_stats[a][t] += 1
                            else:
                                print(f"-----------------\nFailed {number}:\n\tanswer:\n\t\t{all_answers[a]}\n\ttest:\n\t\t{all_generated_tests[t]}\nexception:\n{excp}\n")
                        for process in processes:
                            process.close()
                        
                        max_product, indices, max_answer_num, max_test_num = find_max_product(correct_stats)
                        selected_answer = all_answers[indices[0]] if indices[0] != -1 else None
                        selected_test = all_generated_tests[indices[1]] if indices[1] != -1 else None

                        if not selected_answer:
                            valid = False
                        else:
                            # After selecting the best test-answer pair, we perform the final check to see
                            # if the solution can be accepted by running selected answer on selected test
                            full_code = selected_answer + "\n" + selected_test    
                            def code_to_run(result_queue):
                                try:
                                    exec(full_code, globals())
                                    result_queue.put(True)
                                except Exception as e:
                                    result_queue.put(False)
                            result_queue = multiprocessing.Queue()
                            process = multiprocessing.Process(target=code_to_run, args=(result_queue,))
                            process.start()
                            process.join(1)
                            if process.is_alive():
                                # print("Code took too long to run!")
                                process.terminate()
                                process.join()  # Ensure termination
                                valid = False
                            else:
                                valid = result_queue.get()
                            process.close()

                        selected_dict = {
                            "problem": number,
                            "max_answer_num": int(max_answer_num),
                            "max_test_num": int(max_test_num),
                            "max_product": int(max_product),
                            "total_product": int(len(all_answers)*len(all_generated_tests)),
                            "answer": selected_answer,
                            "test": selected_test,
                            "total_length_in_tokens": total_length_in_tokens,
                            "valid": valid
                        }
                        all_selected.append(selected_dict)

                    # Write to file
                    with open(selected_file, 'w', encoding="utf-8") as f:
                        json.dump(all_selected, f, indent=4)
        

if __name__ == "__main__":
    main()