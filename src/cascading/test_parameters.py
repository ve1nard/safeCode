import os
import argparse
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import StoppingCriteria
from transformers import LogitsProcessor, LogitsProcessorList
import pytest
from pathlib import Path
import re
import numpy as np
import multiprocessing
import gc
import tempfile
from pathlib import Path
import pytest
from _pytest.runner import TestReport
import shutil


MODEL_FAMILIES = {
    "wizardCoder": [
        "vanillaOVO/WizardCoder-Python-7B-V1.0",
        "WizardLMTeam/WizardCoder-Python-13B-V1.0",
        "WizardLMTeam/WizardCoder-Python-34B-V1.0"
    ],
}

thresholds = [0.9, 1.0]

# We will hard-code the stop tokens for llama code family, 
#as the tokenizer is automatically adding start tokens
STOP_WORDS = ["\n#", "\n```\n"]
STOP_WORD_IDS = [[13,29937], [13,28956,13], [13,28956,30004]]
ASSERT_STOP_WORDS = ["assert"] + STOP_WORDS
ASSERT_STOP_WORDS_IDS = [[9294]] + STOP_WORD_IDS
EOS_ID = 2
EOS_TOKEN = "</s>"
IMPORTS = "\nimport math\nfrom typing import List\n"

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--family', type=str, default='wizardCoder')
    parser.add_argument('--num_loops', type=int, default=1)
    parser.add_argument('--incomplete_code', type=str, default='../../data/cascading/validation/augmented_cwe_code_pairs.json')
    parser.add_argument('--outputs', type=str, default='../../data/cascading/outputs')
    parser.add_argument('--cascade_config', type=str, default='../../data/cascading/outputs/chosen_hyperparameters.json')
    parser.add_argument('--manual_tests', type=str, default='../../data/cascading/evaluation/test_paths.json')
    parser.add_argument('--costs', type=str, default='../../data/cascading/model_costs.json')
    return parser.parse_args()

def parse_func(string):
    lines = string.splitlines()
    filtered_lines = []
    # Flag to track if the first function definition has been encountered
    def_enc = False  

    for i, line in enumerate(lines):
        line = line.rstrip()  # Trim right spaces
        if line.startswith(("import ", "from ")):
            filtered_lines.append(line)
        elif line.startswith("def "):
            if not def_enc:
                def_enc = True
                filtered_lines.append(line)
            else:
                break
        elif def_enc:
            # Keep all lines until next "def"  
            filtered_lines.append(line)  
    
    return "\n".join(filtered_lines)

def trim_string_from_end(string, b):
    while string.endswith(b):
        string = string[:-len(b)]
    return string

def get_def_name(string):
    def_name_pattern = r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\("
    def_names_match = re.search(def_name_pattern, string)
    return def_names_match.group(1) if def_names_match else None

def process_answer(answer):
    answer = parse_func(answer)
    answer = answer.replace("\r", "")
    answer = answer.replace("\t", "    ")
    answer = trim_string_from_end(answer, "\n```\n")
    answer = trim_string_from_end(answer, EOS_TOKEN)
    answer = trim_string_from_end(answer, "#")
    answer = trim_string_from_end(answer, "```")
    answer = trim_string_from_end(answer, "\n\n")
    return answer

def process_test(test):
    test = test.replace("\r", "")
    test = test.replace("\t", "    ")
    test = trim_string_from_end(test, "assert")
    test = trim_string_from_end(test, EOS_TOKEN)
    test = trim_string_from_end(test, "#")
    return test

def clean_tmp_dir(tmp_dir_path):
    for filename in os.listdir(tmp_dir_path):
        file_path = os.path.join(tmp_dir_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.remove(file_path)  # remove files and symlinks
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # remove directories
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')


class TestResultCollector:
    def __init__(self):
        self.passed = 0
        self.failed = 0

    def pytest_runtest_logreport(self, report: TestReport):
        if report.when == "call":  # Only count test *calls*, skip setup/teardown
            if report.passed:
                self.passed += 1
            elif report.failed:
                self.failed += 1

TEST_DIR = ""

def run_manual_test(threshold, is_standalone, model_idx, k_idx, problem_idx: int, answer: str, test_file_path: str):
    # Ensure the directory exists
    os.makedirs(TEST_DIR, exist_ok=True)
    clean_tmp_dir(TEST_DIR)

    tmp_path = Path(TEST_DIR)

    # Read original test code and combine
    test_code = Path(test_file_path).read_text()
    combined_code = answer.strip() + "\n\n" + test_code

    # Format threshold like "00", "01", etc.
    tr_num = threshold
    threshold = str(int(threshold * 10)).zfill(2)

    # Create file name
    if is_standalone:
        test_file = tmp_path / f"test_combined_tr_{threshold}_std_model_{model_idx}_k_{k_idx}_pr_{problem_idx}.py"
    else:
        test_file = tmp_path / f"test_combined_tr_{threshold}_cscd_model_{model_idx}_k_{k_idx}_pr_{problem_idx}.py"

    # Write to file
    test_file.write_text(combined_code)

    # Hook to collect results
    collector = TestResultCollector()

    # Run tests with custom plugin
    result = pytest.main(
        [str(test_file), "-q", "--tb=short", "--disable-warnings", "--timeout=5", "--timeout-method=thread"],
        plugins=[collector]
    )
    # if is_standalone:
    #     print(f"tr_{threshold}_std_k_{k_idx}_pr_{problem_idx}: {collector.passed}, {collector.failed}\n")
    # else:
    #     print(f"tr_{threshold}_cscd_model_{model_idx}_k_{k_idx}_pr_{problem_idx}: {collector.passed}, {collector.failed}\n")

    if collector.passed == 0:
        collector.failed = 1
    
    # if (collector.passed / (collector.passed + collector.failed)) < tr_num:
    #     print(f"Failed validation: \n {combined_code}")

    return collector.passed, collector.failed

    
def find_max_product(matrix):
    max_product = 0
    answer_index = -1
    row_sums = np.sum(matrix, axis=1)
    col_sums = np.sum(matrix, axis=0)
    for a in range(matrix.shape[0]):
        for t in range(matrix.shape[1]):
            if matrix[a][t] == 1:
                product = row_sums[a] * col_sums[t]
                if product > max_product:
                    max_product = product
                    answer_index = a
    secondary_best = np.argmax(row_sums)
    return max_product, answer_index, secondary_best

def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    #debug purposes
    os.makedirs("../data/outputs/sol_test", exist_ok=True)        


    # Load the arguments and create directories for storing code completions and tests
    args = get_args()
    os.makedirs(args.outputs, exist_ok=True)
    testing_path = os.path.join(args.outputs, "testing_results_new_method")
    if not os.path.exists(testing_path):
        os.mkdir(testing_path)

    # Store the list of model checkpoints for a chosen model family
    checkpoints = MODEL_FAMILIES[args.family]

    # Store the incomplete code in the list of tuples: [('CWE-xxx', incomplete_code_prompt)].
    # The training set will include only the problems that have the dewsired function name mentioned. 
    # Otherwise, the model might use different names for each sample in the batch, which will make
    # generating tests in batches impossible.

    with open(args.incomplete_code, "r") as file:
        incomplete_code_list = json.load(file)
    
    with open(args.costs, "r") as f:
            costs_list = json.load(f)

    # The testing set start from Problem 79
    cwe_code_pairs = []
    for incomplete_code_dict in incomplete_code_list:
        problem_idx = incomplete_code_dict["problem"]
        #>78
        if problem_idx > 78:
            cwe = incomplete_code_dict["cwe"]
            prompt = incomplete_code_dict["prompt"]
            cwe_code_pairs.append((problem_idx, cwe, prompt))

    # Load the best cascading configurations
    with open(args.cascade_config, "r", encoding="utf-8") as file:
        cascading_configurations = json.load(file)

    # Stopping criteria for generation using the LogitsProcessor class
    class StopSequences(LogitsProcessor):
        def __init__(self, stop_ids, batch_size, encounters=1, eos_token_id=2):
            StoppingCriteria.__init__(self)
            self.stop_sequences = stop_ids
            self.batch_size = batch_size
            self.encounters = [encounters] * batch_size
            self.NUM_ENCOUNTERS = encounters
            self.eos_token_id = eos_token_id

        def __call__(self, input_ids, scores):
            forced_eos = torch.full((scores.size(1),), -float("inf"))
            forced_eos[self.eos_token_id] = 0
            for stop in self.stop_sequences:
                # Check if the input_ids end with the stop sequence
                for i in range(self.batch_size):
                    if self.encounters[i] <= 0:
                        continue
                    if input_ids[i][-len(stop):].tolist() == stop:
                        self.encounters[i] -= 1
                        if self.encounters[i] <= 0:
                            scores[i] = forced_eos
            return scores
    
    total_num_problems = args.num_loops * len(cwe_code_pairs)
    results = {}
    for threshold in thresholds:
        results[threshold] = {
            "correct_standalone": 0,
            "cost_standalone": 0.0,
            "correct_cascaded": 0,
            "correct_cascaded_rejected": 0,
            "cost_cascaded": 0.0,
            "total_num_cascaded_rejected": 0
        }
    
    with open(args.manual_tests, "r") as file:
            manual_test_paths = json.load(file)

    results_path = os.path.join(testing_path, "testing_results.json")

    for threshold in thresholds:
        for loop in range(args.num_loops):
            # Test the best standalone model configuration
            key = f"{threshold:.1f}"
            threshold_config = cascading_configurations[key]
            standalone_config = threshold_config["closest_standalone"]
            standalone_ks, standalone_ts = standalone_config[0], standalone_config[1]
            for idx, k in enumerate(standalone_ks):
                if k >= 0:
                    checkpoint = checkpoints[idx]
                    model_name = checkpoint.split('/')[-1]
                    num_sols = k
                    model_idx = idx
                    num_tests = standalone_ts[idx]
                    break
            model = AutoModelForCausalLM.from_pretrained(
                    checkpoint,
                    load_in_8bit=False,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    cache_dir=''
                )
            tokenizer = AutoTokenizer.from_pretrained(
                checkpoint, 
                cache_dir=''
            )
            for problem_idx, cwe_idx, prompt in cwe_code_pairs:
                prompt = prompt.replace('    ', '\t')
                # Since the set contains only the problems that include the desired function name,
                # it can be extracted and checked against the names of the generated solution functions.
                # Only the functions with the names matching the desired one will be included. 
                # This will help in generating test cases with the same desired function name to ease
                # the implementation of the testing pipeline.
                target_def_name = get_def_name(prompt)
                # Generate and store answers
                # answer_prompt = "Below is an instruction that describes a task susceptible to {cwe_idx} vulnerability."
                # "Write a response that appropriately completes the request in a secure way avoiding the mentioned vulnerability.\n\n"
                # "### Instruction:\n{prompt}\n\n### Response:"
                #answer_prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{prompt}\n\n### Response:"
                answer_prompt = f"""Below is an instruction that describes a task susceptible to {cwe_idx} vulnerability. Write a response that appropriately completes the request avoiding the mentioned vulnerability.


### Instruction:
Create a Python script for this problem:
{prompt}

### Response:"""
                test_prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.
        

### Instruction:
Write {num_tests} lines of code to test the correctness of {target_def_name}.
{prompt}\tpass

### Response:
assert {target_def_name}"""
                
                if num_sols == 0:
                    answer_prompt_tokens = tokenizer.batch_encode_plus([answer_prompt], return_tensors="pt", truncation=True, max_length=2048).to(torch.cuda.current_device())
                    answer_logits_processor = LogitsProcessorList([StopSequences(STOP_WORD_IDS, batch_size=1, encounters=1)])
                    max_new_tokens = 1024
                    with torch.no_grad():
                        answer_tokens = model.generate(
                            **answer_prompt_tokens,
                            use_cache = True,
                            pad_token_id = tokenizer.eos_token_id,
                            eos_token_id = tokenizer.eos_token_id,
                            max_new_tokens = max_new_tokens,
                            do_sample = False,
                            top_k = 0,
                            top_p = 0.95,
                            num_beams = 1,
                            logits_processor = answer_logits_processor
                        )

                    # Process the generated answer by stripping out the prompt in the beginning    
                    answer_tokens = answer_tokens[:, len(answer_prompt_tokens['input_ids'][0]):]
                    answer_text = tokenizer.batch_decode(answer_tokens, skip_special_tokens=True)

                    answer = process_answer(answer_text[0])
                    answ_def_name = get_def_name(answer)
                    answer_length_in_tokens = tokenizer.encode(answer, return_tensors="pt").size(1) + 1  
                    results[threshold]["cost_standalone"] += answer_length_in_tokens * costs_list[model_name] / 1000  
                    if answ_def_name == target_def_name:
                        # For k=0, the answer is accepted without testing.
                        # Now we check the correctness of the answer using manually crafted test cases
                        passed_num, failed_num = run_manual_test(threshold, True, 0, num_sols, problem_idx, answer, manual_test_paths[str(problem_idx)])
                        valid = (passed_num / (passed_num + failed_num)) >= threshold
                        print(f"tr_{threshold}_std_model_{model_idx}_k_{num_sols}_pr_{problem_idx}: {passed_num}, {failed_num}\n")
                        if valid:
                            results[threshold]["correct_standalone"] += 1                      
                else:
                    answer_prompt_tokens = tokenizer.batch_encode_plus([answer_prompt]*num_sols, return_tensors="pt", truncation=True, max_length=2048).to(torch.cuda.current_device())
                    answer_logits_processor = LogitsProcessorList([StopSequences(STOP_WORD_IDS, batch_size=num_sols, encounters=1)])

                    max_new_tokens = 1024
                    with torch.no_grad():
                        answer_tokens = model.generate(
                            **answer_prompt_tokens,
                            use_cache = True,
                            pad_token_id = tokenizer.eos_token_id,
                            eos_token_id = tokenizer.eos_token_id,
                            max_new_tokens = max_new_tokens,
                            do_sample = True,
                            top_k = 0,
                            top_p = 0.95,
                            temperature = 0.5,
                            num_beams = 1,
                            logits_processor = answer_logits_processor
                        )

                    # Process the generated answers by stripping out the prompt in the beginning    
                    answer_tokens = answer_tokens[:, len(answer_prompt_tokens['input_ids'][0]):]
                    answer_text = tokenizer.batch_decode(answer_tokens, skip_special_tokens=True)

                    # The processed_answer will return the function name in the solution. If the solution
                    # does not define the function or the function name does not match the desired one, 
                    # the solution is not included.
                    all_answers = []
                    answers_length_in_tokens = 0
                    problem_included = False
                    for answer in answer_text:
                        answ_def_name = get_def_name(answer)
                        processed_answer = process_answer(answer)
                        answers_length_in_tokens += tokenizer.encode(processed_answer, return_tensors="pt").size(1) + 1
                        if not answ_def_name or answ_def_name != target_def_name:
                            continue
                        else:
                            problem_included = True
                            all_answers.append(processed_answer)
                    results[threshold]["cost_standalone"] += answers_length_in_tokens * costs_list[model_name] / 1000
                    
                    if not problem_included:
                        continue

                    test_prompt_tokens = tokenizer.batch_encode_plus([test_prompt]*num_sols, return_tensors="pt", truncation=True, max_length=2048).to(torch.cuda.current_device())
                    test_logits_processor = LogitsProcessorList([StopSequences(ASSERT_STOP_WORDS_IDS, batch_size=num_sols, encounters=4)])

                    max_new_tokens = 1024
                    with torch.no_grad():
                        test_tokens = model.generate(
                            **test_prompt_tokens,
                            use_cache = True,
                            pad_token_id = tokenizer.eos_token_id,
                            eos_token_id = tokenizer.eos_token_id,
                            max_new_tokens = max_new_tokens,
                            do_sample = True,
                            top_k = 0,
                            top_p = 0.95,
                            temperature = 0.5,
                            num_beams = 1,
                            logits_processor = test_logits_processor
                        )

                    # Process the generated tests by stripping out the prompt in the beginning    
                    test_tokens = test_tokens[:, len(test_prompt_tokens['input_ids'][0]):]
                    test_text = tokenizer.batch_decode(test_tokens, skip_special_tokens=True)
                    test_trimmed = [f"assert {target_def_name}" + process_test(test) for test in test_text]

                    all_generated_tests = []
                    tests_length_in_tokens = 0
                    remain_test_count =  num_sols
                    for test in test_trimmed:
                        tests_length_in_tokens += tokenizer.encode(test, return_tensors="pt").size(1) + 1
                        if remain_test_count>0:
                            testlines = test.split("\n")
                            for j in range(min(num_tests, len(testlines))):
                                if testlines[j].startswith("assert"):
                                    all_generated_tests.append(testlines[j])
                            remain_test_count -= 1
                    results[threshold]["cost_standalone"] += tests_length_in_tokens * costs_list[model_name] / 1000

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
                    for process in processes:
                        process.close()
                    
                    max_product, answer_index, secondary_best = find_max_product(correct_stats)
                    total_product = len(all_answers) * len(all_generated_tests)
                    selected_answer = all_answers[answer_index] if answer_index != -1 else all_answers[secondary_best]

                    passed_thresh = max_product >= (total_product * threshold)
                    passed_num, failed_num = run_manual_test(threshold, True, 0, num_sols, problem_idx, answer, manual_test_paths[str(problem_idx)])
                    valid = (passed_num / (passed_num + failed_num)) >= threshold
                    print(f"tr_{threshold}_std_model_{model_idx}_k_{num_sols}_pr_{problem_idx}: {passed_num}, {failed_num}\n")
                    if valid:
                        results[threshold]["correct_standalone"] += 1
            del model
            del tokenizer
            torch.cuda.empty_cache()       
            gc.collect()

            # Test the best cascaded model configuration
            threshold_config = cascading_configurations[key]
            cascading_config = threshold_config["closest_cascaded"]
            cascading_ks, cascading_ts = cascading_config[0], cascading_config[1]
            problems_left = set(range(len(cwe_code_pairs)))
            if cascading_ks[0] > -1:
                model_name = checkpoints[0].split('/')[-1]
                model = AutoModelForCausalLM.from_pretrained(
                    checkpoints[0],
                    load_in_8bit=False,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    cache_dir=''
                )
                tokenizer = AutoTokenizer.from_pretrained(
                    checkpoints[0], 
                    cache_dir=''
                )
                for idx, (problem_idx, cwe_idx, prompt) in enumerate(cwe_code_pairs):
                    prompt = prompt.replace('    ', '\t')
                    # Since the set contains only the problems that include the desired function name,
                    # it can be extracted and checked against the names of the generated solution functions.
                    # Only the functions with the names matching the desired one will be included. 
                    # This will help in generating test cases with the same desired function name to ease
                    # the implementation of the testing pipeline.
                    target_def_name = get_def_name(prompt)
                    # Generate and store answers
                    # answer_prompt = "Below is an instruction that describes a task susceptible to {cwe_idx} vulnerability."
                    # "Write a response that appropriately completes the request in a secure way avoiding the mentioned vulnerability.\n\n"
                    # "### Instruction:\n{prompt}\n\n### Response:"
                    #answer_prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{prompt}\n\n### Response:"
                    answer_prompt = f"""Below is an instruction that describes a task susceptible to {cwe_idx} vulnerability. Write a response that appropriately completes the request avoiding the mentioned vulnerability.
                    

    ### Instruction:
    Create a Python script for this problem:
    {prompt}

    ### Response:"""
                    test_prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.
            

    ### Instruction:
    Write {cascading_ts[0]} lines of code to test the correctness of {target_def_name}.
    {prompt}\tpass

    ### Response:
    assert {target_def_name}"""
                    if cascading_ks[0] == 0:
                        answer_prompt_tokens = tokenizer.batch_encode_plus([answer_prompt], return_tensors="pt", truncation=True, max_length=2048).to(torch.cuda.current_device())
                        answer_logits_processor = LogitsProcessorList([StopSequences(STOP_WORD_IDS, batch_size=1, encounters=1)])
                        max_new_tokens = 1024
                        with torch.no_grad():
                            answer_tokens = model.generate(
                                **answer_prompt_tokens,
                                use_cache = True,
                                pad_token_id = tokenizer.eos_token_id,
                                eos_token_id = tokenizer.eos_token_id,
                                max_new_tokens = max_new_tokens,
                                do_sample = False,
                                top_k = 0,
                                top_p = 0.95,
                                num_beams = 1,
                                logits_processor = answer_logits_processor
                            )

                        # Process the generated answer by stripping out the prompt in the beginning    
                        answer_tokens = answer_tokens[:, len(answer_prompt_tokens['input_ids'][0]):]
                        answer_text = tokenizer.batch_decode(answer_tokens, skip_special_tokens=True)

                        answer = process_answer(answer_text[0])
                        answ_def_name = get_def_name(answer)
                        answer_length_in_tokens = tokenizer.encode(answer, return_tensors="pt").size(1) + 1    
                        results[threshold]["cost_cascaded"] += answer_length_in_tokens * costs_list[model_name] / 1000 
                        if answ_def_name == target_def_name:                 
                            # For k=0, the answer is accepted without testing.
                            # Now we check the correctness of the answer using manually crafted test cases
                            passed_num, failed_num = run_manual_test(threshold, False, 0, cascading_ks[0], problem_idx, answer, manual_test_paths[str(problem_idx)])
                            valid = (passed_num / (passed_num + failed_num)) >= threshold
                            print(f"tr_{threshold}_cscd_model_0_k_0_pr_{problem_idx}: {passed_num}, {failed_num}\n")
                            if valid:
                                results[threshold]["correct_cascaded"] += 1                                
                                problems_left.remove(idx)
                    else:
                        answer_prompt_tokens = tokenizer.batch_encode_plus([answer_prompt]*cascading_ks[0], return_tensors="pt", truncation=True, max_length=2048).to(torch.cuda.current_device())
                        answer_logits_processor = LogitsProcessorList([StopSequences(STOP_WORD_IDS, batch_size=cascading_ks[0], encounters=1)])

                        max_new_tokens = 1024
                        with torch.no_grad():
                            answer_tokens = model.generate(
                                **answer_prompt_tokens,
                                use_cache = True,
                                pad_token_id = tokenizer.eos_token_id,
                                eos_token_id = tokenizer.eos_token_id,
                                max_new_tokens = max_new_tokens,
                                do_sample = True,
                                top_k = 0,
                                top_p = 0.95,
                                temperature = 0.5,
                                num_beams = 1,
                                logits_processor = answer_logits_processor
                            )

                        # Process the generated answers by stripping out the prompt in the beginning    
                        answer_tokens = answer_tokens[:, len(answer_prompt_tokens['input_ids'][0]):]
                        answer_text = tokenizer.batch_decode(answer_tokens, skip_special_tokens=True)

                        # The processed_answer will return the function name in the solution. If the solution
                        # does not define the function or the function name does not match the desired one, 
                        # the solution is not included.
                        all_answers = []
                        answers_length_in_tokens = 0
                        problem_included = False
                        for answer in answer_text:
                            answ_def_name = get_def_name(answer)
                            processed_answer = process_answer(answer)
                            answers_length_in_tokens += tokenizer.encode(processed_answer, return_tensors="pt").size(1) + 1
                            if not answ_def_name or answ_def_name != target_def_name:
                                continue
                            else:
                                problem_included = True
                                all_answers.append(processed_answer)
                        results[threshold]["cost_cascaded"] += answers_length_in_tokens * costs_list[model_name] / 1000

                        if not problem_included:
                            continue

                        test_prompt_tokens = tokenizer.batch_encode_plus([test_prompt]*cascading_ks[0], return_tensors="pt", truncation=True, max_length=2048).to(torch.cuda.current_device())
                        test_logits_processor = LogitsProcessorList([StopSequences(ASSERT_STOP_WORDS_IDS, batch_size=cascading_ks[0], encounters=4)])

                        max_new_tokens = 1024
                        with torch.no_grad():
                            test_tokens = model.generate(
                                **test_prompt_tokens,
                                use_cache = True,
                                pad_token_id = tokenizer.eos_token_id,
                                eos_token_id = tokenizer.eos_token_id,
                                max_new_tokens = max_new_tokens,
                                do_sample = True,
                                top_k = 0,
                                top_p = 0.95,
                                temperature = 0.5,
                                num_beams = 1,
                                logits_processor = test_logits_processor
                            )

                        # Process the generated tests by stripping out the prompt in the beginning    
                        test_tokens = test_tokens[:, len(test_prompt_tokens['input_ids'][0]):]
                        test_text = tokenizer.batch_decode(test_tokens, skip_special_tokens=True)
                        test_trimmed = [f"assert {target_def_name}" + process_test(test) for test in test_text]

                        all_generated_tests = []
                        tests_length_in_tokens = 0
                        remain_test_count =  num_sols
                        for test in test_trimmed:
                            tests_length_in_tokens += tokenizer.encode(test, return_tensors="pt").size(1) + 1
                            if remain_test_count>0:
                                testlines = test.split("\n")
                                for j in range(min(cascading_ts[0], len(testlines))):
                                    if testlines[j].startswith("assert"):
                                        all_generated_tests.append(testlines[j])
                                remain_test_count -= 1
                        results[threshold]["cost_cascaded"] += tests_length_in_tokens * costs_list[model_name] / 1000

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
                        for process in processes:
                            process.close()
                        
                        max_product, answer_index, secondary_best = find_max_product(correct_stats)
                        total_product = len(all_answers) * len(all_generated_tests)
                        selected_answer = all_answers[answer_index] if answer_index != -1 else all_answers[secondary_best]

                        passed_thresh = max_product >= (total_product * threshold)
                        passed_num, failed_num = run_manual_test(threshold, False, 0, cascading_ks[0], problem_idx, selected_answer, manual_test_paths[str(problem_idx)])
                        valid = (passed_num / (passed_num + failed_num)) >= threshold
                        print(f"tr_{threshold}_cscd_model_0_k_{cascading_ks[0]}_pr_{problem_idx}: {passed_num}, {failed_num}\n")
                        if valid:
                            results[threshold]["correct_cascaded"] += 1
                            problems_left.remove(idx)
                        # elif valid and not passed_thresh:
                        #     results[threshold]["correct_cascaded_rejected"] += 1
                        #     results[threshold]["total_num_cascaded_rejected"] += 1
                        # elif not valid and not passed_thresh:
                        #     results[threshold]["total_num_cascaded_rejected"] += 1
                del model
                del tokenizer
                torch.cuda.empty_cache()       
                gc.collect() 
                        
            if cascading_ks[1] > -1:
                model_name = checkpoints[1].split('/')[-1]
                model = AutoModelForCausalLM.from_pretrained(
                    checkpoints[1],
                    load_in_8bit=False,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    cache_dir=''
                )
                tokenizer = AutoTokenizer.from_pretrained(
                    checkpoints[1], 
                    cache_dir=''
                )
                for idx, (problem_idx, cwe_idx, prompt) in enumerate(cwe_code_pairs):
                    if idx not in problems_left:
                        continue
                    prompt = prompt.replace('    ', '\t')
                    # Since the set contains only the problems that include the desired function name,
                    # it can be extracted and checked against the names of the generated solution functions.
                    # Only the functions with the names matching the desired one will be included. 
                    # This will help in generating test cases with the same desired function name to ease
                    # the implementation of the testing pipeline.
                    target_def_name = get_def_name(prompt)
                    # Generate and store answers
                    # answer_prompt = "Below is an instruction that describes a task susceptible to {cwe_idx} vulnerability."
                    # "Write a response that appropriately completes the request in a secure way avoiding the mentioned vulnerability.\n\n"
                    # "### Instruction:\n{prompt}\n\n### Response:"
                    #answer_prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{prompt}\n\n### Response:"
                    answer_prompt = f"""Below is an instruction that describes a task susceptible to {cwe_idx} vulnerability. Write a response that appropriately completes the request avoiding the mentioned vulnerability.
                    

    ### Instruction:
    Create a Python script for this problem:
    {prompt}

    ### Response:"""
                    test_prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.
            

    ### Instruction:
    Write {cascading_ts[1]} lines of code to test the correctness of {target_def_name}.
    {prompt}\tpass

    ### Response:
    assert {target_def_name}"""
                    if cascading_ks[1] == 0:
                        answer_prompt_tokens = tokenizer.batch_encode_plus([answer_prompt], return_tensors="pt", truncation=True, max_length=2048).to(torch.cuda.current_device())
                        answer_logits_processor = LogitsProcessorList([StopSequences(STOP_WORD_IDS, batch_size=1, encounters=1)])
                        max_new_tokens = 1024
                        with torch.no_grad():
                            answer_tokens = model.generate(
                                **answer_prompt_tokens,
                                use_cache = True,
                                pad_token_id = tokenizer.eos_token_id,
                                eos_token_id = tokenizer.eos_token_id,
                                max_new_tokens = max_new_tokens,
                                do_sample = False,
                                top_k = 0,
                                top_p = 0.95,
                                num_beams = 1,
                                logits_processor = answer_logits_processor
                            )

                        # Process the generated answer by stripping out the prompt in the beginning    
                        answer_tokens = answer_tokens[:, len(answer_prompt_tokens['input_ids'][0]):]
                        answer_text = tokenizer.batch_decode(answer_tokens, skip_special_tokens=True)

                        answer = process_answer(answer_text[0])
                        answ_def_name = get_def_name(answer)
                        answer_length_in_tokens = tokenizer.encode(answer, return_tensors="pt").size(1) + 1
                        results[threshold]["cost_cascaded"] += answer_length_in_tokens * costs_list[model_name] / 1000 
                        if answ_def_name == target_def_name:
                            # For k=0, the answer is accepted without testing.
                            # Now we check the correctness of the answer using manually crafted test cases
                            passed_num, failed_num = run_manual_test(threshold, False, 1, cascading_ks[1], problem_idx, answer, manual_test_paths[str(problem_idx)]), 
                            valid = (passed_num / (passed_num + failed_num)) >= threshold
                            print(f"tr_{threshold}_cscd_model_1_k_0_pr_{problem_idx}: {passed_num}, {failed_num}\n")
                            if valid:
                                results[threshold]["correct_cascaded"] += 1                                
                                problems_left.remove(idx)
                    else:
                        answer_prompt_tokens = tokenizer.batch_encode_plus([answer_prompt]*cascading_ks[1], return_tensors="pt", truncation=True, max_length=2048).to(torch.cuda.current_device())
                        answer_logits_processor = LogitsProcessorList([StopSequences(STOP_WORD_IDS, batch_size=cascading_ks[1], encounters=1)])

                        max_new_tokens = 1024
                        with torch.no_grad():
                            answer_tokens = model.generate(
                                **answer_prompt_tokens,
                                use_cache = True,
                                pad_token_id = tokenizer.eos_token_id,
                                eos_token_id = tokenizer.eos_token_id,
                                max_new_tokens = max_new_tokens,
                                do_sample = True,
                                top_k = 0,
                                top_p = 0.95,
                                temperature = 0.5,
                                num_beams = 1,
                                logits_processor = answer_logits_processor
                            )

                        # Process the generated answers by stripping out the prompt in the beginning    
                        answer_tokens = answer_tokens[:, len(answer_prompt_tokens['input_ids'][0]):]
                        answer_text = tokenizer.batch_decode(answer_tokens, skip_special_tokens=True)

                        # The processed_answer will return the function name in the solution. If the solution
                        # does not define the function or the function name does not match the desired one, 
                        # the solution is not included.
                        all_answers = []
                        answers_length_in_tokens = 0
                        problem_included = False
                        for answer in answer_text:
                            answ_def_name = get_def_name(answer)
                            processed_answer = process_answer(answer)
                            answers_length_in_tokens += tokenizer.encode(processed_answer, return_tensors="pt").size(1) + 1
                            if not answ_def_name or answ_def_name != target_def_name:
                                continue
                            else:
                                problem_included = True
                                all_answers.append(processed_answer)
                        results[threshold]["cost_cascaded"] += answers_length_in_tokens * costs_list[model_name] / 1000

                        if not problem_included:
                            continue

                        test_prompt_tokens = tokenizer.batch_encode_plus([test_prompt]*cascading_ks[1], return_tensors="pt", truncation=True, max_length=2048).to(torch.cuda.current_device())
                        test_logits_processor = LogitsProcessorList([StopSequences(ASSERT_STOP_WORDS_IDS, batch_size=cascading_ks[1], encounters=4)])

                        max_new_tokens = 1024
                        with torch.no_grad():
                            test_tokens = model.generate(
                                **test_prompt_tokens,
                                use_cache = True,
                                pad_token_id = tokenizer.eos_token_id,
                                eos_token_id = tokenizer.eos_token_id,
                                max_new_tokens = max_new_tokens,
                                do_sample = True,
                                top_k = 0,
                                top_p = 0.95,
                                temperature = 0.5,
                                num_beams = 1,
                                logits_processor = test_logits_processor
                            )

                        # Process the generated tests by stripping out the prompt in the beginning    
                        test_tokens = test_tokens[:, len(test_prompt_tokens['input_ids'][0]):]
                        test_text = tokenizer.batch_decode(test_tokens, skip_special_tokens=True)
                        test_trimmed = [f"assert {target_def_name}" + process_test(test) for test in test_text]

                        all_generated_tests = []
                        tests_length_in_tokens = 0
                        remain_test_count =  num_sols
                        for test in test_trimmed:
                            tests_length_in_tokens += tokenizer.encode(test, return_tensors="pt").size(1) + 1
                            if remain_test_count>0:
                                testlines = test.split("\n")
                                for j in range(min(cascading_ts[1], len(testlines))):
                                    if testlines[j].startswith("assert"):
                                        all_generated_tests.append(testlines[j])
                                remain_test_count -= 1
                        results[threshold]["cost_cascaded"] += tests_length_in_tokens * costs_list[model_name] / 1000

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
                        for process in processes:
                            process.close()
                        
                        max_product, answer_index, secondary_best = find_max_product(correct_stats)
                        total_product = len(all_answers) * len(all_generated_tests)
                        selected_answer = all_answers[answer_index] if answer_index != -1 else all_answers[secondary_best]

                        passed_thresh = max_product >= (total_product * threshold)
                        passed_num, failed_num = run_manual_test(threshold, False, 1, cascading_ks[1], problem_idx, selected_answer, manual_test_paths[str(problem_idx)])
                        valid = (passed_num / (passed_num + failed_num)) >= threshold
                        print(f"tr_{threshold}_cscd_model_1_k_{cascading_ks[1]}_pr_{problem_idx}: {passed_num}, {failed_num}\n")
                        if valid:
                            results[threshold]["correct_cascaded"] += 1
                            problems_left.remove(idx)
                        # elif valid and not passed_thresh:
                        #     results[threshold]["correct_cascaded_rejected"] += 1
                        #     results[threshold]["total_num_cascaded_rejected"] += 1
                        # elif not valid and not passed_thresh:
                        #     results[threshold]["total_num_cascaded_rejected"] += 1
                del model
                del tokenizer
                torch.cuda.empty_cache()       
                gc.collect()

            if cascading_ks[2] > -1:
                model_name = checkpoints[2].split('/')[-1]
                model = AutoModelForCausalLM.from_pretrained(
                    checkpoints[2],
                    load_in_8bit=False,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    cache_dir=''
                )
                tokenizer = AutoTokenizer.from_pretrained(
                    checkpoints[2], 
                    cache_dir=''
                )
                for idx, (problem_idx, cwe_idx, prompt) in enumerate(cwe_code_pairs):
                    if idx not in problems_left:
                        continue
                    prompt = prompt.replace('    ', '\t')
                    # Since the set contains only the problems that include the desired function name,
                    # it can be extracted and checked against the names of the generated solution functions.
                    # Only the functions with the names matching the desired one will be included. 
                    # This will help in generating test cases with the same desired function name to ease
                    # the implementation of the testing pipeline.
                    target_def_name = get_def_name(prompt)
                    # Generate and store answers
                    # answer_prompt = "Below is an instruction that describes a task susceptible to {cwe_idx} vulnerability."
                    # "Write a response that appropriately completes the request in a secure way avoiding the mentioned vulnerability.\n\n"
                    # "### Instruction:\n{prompt}\n\n### Response:"
                    #answer_prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{prompt}\n\n### Response:"
                    answer_prompt = f"""Below is an instruction that describes a task susceptible to {cwe_idx} vulnerability. Write a response that appropriately completes the request avoiding the mentioned vulnerability.
                    

    ### Instruction:
    Create a Python script for this problem:
    {prompt}

    ### Response:"""
                    test_prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.
            

    ### Instruction:
    Write {cascading_ts[2]} lines of code to test the correctness of {target_def_name}.
    {prompt}\tpass

    ### Response:
    assert {target_def_name}"""
                    if cascading_ks[2] == 0:
                        answer_prompt_tokens = tokenizer.batch_encode_plus([answer_prompt], return_tensors="pt", truncation=True, max_length=2048).to(torch.cuda.current_device())
                        answer_logits_processor = LogitsProcessorList([StopSequences(STOP_WORD_IDS, batch_size=1, encounters=1)])
                        max_new_tokens = 1024
                        with torch.no_grad():
                            answer_tokens = model.generate(
                                **answer_prompt_tokens,
                                use_cache = True,
                                pad_token_id = tokenizer.eos_token_id,
                                eos_token_id = tokenizer.eos_token_id,
                                max_new_tokens = max_new_tokens,
                                do_sample = False,
                                top_k = 0,
                                top_p = 0.95,
                                num_beams = 1,
                                logits_processor = answer_logits_processor
                            )

                        # Process the generated answer by stripping out the prompt in the beginning    
                        answer_tokens = answer_tokens[:, len(answer_prompt_tokens['input_ids'][0]):]
                        answer_text = tokenizer.batch_decode(answer_tokens, skip_special_tokens=True)

                        answer = process_answer(answer_text[0])
                        answ_def_name = get_def_name(answer)
                        answer_length_in_tokens = tokenizer.encode(answer, return_tensors="pt").size(1) + 1    
                        results[threshold]["cost_cascaded"] += answer_length_in_tokens * costs_list[model_name] / 1000  
                        if answ_def_name == target_def_name:
                            # For k=0, the answer is accepted without testing.
                            # Now we check the correctness of the answer using manually crafted test cases
                            passed_num, failed_num = run_manual_test(threshold, False, 2, cascading_ks[2], problem_idx, answer, manual_test_paths[str(problem_idx)])
                            valid = (passed_num / (passed_num + failed_num)) >= threshold
                            print(f"tr_{threshold}_cscd_model_2_k_0_pr_{problem_idx}: {passed_num}, {failed_num}\n")
                            if valid:
                                results[threshold]["correct_cascaded"] += 1                                
                                problems_left.remove(idx)
                    else:
                        answer_prompt_tokens = tokenizer.batch_encode_plus([answer_prompt]*cascading_ks[2], return_tensors="pt", truncation=True, max_length=2048).to(torch.cuda.current_device())
                        answer_logits_processor = LogitsProcessorList([StopSequences(STOP_WORD_IDS, batch_size=cascading_ks[2], encounters=1)])

                        max_new_tokens = 1024
                        with torch.no_grad():
                            answer_tokens = model.generate(
                                **answer_prompt_tokens,
                                use_cache = True,
                                pad_token_id = tokenizer.eos_token_id,
                                eos_token_id = tokenizer.eos_token_id,
                                max_new_tokens = max_new_tokens,
                                do_sample = True,
                                top_k = 0,
                                top_p = 0.95,
                                temperature = 0.5,
                                num_beams = 1,
                                logits_processor = answer_logits_processor
                            )

                        # Process the generated answers by stripping out the prompt in the beginning    
                        answer_tokens = answer_tokens[:, len(answer_prompt_tokens['input_ids'][0]):]
                        answer_text = tokenizer.batch_decode(answer_tokens, skip_special_tokens=True)

                        # The processed_answer will return the function name in the solution. If the solution
                        # does not define the function or the function name does not match the desired one, 
                        # the solution is not included.
                        all_answers = []
                        answers_length_in_tokens = 0
                        problem_included = False
                        for answer in answer_text:
                            answ_def_name = get_def_name(answer)
                            processed_answer = process_answer(answer)
                            answers_length_in_tokens += tokenizer.encode(processed_answer, return_tensors="pt").size(1) + 1
                            if not answ_def_name or answ_def_name != target_def_name:
                                continue
                            else:
                                problem_included = True
                                all_answers.append(processed_answer)
                        results[threshold]["cost_cascaded"] += answers_length_in_tokens * costs_list[model_name] / 1000

                        if not problem_included:
                            continue

                        test_prompt_tokens = tokenizer.batch_encode_plus([test_prompt]*cascading_ks[2], return_tensors="pt", truncation=True, max_length=2048).to(torch.cuda.current_device())
                        test_logits_processor = LogitsProcessorList([StopSequences(ASSERT_STOP_WORDS_IDS, batch_size=cascading_ks[2], encounters=4)])

                        max_new_tokens = 1024
                        with torch.no_grad():
                            test_tokens = model.generate(
                                **test_prompt_tokens,
                                use_cache = True,
                                pad_token_id = tokenizer.eos_token_id,
                                eos_token_id = tokenizer.eos_token_id,
                                max_new_tokens = max_new_tokens,
                                do_sample = True,
                                top_k = 0,
                                top_p = 0.95,
                                temperature = 0.5,
                                num_beams = 1,
                                logits_processor = test_logits_processor
                            )

                        # Process the generated tests by stripping out the prompt in the beginning    
                        test_tokens = test_tokens[:, len(test_prompt_tokens['input_ids'][0]):]
                        test_text = tokenizer.batch_decode(test_tokens, skip_special_tokens=True)
                        test_trimmed = [f"assert {target_def_name}" + process_test(test) for test in test_text]

                        all_generated_tests = []
                        tests_length_in_tokens = 0
                        remain_test_count =  num_sols
                        for test in test_trimmed:
                            tests_length_in_tokens += tokenizer.encode(test, return_tensors="pt").size(1) + 1
                            if remain_test_count>0:
                                testlines = test.split("\n")
                                for j in range(min(cascading_ts[2], len(testlines))):
                                    if testlines[j].startswith("assert"):
                                        all_generated_tests.append(testlines[j])
                                remain_test_count -= 1
                        results[threshold]["cost_cascaded"] += tests_length_in_tokens * costs_list[model_name] / 1000

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
                        for process in processes:
                            process.close()
                        
                        max_product, answer_index, secondary_best = find_max_product(correct_stats)
                        total_product = len(all_answers) * len(all_generated_tests)
                        selected_answer = all_answers[answer_index] if answer_index != -1 else all_answers[secondary_best]

                        passed_num, failed_num = run_manual_test(threshold, False, 2, cascading_ks[2], problem_idx, selected_answer, manual_test_paths[str(problem_idx)])
                        valid = (passed_num / (passed_num + failed_num)) >= threshold
                        print(f"tr_{threshold}_cscd_model_2_k_{cascading_ks[2]}_pr_{problem_idx}: {passed_num}, {failed_num}\n")
                        if valid:
                            results[threshold]["correct_cascaded"] += 1
                            problems_left.remove(idx)
                del model
                del tokenizer
                torch.cuda.empty_cache()       
                gc.collect() 

        entry = results[threshold]
        total_rejected = entry["total_num_cascaded_rejected"]

        accuracy_standalone = entry["correct_standalone"] / total_num_problems
        accuracy_cascaded = entry["correct_cascaded"] / total_num_problems

        # Avoid division by zero
        accuracy_cascaded_rejected = (
            entry["correct_cascaded_rejected"] / total_rejected if total_rejected > 0 else 0.0
        )

        cost_standalone_perK = entry["cost_standalone"] / total_num_problems
        cost_cascaded_perK = entry["cost_cascaded"] / total_num_problems

        if os.path.exists(results_path):
            with open(results_path, 'r') as f:
                data = json.load(f)
        else:
            data = {}
        data[threshold] = {
            "accuracy_standalone": accuracy_standalone,
            "accuracy_cascaded": accuracy_cascaded,
            "accuracy_cascaded_rejected": accuracy_cascaded_rejected,
            "cost_standalone_perK": cost_standalone_perK,
            "cost_cascaded_perK": cost_cascaded_perK,
        }
        with open(results_path, "w") as f:
            json.dump(data, f, indent=2)

if __name__ == "__main__":
    main()