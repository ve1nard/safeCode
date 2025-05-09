## Prefix-Tuning for CodeQwen and DeepSeek
---

### 1. Download the SVEN Repository

Begin by cloning the official SVEN repository:

```bash
git clone https://github.com/eth-sri/sven
cd sven
```


### 2. Apply Prefix-Tuning Modifications

Navigate to the `sven-codeQwen+deepseek` directory to view the modified files.  
Then, apply these modifications to the corresponding files in the `sven` repository.


### 3. Instruction Tuning 

The instruction tuning repo can be found here : https://github.com/eth-sri/SafeCoder

## Cascading
---

### 1. Data sets

Navigate to *data/cascading* to familiarize yourself with validation and evaluation datasets for cascading experiments as well as our results for the best cascading schemes.

### 2. Reproducing results

Navigate to *src/cascading* directory from the `gen_ans_test.py` script to generate answers and tests for all possible hyperparameter combinations. Next, run `select_ans_tests.py` to simulate the cascading pipeline with different configurations to produce security and cost results for each of them. Afterwards, using `check_parameter_combs.py` select the best combinations according to the security and cost scores. To see which of those points are Pareto-optimal, run `get_pareto_points.py`. After finding Pareto-optimal points, you can visualize them together with the rest of the configurations using `plot_thresholds.py`. Select the best points of your interest for each threshold and indicate them in *data/cascading/outputs/chosen_hyperparameters.json*. To perform testing on the evaluation split, run `test_parameters.py` and see the results in *data/cascading/outputs/testing_results_new_method/testing_results.json*.

