Now I want you to write another script to test steerability of the extracted goal directions.

1. Checkout the "## Test causality" section in the 03_probe.ipynb notebook. I want you to copy (or follow) relevant code from there and prepare a script 'test_causality.py' in the scripts directory.

2. The script should take the following command line arguments:
    * --model: The name or path of the model to test.
    * --goal_dir_path: The path to the extracted goal direction. Should read from "env_utils.DEFAULT_RESULTS_DIR / {model_name_short_form} / sample_{idx}". The script should load the goal directions saved by running the `goal_dir.py` script. 
    * --layer_from: The layer from which to apply the goal direction. default: 0 (the first layer).
    * --layer_to: The layer to which to apply the goal direction. default: mt.n_layer (the last layer).
    * --n_generations: number of generations to sample per prompt from the LM. default: 100.

3.  You will see that there is another hparam `coeff`, which is still a bit finicky and needs to be tuned per sample. You should run a grid search in range(0.02, 0.21, 0.02) to find the best `coeff` for each sample. You don't need to expose `coeff` as a command line argument.

4. You should track these results sample-wise
    * baseline performance: pass_count / n_generations without steering with goal directions
    * contrastive performance: pass_count / n_generations with steering with goal directions (using the best `coeff` found from the grid search)
    * mean_diff performance: pass_count / n_generations with steering using the mean_diff directions. 
You should save the contrastive and mean_diff results for each coeff. baseline can be calculated only once as it doesn't depend on `coeff`. 

5. You should save the sample-wise results in their respective directiories from where you loaded the goal directions. And the aggregated results in a file named "causality_results.json" in the model directory (i.e., "env_utils.DEFAULT_RESULTS_DIR / {model_name_short_form} / causality_results.json").

6. After you process each sample, you should log the aggregated results with the best `coeff` for that sample, free the gpu cache and move to the next sample.

--- 

All good questions.

1. There was a typo in my instructions. Should read from "env_utils.DEFAULT_RESULTS_DIR / {goal_dir_path} / {model_name_short_form} / sample_{idx}"
2. For an array of prefills. by default set `prefills = ["(3", "1 -", "5", "3 +"]`. no need to expose as a command line argument. You can hardcode this in the script. But make sure to mark it so that it is visible and easy to change in the future. For now find the best `coeff` for each sample *and* for each prefills. this is not ideal. you should remember that (note on CLAUDE.md file) and remind me to address this issue in the future.
3. You should do one get_hs call and individually tune `coeff` per prefill.