For each of these well-known programming tasks (goals), I want to be able to ask a code-LM to complete the function. For atoi I have the following `IncompleteCodeSample`:

```python
IncompleteCodeSample(
    instruction="Implement a string to integer conversion in Python. Write your code without any comments or explanations, just the implementation.",
    code='def atoi(s):\n    """',
    unit_tests=[],
    function_name="atoi",
    prefills=[],
    augmentation=None,
)
```

Which generates the following prompt (chat template applied, qwen-2.5-coder tokenizer):
```
'''<|im_start|>system
You are a helpful assistant proficient in programming. When completing code, ensure proper syntax and logical 
correctness. Make sure to have all codes inside the <code> </code> tags.<|im_end|>
<|im_start|>user
Implement a string to integer conversion in Python. Write your code without any comments or explanations, just the 
implementation.<|im_end|>
<|im_start|>assistant
<code>
def atoi(s):
    """'''
```

I want you to create a dataset of such `IncompleteCodeSample`s for every programming tasks (fetch more if possible).

* File structure: Create folder `well-known-tasks` in data. For each task, create a separate json file named `<task_name>.json` containing just a json dict of IncompleteCodeSample.
* You will need to populate the function name and the default arguments. Check out the `atoi` example above for reference.
* Add the following args in `IncompleteCodeSample` dataclass
    * `lang`: (default "python") programming language of the code sample
    * `without_comments`: (default False) whether the code sample should have comments or not. If True, the instruction should explicitly ask for no comments (append `"Write your code without any comments or explanations, just the implementation."`).

* I don't care about the unit tests, prefills, or augmentations for now. You can leave them as empty as default values.

