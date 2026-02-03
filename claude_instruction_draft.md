1. By budget I meant, setting the depth and branching factor limits as mentioned above. I don' tneed to set the node count or time limit explicitly.

2. I should be able to set this threshold. I think the default value can be something like 0.03.

3. I expect most of the simple code generation tasks to be done within a single line. So, you should consider EOS and newline characters as terminal nodes as well. Don't execute the code mid-generation to check for syntax errors.


---


Feedback on MCTS.

1. Node representation: 
- Should be A: single token, as I want to be able to analyze each of the token representations later.
- But, can you elaborate more on your recommendation of C? How are you planning to implement this?

2. Action Space:
- Leaning towards A: Top-K. Agree with your recommendation here. 
- But, how do you plan to implement C: Temperature-scaled sampling? In this situation, will you be potentially sampling the same token with larget logit multiple times in the same node expansion?

3. Reward Function:
- Option C makes sense. However, I am not planning to use MCTS to train any policy model. Is it possible to just get the expansions until end tag </code> without worrying about reward shaping?

4. Selection Policy:
- Yes, let's go with PUCT. 

5. Don't worry about integrating prefills. I actually think integrating IncompleteCodeSample will unnecessarily complicate things at this stage. Let's keep it simple for now: the search should start from a simple text prompt or TokenizerOutput class from src.utils.typing.

- I agree with your recommendation: the search should start after the fixed prefil.

6. How were you planning to update visits in this situation?


On the questions:

I think I've answered 1-3. Let me know if I missed anything.


Output format: 

Both the full tree and all terminal nodes would be unfeasible given the large vocabulary size of the LMs. 

I should be able to allocate some kind of computation budget and heuristic to keep the search tractable. For example:

1. Depth: setting the maximum depth of the tree to a reasonable value (maybe 10-20 tokens)
    * Finish searching further if we reach the end tag </code> before reaching the maximum depth.
2. When branching, limit the number of child nodes to top k. Also, make sure that we are not exploring very low probability tokens (less than a certain threshold). This will make the branching uneven, but it will help keep the search tractable.
3. For now, after the allocation budget is exhausted, return the full expanded tree. Later on, we can think about keeping only the paths that produced valid codes and passes the unit tests.
