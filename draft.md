let's brainstorm on the gaze head attention patterns. When the LM generates the code related to a specific instruction:
  1. some tokens consistently look back to the *key* words of the insruction (`ConditionSpec.id_substrs`), which is expected. These are the important tokens that are bringing the task information and dictating the
  implementation.
  2. some tokens, within the same code segment, have very distributied attention. My hypothesis is that these are not playing a crucial role. They just have to do with the syntactic styles.
  3. at the end of the code (almost always at `"\n"` for monolithic), that token suddenly shifts its attention to either a) distributed across everything else or b) the next instruction or print

The transition signal we are looking for is the property that makes the final token shift its attention from the current task instruction. How do we capture this property/feature? 

Here's an idea: it is present in the nth token, and not in the (n-1)th token. For a gaze head h, its Q projection reads from the latent of previous layer and decides what to look for. So, this is rather a *transition* between what the q states of specific tokens decide to look at. So, does the difference capture the transition? Formally:

$$\Delta q = \mathbb{E}[\text{Q}(h(n) - h[n-1])]$$

Where $\text{Q}$ is the Q-projection matrix of a gaze head $[\ell, j]$. Projecting it back to the residual stream space, we get:
$$\Delta h' = \text{Q}^\dagger (\Delta q)$$
where $\text{Q}^\dagger$ is the pseudo-inverse of $\text{Q}$.

Since, there are multiple gaze heads, we can just sum up the $\Delta h'$ across all gaze heads to get the overall transition signal in the residual stream space. Then we can analyze this $\Delta h'$ to see if it has the targetted transition property. Here are the tests:

1. Causal validation: 

a) Ablate tansition signal: If we ablate $\Delta h'$ from the residual stream at the nth token, does the model fail to transition to the next instruction? What is the expected behavior? Loop around the current instruction repeatedly?

b) Add transition signal mid-way: Stop the current instruction half way, and move on to the next thing? 

Caveat: If the LM also has a *verifier* signal that check *did I do things right up to this point?*, then this signal will fight against both 1a and 1b.

2. Simple verbalization of transition signal: with logit_lens, and goalscope setup.