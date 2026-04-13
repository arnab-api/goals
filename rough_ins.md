Rough plan to implement a optimization method to extract a universal subspace where the LM encodes its *intent/goal* for the immediate future step. And, is able to introspect its latent reasoning by verbalizing the intent (similar to patchscope setup).

---

Ok. Since the attention output patching is sort of working, I think it might be possible to get a subspace of the residual latents that contain the information about the future steps (*intent* or *goal*). We will do this by optimization. Here's the rough outline of the plan:

1. *Subspaces*: 

For a selected range of layers (e.g., layers 35 - 45), we will define a *r*-dimensional subspace (e.g., r = 256) for each of the layers. More concretely:

Let's assume, $d$ is the residual dimension (e.g., 8096). For an individual layer $l$, we will have a matrix $W_{l} \in \mathbb{R}^{r \times d}$ that maps the residual latents ($\in \mathbb{R}^d$) to the subspace ($\in \mathbb{R}^r$). The subspace representation of the residual latents at layer $l$ will be given by:
$$
s_{l} = W_{l} r_{l}
$$

2. *Soft-prompts for the context in the destination run*:

Currently, our destination prompt is pretty simple: 
```python
"""<|im_start|>system
You are a helpful assistant proficient in programming.<|im_end|>
<|im_start|>user
What are you doing at this line of the code. Keep your answer as brief as possible.<|im_end|>
<|im_start|>assistant"""
```

Instead of doing this, I am thinking of learning a soft prompt (a set of learnable tokens) that can be prepended to the destination prompt. The template might look like:

```python
"""<|im_start|>system
You are a helpful assistant proficient in programming.<|im_end|>
<|im_start|>user
{soft_tok} {soft_tok} {soft_tok}<|im_end|>
<|im_start|>assistant"""
```

Just a few tokens (maybe 3-5) should be enough. The soft tokens is supposed to nudge the LM in the destination run to interpret the patched subspace information.


**Optimization objective**: 
* We will compute the subspace from a token-layer position of the source run (a code snippet from `well-known-tasks/generations`). 
* For a token, patch (for the selected range of layers) to the destination run (without any function specific context). 
* Run the destination run to generate $n$ tokens, such that the generated tokens match the corresponding labels from `well-known-tasks/labels`. 

So, the losses are:
* (The main loss) Next word prediction loss (cross-entropy) for the generated tokens.
* Orthogonality loss, $L_{\text{ortho}} = \sum_{l} \|W_{l}^T W_{l} - I\|_F^2$, so that
    * The subspace captures distinct info
    * The inverse of the subspace is well-defined (i.e., $W_{l}^T$ can be used to map back to the residual space).
* Weight decay, on both the subspace projection matrices and the soft prompt tokens, to prevent them from growing too large.

---

Issue 1: ok!

Issue 2: Patching mechanism:

$$
\Delta_l = W_l h_l^{\text{src}} - W_l h_l^{\text{dest}} = W_l (h_l^{\text{src}} - h_l^{\text{dest}}) \\

r_l^{\text{patched}} = r_l^{\text{dest}} + W_l^T \Delta_l = r_l^{\text{dest}} + W_l^T W_l (h_l^{\text{src}} - h_l^{\text{dest}})
$$

Issue 3: I think the soft prompts are fine as they are placed. Since I want *introspection*, I don't wanna apply too much pressure on the LM's context itself.

Issue 4: Yes, good catch! Since we want the *intent*, the tokens should be *before* the LM starts generating the code for this step. So, we should include:

* It should be the first token of the intent code snippet.
* Also, the immediate preceeding token (before the first token of the intent code snippet), for the same reason. 