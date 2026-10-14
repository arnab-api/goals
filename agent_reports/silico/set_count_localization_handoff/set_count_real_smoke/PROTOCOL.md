# Pretrained 32B localization smoke and loss check

Authorized by the researcher: run the new localization path on the actual `Qwen2.5-Coder-32B-Instruct` checkpoint and check whether loss decreases. This is an optimization smoke, not a confirmatory mechanism experiment.

## Identity and resource route

- Model: `Qwen/Qwen2.5-Coder-32B-Instruct`, revision `381fc969f78efac66bc87ff7ddeadb7e73c218a7`.
- Public, ungated access confirmed from model metadata and config. The checkpoint contains 65,527,841,688 safetensors bytes.
- CPU-only checkpoint/environment staging, then one H100 for the actual model. Full BF16 weights, no quantization or model substitution. Load once and reuse for all cases.
- Reuse the previously validated Python 3.11 / PyTorch CUDA environment, adding W&B through a newly retained dependency lock. Only the GPU job receives the configured W&B credential setting.

## Fixed smoke configuration

- Independent masks: set/head, set/SVD, count/head, count/SVD.
- Candidate query layers 20 through 55 inclusive; downstream model computation unchanged.
- Default full-transfer mask initialization 1.0, AdamW learning rate 0.01, weight decay 0, control multiplier 1, KL weight 0, sparsity weight 0.
- Twelve optimizer steps per case. One entire matched family per step, cycled deterministically. No tuning or selection on final loss to manufacture success.
- Prioritize three concrete operation contrasts from the checked-in pool: string lowercase/palindrome, integer even/divisible-by-three, integer-list all-same/all-even. Keep the common expression prefix and neutral output names.
- Select the first valid family per datatype in a fixed candidate order before optimization. The known assignment plus up to three deterministic naming/auxiliary alternatives per datatype; record every rejection. All nine A/B/C by core/swapped/four-recipient views must pass together.
- Readout validation uses real-model next-token distributions, never artificial IDs. Save actual token IDs/strings, clean and teacher distributions, selected prompts, and representative greedy continuations. Full semantic completion correctness remains unverified in this bounded test.

## What establishes loss descent

Evaluate the exact same accepted collection at step 0 and after steps 3, 6, 9, and 12. Report total expected-token cross-entropy, primary (nonnative-expected) CE, and control (native-expected) CE separately, plus expected-token accuracy and log-odds. The final-minus-initial difference answers whether the objective decreased; no monotonically decreasing curve is assumed.

Training-event losses may come from different cycled families and cannot by themselves establish a trend. A reduction driven only by controls must be reported as such. A finite gradient or a changed mask is not evidence of loss descent. A failed descent check is an informative outcome, not permission to change the labels or hide the initial run.

Also record the clean/unpatched expected-token loss from cached readouts, so improvement from the full-transfer initialization is not mistaken for outperforming an unpatched model. Verify correct GPU/dtype placement, preserved input prefix, zero-mask clean behavior, finite gradients, frozen base weights, bounded masks, checkpoint persistence, memory, and timing. Keep raw records and W&B curves. Do not use the test split or call these small training-set measurements generalization evidence.

## Artifacts and authorization boundary

The existing delivered patch remains unchanged unless a concrete implementation defect requires a repair. The real-model harness is separate. No managed repository commits or pushes are authorized by this request. Preserve downloaded weights, checkpoints, curves, failures, and protocol changes for reuse and review.
