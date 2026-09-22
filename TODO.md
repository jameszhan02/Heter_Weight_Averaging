# Paper 2607.18026 Heterogeneous Model Merging: Learning and Implementation Checklist

Status markers: `[ ]` not started, `[~]` in progress, `[x]` complete, `[!]` blocked.

## Current Status

- Current phase: Phase 0 - establish the sources and code entry points.
- Next task: read the method and experiment sections of the paper and complete the paper method card below.
- Current blocker: this repository has no model-loading, merging, or evaluation code and no checkpoint or experiment configuration.
- Collaboration mode: the learner writes the code; the agent provides teaching, guidance, and review.

---

## Phase 0: Preparation and Fact Checking

### 0.1 Paper Method Card

- [ ] Record the paper title, version, and official code repository, if one exists.
- [ ] Write the exact mathematical definitions of Union and Intersection.
- [ ] Determine which model each interpolation coefficient multiplies.
- [ ] Record the recommended alpha values and their experiment settings.
- [ ] Record the mapping rules for layers, hidden dimensions, attention heads, MLP dimensions, and vocabulary.
- [ ] Record the exact definition of identity initialization for additional layers and the architectures to which it applies.
- [ ] Record the model pairs, tokenizers, datasets, metrics, and evaluation settings used by the paper.

Acceptance: every conclusion points to a formula, algorithm, table, or section in the paper rather than relying on memory.

### 0.2 Locate the Real Experiment Entry Points

- [ ] Identify the repository, branch, or directory that contains the existing model-merging experiments.
- [ ] Identify checkpoint formats and local locations. Do not commit model weights to Git.
- [ ] Record the environment, Python/PyTorch/Transformers versions, and available hardware.
- [ ] Select at least one intended small/large model pair.

Acceptance: the real code entry point, configuration entry point, and one accessible model pair are known.

---

## Phase 1: Read the Existing Code Without Changing the Algorithm

- [ ] Find the checkpoint-loading entry point.
- [ ] Find existing `state_dict` and parameter-handling logic.
- [ ] Find the checkpoint-saving entry point.
- [ ] Find tokenizer-loading logic.
- [ ] Find model-evaluation and generation entry points.
- [ ] Find how other merging baselines are registered, configured, and invoked.
- [ ] Draw the call flow from configuration through loading, merging, saving, and evaluation.
- [ ] Decide the module location, configuration name, and smallest public interface for the new method.
- [ ] List utilities that can be reused and evaluation settings that must remain unchanged.

Acceptance: produce a one-page code map with paths, key functions, and data flow without changing implementation code.

---

## Phase 2: Design Contracts and Parameter Mapping

- [ ] Define the target model, output key set, and target shape of every key for Union.
- [ ] Define the target model, output key set, and target shape of every key for Intersection.
- [ ] Define same-name parameter-matching rules.
- [ ] Define layer-index mapping and extra-layer handling when layer counts differ.
- [ ] Identify dimensions that may be truncated or expanded and shape differences that must raise errors.
- [ ] Determine whether embeddings and the LM head are tied and how tying is preserved after saving.
- [ ] Define the supported behavior for different tokenizers or vocabularies. Fail early when unsupported.
- [ ] Define support boundaries for attention heads, GQA/MQA, and rotary-embedding parameters.
- [ ] Define policies for biases, buffers, quantized weights, dtypes, and devices.
- [ ] Design strict-mode behavior and useful error messages.

Acceptance: for one real model pair, list representative parameters with source shapes, target shapes, mapping rules, and expected results.

---

## Phase 3: Intersection - Large to Small

### 3.1 `truncate_tensor`

- [ ] The learner writes the function contract and dimension-level pseudocode.
- [ ] Handle equal shapes with direct-copy semantics.
- [ ] Truncate when every source dimension is at least as large as the corresponding target dimension.
- [ ] Raise clear errors for rank mismatches or source dimensions smaller than target dimensions.
- [ ] Preserve the intended dtype and device and avoid unintended mutable storage sharing.

### 3.2 State-Dict Adaptation

- [ ] Match parameters by name and inspect every target key.
- [ ] Drop extra layers and preserve all target layers.
- [ ] Validate embeddings and the LM head.
- [ ] Validate attention parameters.
- [ ] Validate MLP parameters.
- [ ] Validate normalization parameters and buffers.
- [ ] Report missing, unexpected, and incompatible parameters.

Acceptance: the adapted state dict exactly matches the small model's keys and shapes, loads strictly, and contains no NaN or Inf values.

---

## Phase 4: Union - Small to Large

### 4.1 `expand_tensor`

- [ ] The learner writes the function contract and dimension-level pseudocode.
- [ ] Allocate a zero-initialized tensor with the target shape.
- [ ] Copy the source into the matching region specified by the paper.
- [ ] Raise clear errors for rank mismatches or source dimensions larger than target dimensions.
- [ ] Preserve the intended dtype and device and avoid unintended mutable storage sharing.

### 4.2 State Dict and Additional Layers

- [ ] Match existing layers by name.
- [ ] Implement the paper's exact identity initialization for additional layers.
- [ ] Validate the functional behavior of additional layers rather than checking only their numerical appearance.
- [ ] Validate embeddings and the LM head.
- [ ] Validate attention parameters.
- [ ] Validate MLP parameters.
- [ ] Validate normalization parameters and buffers.
- [ ] Report missing, unexpected, and incompatible parameters.

Acceptance: the adapted state dict exactly matches the large model's keys and shapes, loads strictly, contains no NaN or Inf values, and gives the paper-defined behavior for additional layers.

---

## Phase 5: Weighted Interpolation and Unified Interface

- [ ] Confirm the formula and alpha semantics and document them in docstrings and configuration help.
- [ ] Implement parameter-wise interpolation under explicit key, shape, dtype, and device policies.
- [ ] Support configurable alpha and validate its allowed range.
- [ ] Keep the method training-free; do not introduce an optimizer or backward pass.
- [ ] Design one interface from small model, large model, mode, and alpha to a merged state dict.
- [ ] Support `union` and `intersection` and fail early for invalid modes.
- [ ] Reuse existing saving utilities without overwriting source checkpoints.
- [ ] Make the output checkpoint directly loadable by the existing evaluation entry point.

Acceptance: the interface clearly documents its target space and alpha direction, and output keys and shapes exactly match the target model.

---

## Phase 6: Validation With Real Checkpoints

- [ ] Check every parameter name and output shape.
- [ ] Check missing, unexpected, and incompatible parameters.
- [ ] Check every floating-point tensor for NaN and Inf values.
- [ ] Check dtypes, devices, and tied weights.
- [ ] Strictly load the Intersection checkpoint.
- [ ] Strictly load the Union checkpoint.
- [ ] Verify the exact endpoint meaning of alpha equal to zero.
- [ ] Verify the exact endpoint meaning of alpha equal to one.
- [ ] Verify that Union adaptation does not unexpectedly destroy the small model's core behavior.
- [ ] Verify that Intersection produces a valid small model.
- [ ] Generate from a small fixed prompt set using the existing tokenizer.

Acceptance: saving, reloading, forward passes, and generation succeed, with a machine-readable validation report retained.

---

## Phase 7: Integrate With the Existing Experiment Pipeline

- [ ] Register the `2607.18026` method and its Union and Intersection configurations.
- [ ] Reuse the existing checkpoints, tokenizer, and evaluation setup.
- [ ] Use exactly the same datasets, prompts, decoding settings, and metrics as other methods.
- [ ] Record seeds, versions, hardware, commands, and artifact paths.
- [ ] Run a minimal smoke evaluation before full evaluation.

Acceptance: changing only the method configuration runs the same evaluation pipeline without method-specific evaluation changes.

---

## Phase 8: Baselines and Comparison

Run the following for every existing model pair:

- [ ] Small model.
- [ ] Large model.
- [ ] Naive Merge.
- [ ] HeteroFusion.
- [ ] Paper 2607.18026 - Intersection.
- [ ] Paper 2607.18026 - Union.
- [ ] Other existing baselines.

Alpha sweep, subject to confirmation from the paper method card:

- [ ] 0.01.
- [ ] 0.03.
- [ ] 0.05.
- [ ] 0.10.
- [ ] Exact paper-recommended values and any necessary additional points.

Shared metrics:

- [ ] GSM8K.
- [ ] MMLU.
- [ ] Other existing tasks.
- [ ] Capability retention.
- [ ] Target-task performance.

Acceptance: every method shares the evaluation protocol, and the result table records configurations, means and variation where applicable, logs, and failures.

---

## Phase 9: Paper Reproduction

- [ ] Select at least one model pair from the paper.
- [ ] Use the same merge direction.
- [ ] Use the same interpolation ratio.
- [ ] Match model revisions, data, and evaluation settings where possible.
- [ ] Record conditions that cannot be matched exactly.
- [ ] Compare against the paper and quantify the reproduction gap.
- [ ] Propose testable explanations for gaps rather than unsupported guesses.

Acceptance: retain a reproducible configuration, raw results, a mapping to the relevant paper table, and an explanation of the gap.

---

## Phase 10: Research Conclusions

- [ ] Does the method work for the existing model pairs?
- [ ] Which of Union and Intersection works better, and under what conditions?
- [ ] How sensitive is performance to alpha?
- [ ] Does the method retain the large model's capabilities?
- [ ] How does it compare with HeteroFusion?
- [ ] Does it work across model families?
- [ ] What happens when tokenizers or vocabularies differ?
- [ ] What are the primary failure modes and current support boundaries?

Acceptance: every conclusion cites a concrete experiment or failure case and distinguishes observations, inferences, and untested hypotheses.

---

## Definition of Done

- [ ] Union is implemented and validated.
- [ ] Intersection is implemented and validated.
- [ ] Weighted interpolation and endpoint behavior are validated.
- [ ] Real merged checkpoints load strictly and generate successfully.
- [ ] The existing evaluation pipeline evaluates the method directly.
- [ ] At least one paper experiment is reproduced.
- [ ] Results are included in the heterogeneous-merging comparison.
- [ ] Code, configurations, commands, environment, and results are traceable.
- [ ] The learner can independently explain parameter mapping, initialization, interpolation, and validation design.

---

## Experiment Log Template

- Date:
- Git commit:
- Model pair and revisions:
- Mode and alpha:
- Configuration or command:
- Environment and hardware:
- Results:
- Errors and interpretation:
- Next step:
