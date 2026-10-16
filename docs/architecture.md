# MoRE text architecture

MoRE is the working name for a causal language model that combines two different
forms of conditional computation:

- **Mixture of Experts (MoE):** each active token is sent through the top-k
  feed-forward experts selected by a learned router.
- **Mixture of Recursions (MoR):** one Transformer block is reused for a
  token-dependent number of recursive passes.

The hypothesis is that these mechanisms address different axes of difficulty:
MoE can specialize *what transformation* a token receives, while MoR can
choose *how much computation* a token receives. Combining them may improve
quality per active parameter or reduce average compute, but the combination is
not automatically faster or better. This repository therefore treats MoRE as
a falsifiable research prototype, not a claim of production efficiency.

## Forward pass

```text
UTF-8 bytes + positions
        |
        v
  depth router --------------------+
        |                          |
        | hard max depth           | soft cumulative gates (training)
        v                          v
  shared causal Transformer block x 1..R
        |  each pass contains:
        |    pre-norm causal attention
        |    residual connection
        |    pre-norm top-k MoE FFN
        |    residual connection
        v
  final norm -> tied language-model head -> next-byte logits
```

The depth router emits a distribution over `R` possible terminal depths. A
hard assignment to depth `d` is active on passes `1..d`; it never re-enters
a later pass after halting. During training, the forward update uses a
straight-through cumulative soft gate so the router receives gradients while
inference and routing statistics remain discrete.

The first implementation computes the shared block for the complete sequence
and masks inactive residual updates. This validates causality and routing
semantics but does **not** make the dense PyTorch execution faster. A later
optimization can pack active tokens with gather/scatter or continuous
depth-wise batching. Benchmarks must measure that optimization rather than
assuming a mask reduces FLOPs.

## Why token-choice is the default

The default depth router is token-choice: each token chooses its own maximum
recursion depth from its current causal representation. Expert-choice routing
that globally selects the highest-scoring tokens can be useful for fixed
capacity, but selection over a complete sequence is awkward for autoregressive
inference and can introduce future-token information during training. The
[MoR paper](https://arxiv.org/abs/2507.10524) discusses both choices and the
trade-off between predictable capacity and flexible inference-time depth.

## Losses and diagnostics

The training objective is:

```text
LM loss
+ depth_balance_weight * depth load-balance loss
+ moe_balance_weight * expert load-balance loss
+ router_z_weight * router z-loss
+ compute_penalty_weight * (mean_depth - target_mean_depth)^2
```

The load-balancing terms use hard assignment frequency and mean soft routing
probability, similar to the standard sparse-MoE objective. Every run reports:

- validation loss and perplexity;
- mean selected depth and depth fractions;
- depth/expert routing entropy;
- expert utilization and auxiliary-loss magnitudes;
- parameter count and, after optimization work, throughput.

Reasoning behavior must be evaluated on held-out tasks or datasets. Routing
entropy or a larger average depth is not evidence of reasoning quality by
itself. The repository deliberately does not require exposing hidden
chain-of-thought; task accuracy, calibrated answers, and compute/latency
curves are the observable targets.

## Ablations

Use the supplied configs to compare the same seed and corpus:

| Variant | `use_mor` | `use_moe` | Question |
| --- | ---: | ---: | --- |
| Dense | false | false | What does the ordinary shared block do? |
| MoR-only | true | false | Does adaptive recursive depth help? |
| MoE-only | false | true | Does expert specialization help? |
| MoRE | true | true | Are the effects complementary? |

Compare validation perplexity, parameter count, mean depth, expert utilization,
and tokens/sec. Match model width and training budget before interpreting a
result. A useful follow-up is to add a packed active-token implementation and
then compare actual FLOPs, memory, and latency.

## References

- Kim et al., [Mixture-of-Recursions: Learning Dynamic Recursive Depths for
  Efficient Language Modeling](https://arxiv.org/abs/2507.10524).
- DeepSeek-AI, [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
  for a large-scale sparse-MoE reference and routing/load-balancing context.
- Elbayad et al., [LayerSkip: Enabling Early Exit Inference and Self-Speculative
  Decoding](https://arxiv.org/abs/2404.16710) for related adaptive-depth and
  shared-layer deployment considerations.
