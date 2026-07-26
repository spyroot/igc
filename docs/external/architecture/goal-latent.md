# Goal Latent Boundary

Phase 3 emits an unordered `calls` set. The concrete calls remain the execution and verification
authority. Goal encoders provide two optional, separate RL conditioning views:

```text
call.rest_api + associated resource context -> z_rest
call.http_method + operation_name + argument schema -> z_method
```

For one to three operations, the public shapes are:

```text
z_rest:         [batch, max_operations, d_rest]
z_method:       [batch, max_operations, d_method]
operation_mask: [batch, max_operations]
```

`z_rest` initializes from the promoted Phase 2 `goal_extractor`. `z_method` initializes from the
promoted Phase 3 `argument_extractor`. Both are frozen and version-pinned during an RL experiment.
The binding source and freeze rules are in `configs/contracts/checkpoint_lineage.yaml`; the tensor
boundary is in `configs/contracts/goal_latent.yaml`.

## Literal Values

Literal argument values remain outside both latents. For example, `set x = 1` and `set x = 2` may
have nearby `z_rest` and `z_method`, while the raw bindings remain distinct:

```json
{"x": 1}
```

```json
{"x": 2}
```

The executor and success verifier always retain the concrete `calls` and exact bindings. Latent
distance never substitutes for state verification.

## RL Boundary

Phase 2 and Phase 3 do not encode execution order or prerequisite actions. The RL policy may consume
concrete calls, latent views, or both, then learn ordering, retries, waiting, recovery, and additional
environment-dependent actions from transitions. Version 1 does not claim a unified goal latent, a
shared state/goal latent distribution, proven OEM equivalence, or universal REST transfer.

Author:
Mus mbayramo@stanford.edu
