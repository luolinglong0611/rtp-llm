# TODO

- Hybrid attention group split cleanup: `HybridConfigCreator::calculateGroupLayerNum`, `createLayerGroups`, and `setupCacheConfigSpecs` are no longer on the `createHybridConfig()` production path. Keep the behavior for now, but later move/retire these helpers into tests and use them only to verify equivalence between the old C++ split semantics and the current Python `KVCacheSpecDesc` tag-driven grouping.
