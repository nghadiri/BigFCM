# How BigFCM works

This guide explains the algorithm as implemented in `src/bigfcm`. For the original treatment, experiments and figures, read the [paper](../README.md#paper).

## Background in one minute

**Fuzzy C-Means (FCM)** places `C` centers in the data. Each record gets a membership degree in every cluster (degrees sum to 1) instead of a single hard label. It alternates two steps until the centers stop moving:

1. compute each record's memberships from its distances to the centers;
2. recompute each center as the membership-weighted mean of the records.

A fuzziness exponent `m > 1` controls how soft the assignment is. Every iteration touches every record, which is what makes plain FCM expensive on big data.

**Weighted FCM (WFCM)** is the same algorithm where each record `k` carries a weight `w_k` that scales its pull on the centers. A record with weight 500 behaves like 500 identical records. BigFCM uses this to treat a partition-level center as a stand-in for all the records behind it.

## The three phases

### Phase 1: seed centers from a sample (driver)

`PwfcmDriver.first_run` reads a small sample of records (size = the `<sample-size>` argument) and clusters it with ordinary FCM. The `C` resulting centers are written to `initialize_center` and registered with `job.addCacheFile(...)`, so every task can read them locally.

Why bother: a random start can take hundreds of iterations on big data. Starting from centers fitted to a sample is already close to the final answer.

The sample can be clustered in one pass or in small chunks (chunks of `C` records in the chunked variant). In the chunked variant, after each chunk the new centers are merged with the previous chunk's centers using WFCM (`converge_prev_cur_cntrs`, which clusters the `2C` centers with their weights).

### Phase 2: cluster each partition locally (mapper and combiner)

- `PwfcmMapper` only parses lines, checks that each has the expected number of fields, normalises the separator, and emits `(0, "x1 x2 ... xd")`.
- `PwfcmCombiner` does the real work. It loads the cached seeds and runs FCM over the records it was handed. For each of the `C` clusters it emits a line with the center's coordinates and a **weight**: the sum of memberships of that partition's records in the cluster.

The partition therefore collapses to `C` weighted points. Only those cross the network.

### Phase 3: merge with weights (reducer and driver)

`PwfcmReducer` collects the weighted centers from all combiners and runs WFCM on them, again starting from the cached seeds, to get `C` merged centers. Because weights record how many records each partition-level center represents, a small, odd partition cannot drag the result as much as a large one.

After the job, `PwfcmDriver.last_run` reads `part-r-00000` and runs one more WFCM pass in the driver, then writes the final centers to `real_out` inside the job's output directory.

## What is innovative

1. **One job, not one per iteration.** Iterating happens *inside* the combiner and reducer, on data already in memory. The usual cost of re-launching MapReduce and re-reading HDFS on every iteration disappears.
2. **Complete data, not a sample.** The sample only chooses the starting point. Every record contributes to the final centers through the weighted summaries.
3. **Weights make partial results comparable.** Many "cluster each chunk, then combine" schemes treat all chunk centers equally. BigFCM carries the membership mass forward, so the combination step is principled.
4. **Pre-clustering cuts iterations.** Seeds from a sample typically let the main run converge in a small fraction of the iteration budget.
5. **Tiny shuffle.** The combiner turns a partition into `C` weighted points, so the reducer's input is kilobytes, regardless of dataset size.
6. **Cheaper partition update.** The partition-level update accumulates sums directly from distances and never stores the full membership matrix, giving O(n·C) work per pass instead of O(n·C²).
7. **Distributed Cache for the seeds.** Every node reads the same small file locally, with no extra coordination.

## Mapping to the code

| Concept | Where |
|---|---|
| Sampling and pre-clustering | `PwfcmDriver.prepare_str_2`, `first_run`, `fcm`, `update_centroids_first` |
| Merging chunk centers during pre-clustering | `PwfcmDriver.converge_prev_cur_cntrs`, `wwfcm`, `update_centroids_w` |
| Seeds to every node | `job.addCacheFile(.../initialize_center)` in `RunJob` |
| Parsing and keying | `PwfcmMapper.map` |
| Local FCM with weights out | `PwfcmCombiner.reduce`, `update_centroids`, `round_normal_center_weight` |
| Final merge | `PwfcmReducer.reduce`, `update_centroids` (weighted) |
| Final pass and output | `PwfcmDriver.last_run`, `wfcm`, `update_centroids_weight` |

## Implementation notes

The code is kept as published. Here are some places where it behaves differently from the simplest reading of the algorithm. They matter if you plan to extend it.

- **Two pre-clustering variants, one is always used.** `RunJob` times two variants of the sample clustering (`Instance` 0: one FCM pass over the whole sample; `Instance` 1: chunks of `C` records merged with WFCM). Both outcomes of the comparison set `Instance` to 1, and the value is set after the `Job` has copied the configuration. The combiner also reads the key `instance` (lowercase), which is never set, and then overwrites its `divider` with the number of records it received. In practice the combiner clusters everything it receives in one pass.
- **Single group.** The mapper's key is `counter % 1`, which is always 0. All records share one key, so the work is split by map-side combining over input splits, not by key partitioning.
- **Combiner semantics.** Hadoop may run a combiner zero, one or several times per map task. The design assumes it runs; if you adapt it, make the combiner optional-safe or run the logic in the mapper.
- **Weight rounding.** The combiner rounds its weights to integers (`round_normal_center_weight`).
- **Sampling input format.** The driver's sampler (`prepare_str_2`) splits on a comma, takes lines from the *last* file listed in the input directory, and keeps roughly half of the leading lines via a coin flip until it has enough. The mapper, in contrast, honours the `<separator>` argument.
- **Hard-coded cache path.** Seeds are written to `/usr/local/hadoop/sample/initialize_center`. Change this in `RunJob` for your environment.
- **Single reducer.** With one key there is one reduce group. The paper discusses multi-reducer cascading; this code does not implement it.
