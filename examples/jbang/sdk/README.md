# Gollek JBang SDK Examples

This folder contains ML framework and SDK-facing JBang examples.

During migration, this folder intentionally includes two lanes:

- compatibility examples that still use older `gollek-sdk-*` naming
- newer examples that align better with the canonical module split

## Recommended Order

1. `gollek-quickstart.java`
2. `gollek-sdk-core-example.java`
3. `gollek-sdk-train-example.java`
4. `gollek-sdk-vision-example.java`
5. `gollek-sdk-export-example.java`
6. `gollek-sdk-augment-example.java`

Run from `gollek/examples/jbang`:

```bash
jbang sdk/gollek-quickstart.java
jbang sdk/gollek-sdk-core-example.java
jbang sdk/gollek-sdk-train-example.java 2
jbang sdk/gollek-sdk-vision-example.java
jbang sdk/gollek-sdk-export-example.java
jbang sdk/gollek-sdk-augment-example.java
```

## Additional Samples

Compatibility-named files now rewritten as runnable scripts:

- `gollek-quickstart.java`
- `gollek-sdk-core-example.java`
- `gollek-sdk-vision-example.java`
- `gollek-sdk-train-example.java`
- `gollek-sdk-augment-example.java`
- `gollek-sdk-export-example.java`

Exploratory v0.3-style demos:

- `unified_framework_demo.java`
- `graph_fusion_example.java`

Legacy v0.2/v0.3 references:

- `tensor_operations_v02.java`
- `vision_transforms_v02.java`
- `tokenization_v02.java`
- `mnist_training_v02.java`
- `pytorch_comparison_v02.java`

Treat these as evolving references rather than the canonical trainer/runtime
surface.

## Prerequisites

From `gollek/` project root:

```bash
./run-install-local-macos.sh
```

Or publish only the modules a specific script needs with targeted
`publishToMavenLocal` tasks.

Then run with JBang:

```bash
jbang --fresh sdk/gollek-quickstart.java
```
