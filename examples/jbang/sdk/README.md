# Gollek v0.3 SDK Examples

New examples demonstrating Gollek v0.3 framework capabilities.

## New Examples

| Example | Description | Usage |
|---------|-------------|-------|
| `unified_framework_demo.java` | Comprehensive demo of all v0.3 capabilities | `jbang unified_framework_demo.java --demo all` |
| `graph_fusion_example.java` | Graph fusion benchmark and usage | `jbang graph_fusion_example.java --size 8192` |

## Unified Framework Demo

Demonstrates all v0.3 capabilities in a single runnable example:

```bash
# Run all demos
jbang unified_framework_demo.java --demo all

# Run specific demos
jbang unified_framework_demo.java --demo runner
jbang unified_framework_demo.java --demo batching
jbang unified_framework_demo.java --demo fusion
jbang unified_framework_demo.java --demo quantization
jbang unified_framework_demo.java --demo memory-pool

# Specify model and device
jbang unified_framework_demo.java --model model.onnx --device cuda
```

## Graph Fusion Example

Shows how to fuse operations for performance:

```bash
# Default: 4096 size
jbang graph_fusion_example.java

# Custom size
jbang graph_fusion_example.java --size 8192

# Disable fusion for comparison
jbang graph_fusion_example.java --fuse false
```
