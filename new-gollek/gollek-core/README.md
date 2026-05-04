


TOP-LEVEL VIEW
┌──────────────────────────────┐
│ USER / DEV │
│ (Notebook, API, SDK, CLI) │
└─────────────┬────────────────┘
│
▼
┌──────────────────────────────┐
│ MODEL INGESTION │
│ (Plugins: ONNX, GGUF, etc.) │
└─────────────┬────────────────┘
│
▼
┌──────────────────────────────┐
│ GOLLEK IR │ ✅ (DONE)
│ (Graph, Ops, Values, Types)│
└─────────────┬────────────────┘
│
▼
┌──────────────────────────────┐
│ COMPILER │ 🔥 (CURRENT)
│ (Passes, Fusion, Lowering) │
└─────────────┬────────────────┘
│
▼
┌──────────────────────────────┐
│ EXECUTION PLANNER │ 🚀 (NEXT)
│ (Scheduling, Memory Plan) │
└─────────────┬────────────────┘
│
▼
┌──────────────────────────────┐
│ RUNTIME │ ✅ (PARTIAL)
│ (ExecutionEngine, KV Cache) │
└─────────────┬────────────────┘
│
▼
┌──────────────────────────────┐
│ BACKEND / DEVICES │
│ CPU / GPU / Metal / Remote │
└──────────────────────────────┘



FULL INTERNAL LAYERS
🔵 A. M
ODEL INGESTION (PLUGIN WO
R
LD)
[ safetensors ] [ gguf ] [ onnx ] [ torch ]
│ │ │ │
└─────── ModelLoader Plugins ───────┘
│
▼
GGraph + WeightStore



# IR LAYER  GGraph
├── GOp (typed, validated)
├── GValue (immutable, storageKey)
├── GAttrValue (typed attrs)
└── GType / GShape


✔ format-agnostic
✔
plugin-safe
✔
compiler-ready




Mask generation (IR level)
OpDescriptor CAUSAL_MASK = new OpDescriptor(
new OpId("nn", "causal_mask", 1)
);


OpDescriptor MASKED_ATTENTION = new OpDescriptor(
new OpId("nn", "masked_attention", 1)
);