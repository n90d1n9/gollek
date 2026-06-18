


## Tokenizer Architecture

```text
SentencePiece C++ (libsentencepiece.so / .dylib)
        │
        ▼
FFM Binding (JDK 25)
        │
        ▼
SentencePieceTokenizer (Java, implements Tokenizer)
        │
        ▼
Gollek Tokenizer SPI
```

---

# ⚙️ 2. Native library requirement

You need SentencePiece compiled as shared lib:

### Linux / Mac

```bash
cd ~/.gollek/source/vendor/sentencepiece
git clone https://github.com/google/sentencepiece.git
cd sentencepiece
mkdir build && cd build
cmake .. -DSPM_ENABLE_SHARED=ON
make -j
```

### Output:

```bash
libsentencepiece.so   (Linux)
libsentencepiece.dylib (Mac)
```

---


### Compile (manual):

```bash
g++ -shared -fPIC spm_bridge.c -o libspm_bridge.so \
    -lsentencepiece
```

### Compile (recommended):

```bash
make -C src/main/native install
```

If SentencePiece is installed in a non-standard prefix, provide `SPM_PREFIX`:

```bash
make -C src/main/native install SPM_PREFIX=/path/to/sentencepiece
```

The installer copies the library to:

```
~/.gollek/native/libspm_bridge.(so|dylib)
```


---

### Usage

```bash
TokenizerPool pool = new TokenizerPool(
        Path.of("libspm_bridge.so"),
        Path.of("tokenizer.model")
);

var tokenizer = pool.current();

long[] tokens = tokenizer.encode("Hello world", EncodeOptions.defaultOptions());

String text = tokenizer.decode(tokens, DecodeOptions.defaultOptions());
```

### 🔥 Performance optimization (next step)

* Reuse handle per thread
* Avoid Arena per call (use scoped pool)
* Batch encode

---

# 🧠 Tokenizer supports:

| Model   | Tokenizer     | Status         |
| ------- | ------------- | -------------- |
| GPT-2   | BPE           | ✅              |
| LLaMA   | SentencePiece | ✅ (FFM native) |
| Mistral | SentencePiece | ✅              |
| T5      | SentencePiece | ✅              |

---


## Integration with Gollek Inference

```bash
package ai.golek.inference;

import ai.golek.tokenizer.runtime.TokenizerPool;

public class InferenceLoop {

    private final TokenizerPool tokenizerPool;

    public InferenceLoop(TokenizerPool tokenizerPool) {
        this.tokenizerPool = tokenizerPool;
    }

    public void run(String prompt) {

        var tokenizer = tokenizerPool.current();

        // 🔹 Encode (zero-copy)
        int len = tokenizer.encodeInto(prompt);

        for (int i = 0; i < len; i++) {
            int token = tokenizer.getToken(i);

            // 👉 SEND DIRECTLY TO LLM (llama.cpp / KV cache)
            processToken(token);
        }

        // 🔹 Simulate generation loop
        for (int i = 0; i < 10; i++) {
            int generated = fakeModelStep();

            // 🔹 Streaming decode
            String piece = tokenizer.decodeSingle(generated);

            System.out.print(piece); // streaming output
        }
    }

    private void processToken(int token) {
        // hook to model input
    }

    private int fakeModelStep() {
        return 42;
    }
}
```
