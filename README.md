---
license: apache-2.0
library_name: mlx
base_model: chaoliangUNSW/MacJev-322M-4K-Laya
language:
- en
- zh
pipeline_tag: text-classification
tags:
- mlx
- apple-silicon
- macos
- jev
- laya
- modernbert
- decision-model
- tool-routing
- local-agents
---

# MacJev-322M-4K-Laya-MLX

**MacJev running natively on Apple silicon with MLX. No PyTorch, exact FP32 results.**

[MacJev-322M-4K-Laya](https://huggingface.co/chaoliangUNSW/MacJev-322M-4K-Laya) is a compact decision model for local Mac agents. You give it an observed state and one typed question with candidate answers. In a single forward pass it returns a probability for every candidate. Use it to rank candidate actions, route tool calls and check task state.

This repository contains the complete model and `macjev_mlx.py`, a native MLX runtime for all of it: the mmBERT/ModernBERT encoder, type embeddings, decision transformer, candidate scorer and tokenizer.

![MacJev vs. the Laya multilingual checkpoint it starts from](assets/macjev_vs_laya_multilingual.png)

## Highlights

Compared with the Laya multilingual checkpoint it was trained from, with the same 4096-token input budget, on held-out test sets:

| Task | Laya multilingual | MacJev |
|---|---:|---:|
| Yes/no checks on 2K–4K-token inputs | 31.0% | **89.1%** |
| Rule decisions on 2K–4K-token inputs | 23.7% | **44.7%** |
| Rule decisions on long inputs, all lengths | 23.2% | **39.7%** |
| Typed decisions | 35.2% | **42.2%** |
| Calibration error on long inputs, lower is better | 0.253 | **0.032** |

- **Reads long observations.** Yes/no task-state checks on 2K–4K-token inputs rise from 31% to 89% accuracy.
- **Probabilities you can threshold.** Calibration error on long inputs is about 8 times lower, so stated confidence closely tracks actual accuracy.
- **Better on public benchmarks too.** Across 11,600 public decisions from typed decisions, Emotion and AG News, accuracy rises by 1.5 points, with a 95% interval of 1.1 to 1.8.
- **Reproducible.** The long-input and public-benchmark gains appear in all three independently trained seeds. This release is the seed chosen in advance on development data.

## Exact and fast on a Mac

- **Same answers as the reference.** On all 9,000 validation decisions, the MLX runtime picks the same top answer as the PyTorch FP32 model, with probabilities within 0.00026.
- **Light dependencies.** Only `mlx` and `tokenizers`; no PyTorch or Transformers.
- **Checked on load.** SHA-256 checksums of every file it loads are verified in about 2 seconds.

Forward-pass time on an M1 Max with MLX 0.32.2:

| Input length | Median | 90th percentile |
|---|---:|---:|
| up to 1024 tokens | 0.05 s | 0.15 s |
| 1025 to 2048 tokens | 0.22 s | 0.34 s |
| 2049 to 4096 tokens | 0.54 s | 0.85 s |

## Quickstart

Requirements: Apple silicon Mac, Python 3.11 or newer.

```bash
hf download chaoliangUNSW/MacJev-322M-4K-Laya-MLX --local-dir MacJev-MLX
cd MacJev-MLX
python -m pip install -r requirements.txt
```

```python
from macjev_mlx import MacJevMLX

model = MacJevMLX(".")  # verifies checksums, loads the FP32 weights
result = model.decide(
    state={
        "request": "打开浏览器",
        "available_actions": ["open_browser", "copy_file", "ask_user"],
    },
    question={
        "t": "choice",
        "ins": "Which available action best matches the user's request?",
        "crit": {
            "open_browser": "Open the user's browser",
            "copy_file": "Copy a file inside the approved folder",
            "ask_user": "Ask for clarification",
        },
    },
)
print(result["answer"], result["probabilities"])  # A decision only. Nothing is executed.
```

Or download from Python, fetching only the files it needs:

```python
model = MacJevMLX.from_pretrained("chaoliangUNSW/MacJev-322M-4K-Laya-MLX")
```

The result contains `answer`, `probabilities`, `top_probability`, `input_tokens`, `latency_ms`, and `actions_executed: False`. Keep one model object loaded across requests.

Other question shapes:

```python
score = {"t": "score", "ins": "How complete is the observed task?",
         "crit": ["Not started", "Partly complete", "Complete and verified"]}
boolean = {"t": "noul", "ins": "Does the observation prove that the requested file exists?"}
```

Score answers are ordered string indices (`"0"`, `"1"`, ...). Noul answers are `"false"` and `"true"`. Choice keys keep their insertion order.

Command line, one JSON object per line with `state` and `question`:

```bash
echo '{"state": {"request": "open the browser"}, "question": {"t": "noul", "ins": "Does the user want a browser opened?"}}' \
  | python macjev_mlx.py --model .
```

Inputs can use up to 4096 tokens in total and 1024 for the question and options. Anything larger raises `InputBudgetError` instead of being truncated, so no candidate or evidence is dropped without you knowing.

### Using it with LM Studio

MacJev is a decision model rather than a chat model, so it runs through `macjev_mlx.py` instead of LM Studio's chat engine. To let an LM Studio chat model use it, expose `decide()` as a tool or MCP server and have the chat model call it.

## Files

| File | Purpose |
|---|---|
| `model.safetensors` | All 322M parameters in FP32, 1.29 GB, identical to the FP32 release |
| `macjev_mlx.py` | Native MLX runtime: `MacJevMLX` class and a command-line interface |
| `macjev_inputs.py` | Tokenization and input layout with strict 4096/1024 budgets |
| `encoder/config.json`, `rl_agent_config.json` | Architecture and serving configuration |
| `tokenizer/` | The mmBERT tokenizer |
| `manifest.json` | SHA-256 of every file and the calibration temperatures |
| `validation.json` | Measured agreement with the FP32 reference model |
| `LICENSE`, `NOTICE`, `MMBERT_LICENSE` | License and attribution |

## Other formats

- [PyTorch FP32](https://huggingface.co/chaoliangUNSW/MacJev-322M-4K-Laya), with training details
- [GGUF for llama.cpp](https://huggingface.co/chaoliangUNSW/MacJev-322M-4K-Laya-GGUF)

## License and attribution

Apache-2.0, inherited from Laya; see `LICENSE` and `NOTICE`. The mmBERT-base backbone is MIT-licensed; see `MMBERT_LICENSE`. MacJev is an independent project. It is not affiliated with or endorsed by JEV or the Laya authors, and it contains no JEV weights.

- [Laya model](https://huggingface.co/convaiinnovations/laya) and [source](https://github.com/NandhaKishorM/laya)
- [mmBERT-base](https://huggingface.co/jhu-clsp/mmBERT-base)

## Community

Interested in compact decision models, local agents, and practical macOS automation? Join our [Discord community](https://discord.gg/udfvMu2GM).

如果你也对轻量决策模型、本地 Agent 和 macOS 自动化感兴趣，欢迎加入我们的 [Discord 社区](https://discord.gg/udfvMu2GM)。
