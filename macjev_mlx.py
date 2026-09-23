"""MacJev native MLX runtime for Apple silicon.

MacJev ranks the candidates of one typed question in a single bidirectional
forward pass. It returns probabilities; it never executes Mac actions.

The precisions available are declared in ``manifest.json``. This release ships
FP32 weights (``model.safetensors``); smaller FP16 and 8-bit variants did not
pass the release's pre-registered drift gates and are not included. All
arithmetic, including the encoder, decision head, scorer and softmax, is FP32.
The loader can also read manifest-declared MLX affine-quantized weights.

Derived from Apache-2.0 Laya components and the MIT-licensed mmBERT/ModernBERT
architecture. See LICENSE, NOTICE and MMBERT_LICENSE. Dependencies: mlx and
tokenizers (huggingface-hub only for ``from_pretrained``). No PyTorch.
"""
import json
import time
from pathlib import Path, PurePosixPath

import mlx.core as mx

from macjev_inputs import QTYPES, InputBudgetError, TokenizerAdapter, build_sequence, file_hash

__all__ = ["MacJevMLX", "InputBudgetError"]

PARAMETERS = 321908995
REPO = "chaoliangUNSW/MacJev-322M-4K-Laya-MLX"


class MacJevMLX:
    """Load one precision of the MLX release from a local directory."""

    def __init__(self, model_directory=".", precision="fp32", verify=True):
        self.directory = Path(model_directory).expanduser().resolve(strict=True)
        self.precision = precision
        self.dtype = mx.float32
        manifest = json.loads((self.directory / "manifest.json").read_text())
        if precision not in manifest["precisions"]:
            raise ValueError(f"precision must be one of {sorted(manifest['precisions'])}")
        weights_file = manifest["precisions"][precision]["file"]
        needed = ["encoder/config.json", "rl_agent_config.json", "tokenizer/tokenizer.json",
                  "tokenizer/tokenizer_config.json", weights_file]
        if verify:
            for relative in needed:
                # Manifest paths must stay inside the package. Files may be symlinks
                # (the Hugging Face cache stores snapshots as links to blobs); the
                # SHA-256 is computed over the linked file's content.
                path = PurePosixPath(relative)
                expected = manifest["files_sha256"].get(relative)
                if expected is None or path.is_absolute() or ".." in path.parts:
                    raise ValueError("Invalid model manifest entry: " + relative)
                target = self.directory / relative
                if not target.is_file() or file_hash(target) != expected:
                    raise ValueError("Model package integrity mismatch: " + relative)
        self.encoder_config = json.loads((self.directory / "encoder/config.json").read_text())
        self.config = json.loads((self.directory / "rl_agent_config.json").read_text())
        self.max_len, self.head_max_len = self.config["max_len"], self.config["head_max_len"]
        if (self.max_len, self.head_max_len) != (4096, 1024):
            raise ValueError("Unexpected model input budget")
        self.temperatures = manifest["temperatures"]
        if self.config["temperature"] != [self.temperatures[k] for k in QTYPES]:
            raise ValueError("Serving configuration and temperatures differ")
        self._check_architecture()
        self.quantization = manifest["precisions"][precision].get("quantization")
        self.w = self._load(self.directory / weights_file)
        self.tokenizer = TokenizerAdapter((self.directory / "tokenizer/tokenizer.json").read_text(),
                                          json.loads((self.directory / "tokenizer/tokenizer_config.json").read_text()))
        c = self.encoder_config
        self._local_window = c["local_attention"] // 2
        self._theta = {kind: float(v["rope_theta"]) for kind, v in c["rope_parameters"].items()}
        self._eps = c["norm_eps"]

    @classmethod
    def from_pretrained(cls, repo_id=REPO, precision="fp32", revision=None, **kwargs):
        """Download only the files this precision needs from the Hugging Face Hub."""
        from huggingface_hub import hf_hub_download, snapshot_download
        manifest = json.loads(Path(hf_hub_download(repo_id, "manifest.json", revision=revision)).read_text())
        if precision not in manifest["precisions"]:
            raise ValueError(f"precision must be one of {sorted(manifest['precisions'])}")
        patterns = ["manifest.json", "encoder/config.json", "rl_agent_config.json", "tokenizer/*",
                    manifest["precisions"][precision]["file"]]
        path = snapshot_download(repo_id, revision=revision, allow_patterns=patterns)
        return cls(path, precision=precision, **kwargs)

    def _check_architecture(self):
        c, a = self.encoder_config, self.config
        shape = (c["hidden_size"], c["num_attention_heads"], c["num_hidden_layers"], c["intermediate_size"], a["head_layers"])
        if shape != (768, 12, 22, 1152, 2):
            raise ValueError("Unsupported MacJev architecture configuration")
        if c["hidden_activation"] != "gelu" or c["attention_bias"] or c["mlp_bias"] or c["norm_bias"]:
            raise ValueError("Unsupported encoder activation/bias configuration")
        if len(c["layer_types"]) != 22 or set(c["layer_types"]) != {"full_attention", "sliding_attention"}:
            raise ValueError("Invalid hybrid attention pattern")
        if any(v["rope_type"] != "default" for v in c["rope_parameters"].values()):
            raise ValueError("Unsupported RoPE scaling")

    def _load(self, path):
        raw = mx.load(str(path))
        quantized = {k[: -len(".scales")] for k in raw if k.endswith(".scales")}
        total = 0
        for name, value in raw.items():
            if name == "temperature" or (name.endswith((".scales", ".biases")) and name.rsplit(".", 1)[0] in quantized):
                continue
            if name.rsplit(".", 1)[0] in quantized and name.endswith(".weight"):
                total += value.shape[0] * value.shape[1] * 32 // self.quantization["bits"]
            else:
                total += value.size
        if total != PARAMETERS:
            raise RuntimeError(f"Unexpected parameter count {total}")
        weights = {}
        for name, value in raw.items():
            prefix = name.rsplit(".", 1)[0]
            if prefix in quantized and name.endswith(".weight"):
                weights[name] = value  # packed uint32
            elif prefix.startswith("encoder."):
                weights[name] = value.astype(self.dtype)
            else:  # decision head, scorer, act_head, type embeddings: FP32 arithmetic
                weights[name] = value.astype(mx.float32)
        mx.eval(weights)
        self._quantized = quantized
        return weights

    # ---- primitives -------------------------------------------------------
    def _linear(self, x, prefix):
        w = self.w
        if prefix in self._quantized:
            q = self.quantization
            y = mx.quantized_matmul(x, w[prefix + ".weight"], w[prefix + ".scales"].astype(x.dtype),
                                    w[prefix + ".biases"].astype(x.dtype), transpose=True,
                                    group_size=q["group_size"], bits=q["bits"])
        else:
            y = x @ w[prefix + ".weight"].T
        bias = w.get(prefix + ".bias")
        return y if bias is None else y + bias

    def _norm(self, x, prefix):
        return mx.fast.layer_norm(x, self.w[prefix + ".weight"], self.w.get(prefix + ".bias"), self._eps)

    def _embed(self, ids):
        name = "encoder.embeddings.tok_embeddings"
        if name in self._quantized:
            q = self.quantization
            rows = mx.dequantize(self.w[name + ".weight"][ids], self.w[name + ".scales"][ids],
                                 self.w[name + ".biases"][ids], group_size=q["group_size"], bits=q["bits"])
            return rows.astype(self.dtype)
        return self.w[name + ".weight"][ids]

    @staticmethod
    def _gelu(x):
        return x * (1 + mx.erf(x / (2.0 ** 0.5))) / 2

    @staticmethod
    def _split_heads(x):
        return x.reshape(1, x.shape[1], 12, 64).transpose(0, 2, 1, 3)

    @staticmethod
    def _merge_heads(x):
        return x.transpose(0, 2, 1, 3).reshape(1, x.shape[2], 768)

    # ---- model ------------------------------------------------------------
    def logits_from_ids(self, ids, markers, qtype_index):
        """Uncalibrated candidate logits for one prepared sequence (FP32 list)."""
        if not 1 <= len(ids) <= self.max_len or not markers:
            raise InputBudgetError("Prepared input is empty or exceeds 4096 tokens")
        c = self.encoder_config
        h = self._embed(mx.array(ids, dtype=mx.int32))[None]
        h = self._norm(h, "encoder.embeddings.norm")
        positions = mx.arange(len(ids))
        local = (mx.abs(positions[:, None] - positions[None, :]) <= self._local_window)[None, None]
        for i, kind in enumerate(c["layer_types"]):
            prefix = f"encoder.layers.{i}"
            x = h if i == 0 else self._norm(h, prefix + ".attn_norm")
            q, k, v = (self._split_heads(t) for t in mx.split(self._linear(x, prefix + ".attn.Wqkv"), 3, axis=-1))
            theta = self._theta[kind]
            q = mx.fast.rope(q, 64, traditional=False, base=theta, scale=1.0, offset=0)
            k = mx.fast.rope(k, 64, traditional=False, base=theta, scale=1.0, offset=0)
            mask = local if kind == "sliding_attention" else None
            attention = mx.fast.scaled_dot_product_attention(q, k, v, scale=64 ** -0.5, mask=mask)
            h = h + self._linear(self._merge_heads(attention), prefix + ".attn.Wo")
            x = self._norm(h, prefix + ".mlp_norm")
            value, gate = mx.split(self._linear(x, prefix + ".mlp.Wi"), 2, axis=-1)
            h = h + self._linear(self._gelu(value) * gate, prefix + ".mlp.Wo")
            mx.eval(h)
        h = self._norm(h, "encoder.final_norm").astype(mx.float32)
        h = h + self.w["type_emb.weight"][int(qtype_index)][None, None, :]
        for i in range(self.config["head_layers"]):
            prefix = f"head.layers.{i}"
            x = self._norm(h, prefix + ".norm1")
            qkv = x @ self.w[prefix + ".self_attn.in_proj_weight"].T + self.w[prefix + ".self_attn.in_proj_bias"]
            q, k, v = (self._split_heads(t) for t in mx.split(qkv, 3, axis=-1))
            attention = mx.fast.scaled_dot_product_attention(q, k, v, scale=64 ** -0.5)
            h = h + self._linear(self._merge_heads(attention), prefix + ".self_attn.out_proj")
            x = self._norm(h, prefix + ".norm2")
            h = h + self._linear(mx.maximum(self._linear(x, prefix + ".linear1"), 0), prefix + ".linear2")
            mx.eval(h)
        hidden = h[0, mx.array(markers, dtype=mx.int32)]
        hidden = self._gelu(self._linear(self._norm(hidden, "scorer.0"), "scorer.1"))
        logits = self._linear(hidden, "scorer.3")[:, 0]
        mx.eval(logits)
        return logits.tolist()

    def decide(self, state, question):
        started = time.monotonic()
        ids, markers = build_sequence(self.tokenizer, state, question, self.max_len, self.head_max_len)
        logits = mx.array(self.logits_from_ids(ids, markers, QTYPES[question["t"]]))
        probs = mx.softmax(logits / self.temperatures[question["t"]]).tolist()
        names = (list(question["crit"]) if question["t"] == "choice" else
                 [str(i) for i in range(len(question["crit"]))] if question["t"] == "score" else ["false", "true"])
        best = max(range(len(probs)), key=probs.__getitem__)
        return {"answer": names[best], "probabilities": dict(zip(names, probs)), "top_probability": probs[best],
                "input_tokens": len(ids), "latency_ms": (time.monotonic() - started) * 1000,
                "model": "MacJev-322M-4K-Laya", "backend": "mlx", "precision": self.precision,
                "compute": "float32", "actions_executed": False}


if __name__ == "__main__":
    import argparse
    import sys
    parser = argparse.ArgumentParser(description="One JSON object per line with state and question. No Mac actions are executed.")
    parser.add_argument("--model", default=".", help="local directory of the MLX release")
    parser.add_argument("--precision", default="fp32")
    args = parser.parse_args()
    model = MacJevMLX(args.model, args.precision)
    for line in sys.stdin:
        if line.strip():
            request = json.loads(line)
            try:
                result = model.decide(request["state"], request["question"])
            except InputBudgetError as error:
                result = {"error": "input_budget", "message": str(error), "actions_executed": False}
            print(json.dumps(result, ensure_ascii=False), flush=True)
