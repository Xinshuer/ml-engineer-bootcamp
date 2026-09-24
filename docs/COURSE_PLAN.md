# Course plan

26 days in 10 parts. Days 1–14 come from the classic terminal drills (`classic/`); days 15–26 are new.
Part names: Chinese / English (the `part:` line of each day file).

| Days | Part (zh / en) |
|---|---|
| 1–2 | 地基 / Foundations |
| 3–6 | 大语言模型 / LLMs |
| 7–10 | 图像生成 / Image generation |
| 11–13 | 音频 / Audio |
| 14 | 元技能 / Meta-skills |
| 15–17 | 调试与排错 / Debugging |
| 18–20 | 生产环境常见坑 / Production pitfalls |
| 21–23 | 部署 / Deployment |
| 24–25 | 上线后的运维 / Operations |
| 26 | 毕业项目 / Capstone |

## Days 1–14 (classic)

1 Python for model code · 2 shapes, hand-written layers, autograd · 3 tokenizers and batches · 4 nanoGPT ·
5 GPT-2 → Llama (RMSNorm, RoPE, SwiGLU, GQA, KV cache, sampling) · 6 LoRA, SFT, schedules, memory ·
7 VAE · 8 DDPM · 9 conditioning, CFG, DDIM · 10 DiT + flow matching · 11 audio shapes, STFT, mel ·
12 TTS · 13 audio codecs + LM · 14 from paper to code (smoke tests, overfit a batch, VQ).

## Days 15–26 (new)

Every day: 5–8 cards, 10–13 practice items (mix of code / fix / choice / predict / fill, plus a `self`
checklist where it fits), 1–2 closed-book (`exam: yes`) items. All runnable on a CPU in seconds.
Available packages: torch, numpy, onnx, onnxruntime, onnxscript, fastapi, pydantic, httpx (FastAPI TestClient),
safetensors, psutil, pytest. Not available: scipy, docker (only as a `local` item or as text).

**15 · 调试一：loss 不对劲 / Debugging 1: when the loss looks wrong.**
Reading loss curves (initial loss ≈ ln V, flat, exploding, NaN, oscillating); where NaN/inf come from (log 0,
exp overflow, division by 0, fp16 overflow, too high lr); `torch.isfinite`, `torch.autograd.detect_anomaly`;
gradient norms per layer; overfit one batch first; the "loss does not go down" checklist (lr, missing
`zero_grad`, frozen params, labels misaligned with inputs, softmax before `cross_entropy`).

**16 · 调试二：形状、设备、数据类型 / Debugging 2: shapes, devices and dtypes.**
Reading a PyTorch traceback (the last line + the last frame in your code); silent broadcasting bugs
((B,) vs (B,1)); `view` vs `reshape` / `contiguous`; device mismatches, `.to(x.device)`, `register_buffer`
moves with the module (test with a dtype change instead of a GPU); dtypes (long labels, float64 from numpy,
integer division); forward hooks that print every layer's output shape; moving nested batches to a device.

**17 · 调试三：不报错的 bug / Debugging 3: bugs that raise no error.**
`train()` vs `eval()` (dropout, batch norm); data leakage (statistics fitted on test data); off-by-one labels
(next-token shift); padding counted in the loss / mean; gradient accumulation without dividing; weight decay on
norms and biases; scheduler step order; catching silent bugs with invariant tests (causality, permutation
invariance, compare with a reference, overfit a batch).

**18 · 生产坑一：训练和推理不一致 / Pitfalls 1: train/serve skew.**
One preprocessing code path for training and serving (saved normalisation stats, tokenizer version);
`eval()` + `torch.inference_mode()`; left padding + attention mask + position ids for batched generation with
decoder-only models; results that change with batch size (batch norm in train mode, padding); determinism
(seeds for torch / numpy / random, `cudnn.benchmark`, deterministic algorithms); precision (fp32 / TF32 / bf16 /
fp16) and choosing tolerances.

**19 · 生产坑二：显存、内存与速度 / Pitfalls 2: memory and speed.**
Where memory goes (weights, gradients, optimizer states, activations) with worked numbers; leaks from keeping
tensors with graphs (`losses.append(loss)` vs `.item()`); `no_grad` / `inference_mode`; timing correctly
(warm-up, `torch.cuda.synchronize`, `perf_counter`, several runs, median); DataLoader (`num_workers`,
`pin_memory`, Windows needs `if __name__ == "__main__"`); `torch.compile` recompiles on changing shapes;
activation checkpointing trades compute for memory (`torch.utils.checkpoint`, same gradients).

**20 · 生产坑三：checkpoint 的坑 / Pitfalls 3: checkpoints.**
`state_dict` vs pickling the whole model; `torch.load(weights_only=True)` and why pickles are unsafe;
safetensors (and tied weights); `module.` / `_orig_mod.` prefixes from DDP / `torch.compile`; `strict=False`
hides bugs (check missing / unexpected keys); what a resumable checkpoint needs (model, optimizer, scheduler,
step, RNG states, EMA) and proving resume == uninterrupted; atomic save (temp file + `os.replace`).

**21 · 部署一：导出与压缩 / Deployment 1: export and compress.**
The inference checklist (eval, inference_mode, no dropout); `torch.export` / `torch.onnx.export(dynamo=True)`
(static graphs, data-dependent control flow, dynamic batch dimension); always compare the export with PyTorch
(onnxruntime, tolerances); dynamic int8 quantization of Linear layers (size, speed, accuracy trade-off);
fp16 / bf16 inference; common export failures and fixes (`torch.where` instead of Python `if` on tensor values).

**22 · 部署二：把模型做成服务 / Deployment 2: serving a model.**
Service structure (load once at start-up, warm-up, health check); input validation with pydantic (422 on bad
input); FastAPI + TestClient; micro-batching (collect up to N requests or T ms; simulate time); one model per
process / GPU and a queue; timeouts, limits, error codes; latency percentiles p50/p95/p99 and throughput vs batch
size.

**23 · 部署三：生成式模型的推理服务 / Deployment 3: serving generative models.**
LLM inference = prefill + decode; KV-cache memory arithmetic; streaming tokens (generators); stop conditions
(eos, max_new_tokens, stop strings); validating sampling parameters; continuous batching (simulate a scheduler);
diffusion: steps vs latency; a Dockerfile for a GPU inference image (pin versions, model files outside the image,
as a `fill` item on the text + an optional `local` item).

**24 · 运维一：实验可复现、模型可追溯 / Operations 1: reproducible runs, traceable models.**
What to record per run (config, git commit, data fingerprint, seeds, library versions, metrics); comparing runs;
a small model registry (register, promote, roll back; JSON in a temp dir); model cards.

**25 · 运维二：上线后的监控 / Operations 2: monitoring in production.**
What to watch (latency, error rate, throughput, GPU use, cost); input drift (PSI, a KS statistic written by
hand), prediction drift; shadow traffic, canary and A/B decisions (minimum sample sizes); alert rules on rolling
windows; cost per 1,000 requests.

**26 · 毕业项目：从训练到上线 / Capstone: from training to production.**
One small model end to end, one step per item, each item's setup giving the previous steps' reference code:
train a tiny model on CPU in seconds (with the day-14 checks: smoke test, overfit one batch) → resumable
checkpoint with safetensors → export to ONNX and verify with onnxruntime → FastAPI service with validation,
warm-up and batching → latency percentiles and a drift check → a final `local` item that runs the whole pipeline
script on the learner's machine (GPU if present) and pastes the summary.
