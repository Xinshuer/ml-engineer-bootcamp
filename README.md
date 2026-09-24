# ML Engineer Bootcamp

**English** · [中文](#中文)

A 26-day interactive course that teaches you to build, debug, ship and operate deep-learning models in
PyTorch, by writing them yourself: a GPT, Llama's upgrades, LoRA, a VAE, diffusion (DDPM, DDIM, DiT, flow
matching), text-to-speech and audio codecs; then debugging, the pitfalls that bite in production, exporting and
serving models, and monitoring them. Every exercise is graded by real PyTorch on your own machine.

## Start

```bash
pip install -r requirements.txt     # install torch for your machine first: https://pytorch.org/get-started/locally/
python serve.py                     # opens http://127.0.0.1:8765
```

The page looks like a website, but everything runs locally: `serve.py` listens only on 127.0.0.1 and runs your
code in a PyTorch process on your computer (GPU if you have one; every exercise also runs on a CPU). Your
progress is saved in `progress.json` in this folder. Switch between Chinese and English at the top right.

Prefer your editor? Every coding exercise has a **Work on it in VS Code** button: it saves the exercise to
`workspace/<id>.py`; run `python workspace/<id>.py` to check it with the same checks as the page.

## What is in it

| Days | Part |
|---|---|
| 1–2 | Foundations: the Python that model code uses; shapes, hand-written layers, autograd |
| 3–6 | LLMs: tokenizers, nanoGPT, Llama (RMSNorm, RoPE, SwiGLU, GQA, KV cache), LoRA and SFT |
| 7–10 | Image generation: VAE, DDPM, conditioning and CFG, DDIM, DiT, flow matching |
| 11–13 | Audio: STFT and mel, TTS, audio codecs + a language model |
| 14 | Meta-skills: from a paper to working code |
| 15–17 | Debugging: when the loss looks wrong; shapes, devices, dtypes; bugs that raise no error |
| 18–20 | Production pitfalls: train/serve skew; memory and speed; checkpoints |
| 21–23 | Deployment: export and quantization; serving a model; serving generative models |
| 24–25 | Operations: reproducible runs and a model registry; monitoring and drift |
| 26 | Capstone: one model from training to a monitored service |

Each day has knowledge cards with runnable examples, practice exercises (write code, fix bugs, fill in, order
lines, predict output, choose), and a closed-book challenge. Exercises where you needed the answer go to a
review list until you solve them again on your own.

## Folders

| Folder | What it is |
|---|---|
| `content/` | the course: `dayN.txt` (Chinese), `en/dayN.txt` (English), `ref/dayN.py` (reference code used by the checks) |
| `app/` | the page (`static/`), the code runner, the local server's helpers |
| `tools/build.py` | validates every exercise with real PyTorch (answers pass, starters fail) and builds the page's data |
| `docs/` | how to write content (`AUTHORING.md`) and the course plan |
| `classic/` | the original terminal version of days 1–14 (`python classic/check.py`) |

To change the course, edit `content/`, then run `python tools/build.py` (see `docs/AUTHORING.md`).

---

<a id="中文"></a>

# ML 工程师训练营

[English](#ml-engineer-bootcamp) · **中文**

26 天的互动课程：用 PyTorch 亲手写出、调试、上线和运维深度学习模型。先手写 GPT、Llama 的改进、LoRA、VAE、扩散模型（DDPM、DDIM、DiT、flow matching）、语音合成和音频 codec；再学调试、生产环境里常见的坑、模型导出和推理服务、上线后的监控。每道题都用你电脑上真正的 PyTorch 判题。

## 开始

```bash
pip install -r requirements.txt     # 先按你的电脑装好 torch：https://pytorch.org/get-started/locally/
python serve.py                     # 自动打开 http://127.0.0.1:8765
```

页面看起来像网站，其实全在本机运行：`serve.py` 只监听 127.0.0.1，你的代码在本机的 PyTorch 进程里跑（有显卡就用显卡；每道题在 CPU 上也能跑）。进度保存在这个文件夹的 `progress.json` 里。右上角可以切换中文和英文。

想用编辑器写？每道写代码的题都有「在 VS Code 里做」按钮：它把题目存到 `workspace/<题号>.py`，改完运行 `python workspace/<题号>.py`，用和页面一样的检查判题。

## 内容

| 天 | 部分 |
|---|---|
| 1–2 | 地基：写模型用得上的 Python；shape、手写各层、自动求导 |
| 3–6 | 大语言模型：tokenizer、nanoGPT、Llama（RMSNorm、RoPE、SwiGLU、GQA、KV cache）、LoRA 与 SFT |
| 7–10 | 图像生成：VAE、DDPM、条件化与 CFG、DDIM、DiT、flow matching |
| 11–13 | 音频：STFT 与 mel、语音合成、音频 codec + 语言模型 |
| 14 | 元技能：把论文变成能跑的代码 |
| 15–17 | 调试与排错：loss 不对劲；形状、设备、数据类型；不报错的 bug |
| 18–20 | 生产环境常见坑：训练和推理不一致；显存、内存与速度；checkpoint |
| 21–23 | 部署：导出与量化；把模型做成服务；生成式模型的推理服务 |
| 24–25 | 上线后的运维：实验可复现与模型登记；监控与漂移 |
| 26 | 毕业项目：一个模型从训练走到有监控的线上服务 |

每天有带可运行例子的知识卡片、练习题（写代码、修 bug、填空、排代码、预测输出、选择）和闭卷挑战。看过答案的题会进复习区，直到你不看答案再做对一次。

## 文件夹

| 文件夹 | 内容 |
|---|---|
| `content/` | 课程内容：`dayN.txt`（中文）、`en/dayN.txt`（英文）、`ref/dayN.py`（判题用的参考代码） |
| `app/` | 页面（`static/`）、运行代码的进程、本地服务器的辅助代码 |
| `tools/build.py` | 用真 PyTorch 验证每道题（参考答案必须通过、起始代码必须不通过），并生成页面数据 |
| `docs/` | 怎么写内容（`AUTHORING.md`）和课程大纲 |
| `classic/` | 第 1–14 天原来的终端版（`python classic/check.py`） |

要改课程，就改 `content/`，然后运行 `python tools/build.py`（见 `docs/AUTHORING.md`）。
