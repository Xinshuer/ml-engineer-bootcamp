# Writing course content

Everything the page shows comes from plain-text files:

```
content/day4.txt        the Chinese course for day 4
content/en/day4.txt     the English course for day 4 (same blocks, same ids, same order; only the words differ)
content/ref/day4.py     reference implementations that exercises can use as hidden setup
```

After editing, validate the day (both languages, with real PyTorch):

```bash
python tools/build.py --day 4          # validates day 4 only, writes nothing
python tools/build.py                  # validates everything and writes app/static/course.js (+ course-en.js)
python tools/build.py --show d4-02     # print what the runner returned for one item
```

`BUILD_WORKERS=2` (environment variable) limits the number of parallel runner processes (default 6).

## File format

A file is a sequence of blocks. Each block starts with `@@ <kind> [id]`, then `key: value` lines, then
`--- section` lines that start multi-line sections (everything up to the next `--- ` line or block).

### The day block (first block of every file)

```
@@ day 4
title: nanoGPT，一天手写一个 Transformer
part: 大语言模型                     (English file: LLMs; days with the same part are grouped in the menu)
hours: 3–4 小时
--- goal
One or two sentences: what you can do after today.
--- checklist
- 先清空复习区（昨天看过答案的题从空白重做）
- ...
```

### Knowledge cards (`@@ concept`)

Cards are the teaching part (the "学" tab). Write them before the items; items link to them.

```
@@ concept
title: 因果 mask：只能看过去
--- body
Plain explanation. Markdown subset: paragraphs, `- ` lists, `1. ` lists, ```` ``` ```` code fences,
`### heading`, `inline code`, **bold**. No tables, no links, no images.
--- code
import torch
print(torch.tril(torch.ones(3, 3)))
```

- `--- code` is optional. It runs on the learner's machine when they press Run (setup = the prelude below,
  plus `ref:` modules if the card has a `ref:` line). It must finish in a second or two on a CPU and print
  something worth looking at.
- `error_demo: yes` on a card whose code is *meant* to raise (shows a real error message).
- Card ids are automatic (`d4-c1`, `d4-c2`, ...), so the number and order of cards must match in both languages.

### Items (`@@ item dN-NN`)

Ids are `d<day>-<two digits>`, numbered in order. Common fields:

```
@@ item d4-02
type: code                 code | fix | fill | order | choice | predict | self | local
title: 手写 attention，对拍 SDPA
level: 2                   1 easy · 2 normal · 3 hard
cards: 缩放点积注意力; 因果 mask：只能看过去      (titles of cards on the same day; another day: d3:Title)
targets: attention         names the learner defines (code/fix): removed from the hidden setup
ref: day4                  reference modules (content/ref/day4.py) loaded as hidden setup
timeout: 60                seconds (default 60); keep real runs far below it
exam: yes                  closed-book item: shown only on the 闭卷 tab, no hints/answers
--- context
1–3 sentences: why this matters in real ML work (max 280 characters in Chinese, 600 in English).
--- prompt
What to do, precisely: signature, shapes, constraints ("不许用 F.softmax").
--- starter
code the editor starts with (use `raise todo()` where the learner writes)
--- solution
the reference answer
--- setup
optional extra hidden code for this item (runs after the ref modules)
--- tests
@check("causal mask 的方向")
def _():
    m = causal_bias(4)
    shape_is(m, (1, 1, 4, 4))
    true(m[0, 0, 0, 1] == float("-inf"), "第 0 个位置不能看到第 1 个 —— 你的 mask 反了")
--- hints
- first hint (small nudge)
- second hint (almost the answer)
```

Type-specific sections:

| type | what the learner does | needs |
|---|---|---|
| `code` | writes a function / class | starter, solution, tests, targets |
| `fix` | fixes buggy code | starter (the buggy code), solution, tests, targets |
| `fill` | fills blanks in code | `--- template` with `[[answer]]` blanks, tests. A blank ends at the first `]]`, so an answer cannot end in `]` (write `x.unsqueeze(1)`, not `x[:, None]`); the build rejects blanks with unbalanced brackets |
| `order` | puts shuffled lines in order | `--- lines` (correct order), optional `--- distractors`, tests |
| `choice` | picks option(s) | `--- options` (`- ` list), `answer: 2` (1-based; `answer: 1,3` = several), `--- explain` |
| `predict` | writes what the code prints | `--- code` (the page runs it at build time to get the expected output), `--- explain` |
| `self` | ticks a checklist honestly (dictation, "ran it on the GPU") | `--- checklist` (`- ` list) |
| `local` | does something outside the page (Docker, a terminal) and pastes the output | `--- prompt` with steps, `verify: <regex the pasted output must match>`, optional `placeholder:` |

### Sample data and "try it" (`--- above`, `--- below`)

Every `code` and `fix` item (closed-book ones too) has them, so the editor reads like a real script instead of
a bare signature:

- `--- above` goes before the starter: a first comment line saying where the input comes from in a real model
  or project (`# 上文：…` / `# Where the input comes from: …`), then a small sample of it with exactly the
  shapes, dtypes and keys the tests use (`x = torch.randn(2, 5, 12)  # 2 句话，每句 5 个 token…`).
- `--- below` goes after it, under a "try it" line the build adds: call the learner's function(s) on the
  sample and print something readable (shapes, a few rounded values, `.tolist()` of a small tensor).

The build joins them to the starter and the solution and writes under the call what the reference solution
prints ("写对了会打印：…"). `app/runner.py` grades only the code above the try-it line and runs the try-it
part after the checks (seeds reset first), reporting its output and errors on their own: a function that is
still `raise todo()` shows "not written yet" there and never changes the verdict. The build rejects a try-it
part that fails with the reference solution or prints nothing, output longer than 12 lines × 110
characters, output that differs between two runs, and an `above`/`below` that defines the learner's function.
`--show <id>` prints the runs, including the try-it output.

## How an exercise runs

The server runs, in one fresh namespace:

1. **prelude**: `import math, torch, torch.nn as nn, torch.nn.functional as F`, plus the check helpers
2. **ref modules** listed in `ref:` (e.g. the whole of `content/ref/day4.py`), then the item's `--- setup`
3. the **targets are deleted** again (so an empty answer fails, but a reference class that *inherits* from a
   target still exists)
4. the **learner's code**
5. the **tests**: every `@check("title")` function runs; each is one line on the page (✓ / ✗ / – not written)

Check helpers (available in tests, setup and learner code):

| helper | meaning |
|---|---|
| `@check("title")` | registers a check |
| `true(cond, msg)` | fails with `msg` when `cond` is false |
| `eq(got, want, tol=1e-4, msg="")` | tensors / numbers: shape first, then max absolute difference |
| `shape_is(t, shape, msg="")` | shape check |
| `close(a, b, tol=1e-4)` | returns a bool |
| `seed(n)` | `torch.manual_seed(n)` |
| `params(m)`, `trainable(m)` | parameter counts |
| `raise todo()` | "not written yet" (shown as –, not ✗). It raises a `BaseException`, not an `Exception`, so a test may wrap the learner's call in `except Exception` / `except RuntimeError` without swallowing it |

Rules for tests:

- Deterministic: call `seed(...)` before random data. Every run and every check starts with `torch.manual_seed(0)`,
  `random.seed(0)` and (if numpy is loaded) `numpy.random.seed(0)`, and with the cudnn / TF32 switches back at
  their defaults.
- Tests that swap out a function (`torch.save`, `F.softmax`, ...) must put it back in `finally`: the worker process is
  reused for the next run.
- Fast: the whole item must run in a few seconds **on a CPU** (learners without a GPU must be able to do the course).
  Use tiny sizes. A check may use CUDA only if it also passes when `torch.cuda.is_available()` is False.
- Failure messages teach: say what is wrong and the likely cause ("改了未来的 k/v，前 4 个位置的输出也变了 —— 信息泄漏了").
- Messages are in the file's language (Chinese in `content/`, English in `content/en/`).
- Files: write only under `tempfile.mkdtemp()` / `tempfile.TemporaryDirectory()`; never into the course folder.
- No network access in tests.

The build checks, for every runnable item and in both languages: the solution passes every check, the starter
fails at least one, and (when there are `targets:`) an empty answer fails.

## Writing style

The learner is new to programming and to ML (they started coding a few weeks ago). So:

- Short sentences. One idea per sentence. Explain a term the first time it appears ("logits（模型对每个词打的原始分数，还没过 softmax）").
- Concrete over abstract: shapes with real numbers, a 3×3 example, a printed tensor.
- Say *why* it matters in real work (context), then *what* to do (prompt), then small nudges (hints).
- Chinese: plain, spoken-style Chinese; keep code identifiers and common English terms (batch, loss, token,
  shape) as they are. Use 「」 for quoting UI/text in Chinese prose.
- English: plain English (short sentences, active voice), not a literal translation. Same facts, same numbers.
- Never promise what the code does not do. Every number in a card or explanation must be true for the code shown.
- Don't point by position. A choice item's `--- code` is shown *below* its prompt, and a card's code sits beside
  the text on a wide screen but under it on a phone: write "the code below" / "this card's code", never "on the right".
- `title:` and `cards:` values are taken literally: don't wrap them in quotes (the build rejects quoted titles).

## The English file

`content/en/dayN.txt` mirrors the Chinese file block by block: same days, same card count/order, same item ids,
types, levels, `targets`, `ref`, `answer`, `exam`, `timeout`, same number of options / hints / checks / blanks /
checklist lines, and the same `cards:` links (by the English card titles). The build refuses a mismatch.

Translate everything the learner reads: titles, goal, checklist, card bodies, context, prompt, options, explain,
hints, checklist — and inside code: comments, docstrings, check titles and failure messages. Keep the code itself
identical (names, logic, numbers).
