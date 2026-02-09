# 记忆锚点：SPE (Structured Prompt Evolution) + BBH 实验框架

本文档用于“快速恢复上下文”：说明当前目录的目标、代码组织、核心算法实现、关键参数、日志字段含义、以及后续扩展应从哪里改起。

> 目录：`experiments/lxb_spe/`

---

## 0. 当前目标与假设

### 0.1 目标
- 在 **受限评估预算** 下，做 **多目标**（准确率 + 简洁度）提示词优化。
- 实现两套可对照的优化器：
  - **SPE（结构化）**：Prompt 拆成基因座，算子受不变量约束（默认保持 `L_const` 不变）。
  - **Flat baseline（扁平化）**：Prompt 作为单一字符串整体改写，容易发生“格式/约束漂移”。
- 统一接入 **BBH** 数据集：支持自动下载、缓存、加载、few-shot 拼接、随机抽样评测。
- 输出可用于论文验证的日志：包含“search radius（prompt embedding 位移）”与“output variance（输出方差）”等指标。

### 0.2 预算口径（非常重要）
- `budget` 默认以 **评测调用次数** 为主（每评一次任务算一次预算）。
- 同时，部分“生成算子”（rewrite/refine/mix）会额外消耗一次模型调用，在优化器中被记为 operator cost（可理解为“变异核调用”）。
- 因此：总预算约束包含了评测开销 + 部分算子开销（保持与论文“API 调用预算”一致的直觉）。

---

## 1. 目录结构（稳定入口）

### 1.1 主要入口脚本
- `spe.py`
  - 统一 CLI 入口：跑结构化 SPE（BBH）。
  - 适合当作“主实验脚本”。
- `run_spe_full.py`
  - 单独的结构化 SPE runner（BBH），更偏“一个脚本跑一次实验”。
- `run_flat_baseline.py`
  - Flat baseline runner（BBH）。

### 1.2 核心包
- `spe_full/`
  - SPE 与 Flat 的实现、BBH 下载与评测、日志字段与统计都在这里。

### 1.3 配置与测试
- `configs/`
  - 预置“失败任务”配置等（不一定会自动被 runner 读取，目前更多是留作对照/复现实验的参数模板）。
- `tests/`
  - 离线 mock 测试（不打外部 API），保证核心流程可运行。

---

## 2. API Key 与网络

### 2.1 DeepSeek 配置加载
位置：`spe_full/config.py`
- 优先从环境变量读取：
  - `DEEPSEEK_API_KEY`（必需）
  - `DEEPSEEK_BASE_URL`（默认 `https://api.deepseek.com`）
  - `DEEPSEEK_MODEL`（默认 `deepseek-chat`）
  - `DEEPSEEK_TEMPERATURE`（默认 `0.6`）
  - `DEEPSEEK_TIMEOUT_S`（默认 `120`）
- 若环境变量缺失，则从 `apikey.txt` 中解析 `model_one: deepseek` 段落的 `api_key/base_url`。

### 2.2 Kernel（LLM 调用）
位置：`spe_full/kernel.py`
- `DeepSeekKernel.chat(system_msg, user_msg, expect_json, stream=True|False, ...)`
- 默认使用 stream 方式拼接输出（更抗长回答/网络波动）。

---

## 3. BBH 数据集（自动下载、缓存、加载）

位置：`spe_full/bbh.py`

### 3.1 下载策略
- `ensure_bbh_downloaded(cache_dir)`
  - 从 GitHub 仓库 ZIP 下载：`suzgunmirac/BIG-Bench-Hard` 的 main 分支压缩包。
  - 解压 `bbh/*.json` 到：`{cache_dir}/bbh/`
  - 写 marker：`{cache_dir}/.bbh_ready`

### 3.2 任务加载与过滤
- `BBHEvaluator.from_cache(cache_dir, tasks=None|[...], seed, n_shot, include_description)`
  - `tasks=None`：加载 cache 下所有 `*.json` 任务
  - `tasks=[...]`：按任务名过滤（匹配 JSON 内 name 或文件 stem）

### 3.3 Few-shot 构造（当前实现）
对于每次评测：
- 随机采样一个 task
- 再随机选择一个 example 作为 query
- 从该 task 的剩余 examples 中随机抽 `n_shot` 作为示例
- 拼成 user message：
  - `Task: {name}`
  - `{description}`（可选）
  - 若干 `Input/Target` few-shot
  - 最后 `Input: ...\nTarget:` 让模型补全

### 3.4 判分规则（当前实现，故意保持“轻依赖”）
位置：`bbh.answers_equivalent`
- 默认做轻量归一化：
  - 去多余空白
  - yes/no → 小写
  - 尝试提取单字母选项 `A/B/C/...`
  - gold 支持 `|||` 表示多个等价答案
- 注意：这不是 “官方最强判分”，但满足“可自动跑 + 可对照实验”。
  - 若要更严格/更贴合任务，可以做 task-specific parser 映射表。

---

## 4. 目标函数 F(p)：多目标评估定义

评测输出向量：`y = [accuracy, compactness]`
- `accuracy`：本次样本是否判为正确（0/1）
- `compactness`：`1/(1+log1p(len(response)))`，回答越短越高

输出还包含：
- `response_len`
- 以及可选 `response_emb`（用于输出方差统计）

---

## 5. Search Radius Logger（论文关键指标之一）

### 5.1 Prompt embedding displacement（搜索半径）
实现位置：
- Structured：`spe_full/optimizer.py::_make_offspring`
- Flat：`spe_full/flat.py::_make_offspring`

做法：
- 用 `HashingNgramEmbedder` 对 **prompt 文本**做 embedding（不依赖外部模型）。
- 计算子代与父代 prompt embedding 的 L2 距离，并记录到：
  - `prompt_radius.prompt_disp_l2_p1`
  - `prompt_radius.prompt_disp_l2_p2`（二元算子）
  - `prompt_radius.prompt_disp_l2_mean`（二元算子）

### 5.2 Output variance（输出方差）
实现位置：
- Structured：`spe_full/genome.py` + `spe_full/optimizer.py::_evaluate_once`
- Flat：`spe_full/flat.py`

累计统计：
- `output_emb_trace_var`：输出 embedding 的方差迹（trace(var)）
- `output_len_var`：输出长度方差
- `y_var`：目标向量的方差（accuracy/compactness）

用途：
- 对比 SPE vs Flat：在同等预算下，是否更能压缩输出漂移与评测不确定性。

---

## 6. SPE（结构化）实现：核心组件

### 6.1 Genome
位置：`spe_full/genome.py`
- 4 loci：`L_role, L_instruct, L_const, L_style`
- 统计量：
  - `n / mean / m2`：y 的在线均值方差
  - output embedding 与 output length 的在线方差统计
  - `radius`：search radius 字段（在 offspring 生成时填）

### 6.2 算子族（Operator Set）
位置：`spe_full/operators.py`
- `IntraLocusRewrite (K_rew)`：大步重写 `L_instruct`
- `IntraLocusRefine (K_ref)`：小步润色 `L_instruct`
- `LocusCrossover (K_swp)`：交换某个非不变量基因座（默认不含 `L_const`）
- `SemanticInterpolation (K_mix)`：融合两段 `L_instruct`

不变量（Kernel Projection）：
- `invariant_loci` 默认 `("L_const",)`，子代自动投影回父代的 const。

### 6.3 调度与选择
- HVC-UCB 调度：`spe_full/scheduler.py` + `spe_full/hypervolume.py`
- NSGA-II 生存选择：`spe_full/selection.py` + `spe_full/pareto.py`

### 6.4 优化器（协议总控）
位置：`spe_full/optimizer.py`
流程：
1. init population 初始评估 `n_init`
2. 每代：
   - NSGA-II ranking + tournament 选父代
   - 生成 `lambda` 个 offspring（部分算子会消耗 operator cost）
   - offspring 初始评估 `n_init_offspring`
   - HVC-UCB schedule：对 pool 做额外评估（`schedule_multiplier * |pool|` 次）
   - NSGA-II 选出下一代 `mu`
3. 返回最终 population，并可 extract pareto front

---

## 7. Flat baseline（最小对照）

位置：`spe_full/flat.py`

关键点：
- `FlatGenome.text` 是整个 prompt 文本，没有 loci 拆分，也没有 `L_const` 投影保护。
- 算子：
  - `FlatRewrite / FlatRefine / FlatMix / FlatSwap`
- 调度与选择复用同样的 HVC-UCB 与 NSGA-II，用于公平对照。

预期现象：
- Flat 更容易“输出格式漂移/指令丢失”，导致判分噪声变大、search radius 变大、output variance 变大。

---

## 8. 日志（JSONL）字段约定

日志由 runner 写入（每次评估、每代结束都会写），适合后处理统计与绘图。

常见字段：
- `phase`：`init | offspring_init | schedule | gen_end | run_start | run_end`
- `uid / parents / operator`
- `n / mu / y / y_var`
- `prompt_radius`（search radius）
- `output_emb_trace_var / output_len_var`
- BBH 相关：
  - `task / example_idx / gold / pred`
  - `response_len`

---

## 9. Runner 使用（最短命令）

### 9.1 结构化 SPE（BBH）
```bash
py spe.py --bbh_cache_dir data/bbh --bbh_tasks boolean_expressions --bbh_n_shot 3 --budget 200 --gens 10
```

### 9.2 Flat baseline（BBH）
```bash
py run_flat_baseline.py --bbh_cache_dir data/bbh --bbh_tasks boolean_expressions --bbh_n_shot 3 --budget 200 --gens 10
```

---

## 10. 失败任务配置（模板）

位置：`configs/failure_task_flat_bbh_format.json`
- 目的：制造“扁平化输出格式漂移”案例（不一定要立即执行）。
- 当前 runner 不会自动读 configs；该文件作为参数模板/实验记录。

---

## 11. 测试与可用性保证

离线测试（不打外部 API）：
- `tests/test_spe_full_mock.py`：结构化优化器最小跑通 + 输出统计字段
- `tests/test_flat_baseline_mock.py`：Flat baseline 最小跑通
- `tests/test_bbh_loader.py`：BBH 本地任务加载、判分逻辑、evaluate_once 输出字段

运行：
```bash
py -m pytest -q
```

---

## 12. 后续最常见扩展点（建议优先级）

1. **BBH 任务定制判分**
   - 在 `bbh.py` 增加 task-specific parser（比如多选题/数值题/字符串格式等）。
2. **Search radius 的 embedding 更换**
   - 目前是 hashing ngram embedder（轻依赖）。如果要更语义化，可切到句向量模型，但需要额外依赖与缓存策略。
3. **更严格的预算口径**
   - 目前 operator cost 只对部分算子计 1（rewrite/refine/mix）。若引入更多核调用（如验证器/critic），要同步纳入预算。
4. **日志后处理脚本**
   - 建议写一个 `scripts/analyze_logs.py`：读取 JSONL → 出曲线（mu/pareto/search radius/variance）→ 输出表格（均值±方差/显著性检验）。

