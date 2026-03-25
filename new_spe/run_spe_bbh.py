import argparse
import json
import time
import dataclasses
from pathlib import Path
import numpy as np

# 导入核心组件
from new_spe.utils.config_loader import load_deepseek_config
from new_spe.models.deepseek_kernel import DeepSeekKernel
from new_spe.evaluation.bbh_evaluator import BBHEvaluator
from new_spe.search.embedding import HashingNgramEmbedder
from new_spe.search.optimizer import SPEOptimizer, SPEOptimizerConfig
from new_spe.core.genome import StructuredGenome


# ==========================================
# 工具函数
# ==========================================
def _jsonl_logger(path: Path):
    """日志记录器：将每一步实验数据存入 JSONL"""
    path.parent.mkdir(parents=True, exist_ok=True)

    def log_fn(obj):
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    return log_fn


def _to_jsonable(x):
    """递归转换对象为可序列化的 JSON 格式"""
    if dataclasses.is_dataclass(x):
        return _to_jsonable(dataclasses.asdict(x))
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, dict):
        return {str(k): _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    return x


# ==========================================
# 参数解析 (采用弱种子策略)
# ==========================================
def parse_args():
    p = argparse.ArgumentParser(description="SPE Large-Scale Mixed Task Evolution (Weak Seed)")

    # 数据路径
    p.add_argument("--train_json", type=str, default="data/bbh/merged_train.json")
    p.add_argument("--test_json", type=str, default="data/bbh/merged_test.json")
    p.add_argument("--bbh_n_shot", type=int, default=3)

    # 预算与进化参数
    p.add_argument("--budget", type=int, default=2500)  # 总 API 预算
    p.add_argument("--batch_size", type=int, default=20)  # 保持与消融实验相同的 20 题难度
    p.add_argument("--gens", type=int, default=30)
    p.add_argument("--mu", type=int, default=6)
    p.add_argument("--lambda_", type=int, default=8)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_dir", type=str, default="logs")

    p.add_argument("--role", type=str, default="You are a brilliant mathematician and logician.")
    p.add_argument("--instruct", type=str, default="Solve the problem.")
    p.add_argument("--const", type=str, default="")
    p.add_argument("--style", type=str, default="")

    return p.parse_args()


# ==========================================
# 主流程
# ==========================================
def main():
    from my_api_key import inject_api_key
    inject_api_key()

    args = parse_args()
    print("\n" + "=" * 60)
    print("🚀 SPE 主实验: FULL-SCALE CROSS-TASK EVOLUTION (弱种子启动)")
    print("=" * 60 + "\n")

    cfg = load_deepseek_config("config/apikey.txt")
    kernel = DeepSeekKernel(cfg, verbose=False)
    embedder = HashingNgramEmbedder()

    print(f"📖 正在加载合并数据集...")
    train_evaluator = BBHEvaluator(path=args.train_json, n_shot=args.bbh_n_shot, seed=args.seed)
    test_evaluator = BBHEvaluator(path=args.test_json, n_shot=args.bbh_n_shot, seed=args.seed + 1)

    train_total = len(train_evaluator.flat_dataset)
    test_total = len(test_evaluator.flat_dataset)
    print(f"✅ 加载完成 | 训练样本: {train_total} | 测试样本 (全量大考): {test_total}")

    # ==========================================
    # 【核心修改 2】：添加探针，但保留 MOO 紧凑度目标！
    # ==========================================
    def train_eval_fn(prompt: str):
        res = train_evaluator.evaluate_once(kernel=kernel, prompt=prompt, embedder=embedder)

        prompt_len = len(prompt)
        est_tokens = prompt_len / 4.0

        if 'y' in res and res['y'] is not None and len(res['y']) > 1:
            acc = res['y'][0]
            compactness = res['y'][1]
            # 实时打印：观察准确率和紧凑度是如何在帕累托前沿上博弈的
            print(
                f"   [DEBUG] 长度: {prompt_len} 字符 (约 {est_tokens:.0f} Tokens) | 准确率: {acc:.2%} | 紧凑度: {compactness:.4f}")
            # ⚠️ 注意：这里绝对不能把 compactness 设为 0，我们要让 MOO 发挥作用！

        return res

    run_id = time.strftime("%Y%m%d_%H%M%S")
    log_path = Path(args.log_dir) / f"spe_full_mix_{run_id}.jsonl"
    log_fn = _jsonl_logger(log_path)

    optimizer_cfg = SPEOptimizerConfig(
        budget=args.budget,
        mu=args.mu,
        lambd=args.lambda_,
        gens=args.gens,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    optimizer = SPEOptimizer(kernel=kernel, cfg=optimizer_cfg, embedder=embedder)

    init_pop = []
    for i in range(args.mu):
        loci = {"L_role": args.role, "L_instruct": args.instruct, "L_const": args.const, "L_style": args.style}
        init_pop.append(StructuredGenome(loci=loci, uid=f"init_{i}", operator="init"))

    log_fn({"phase": "experiment_start", "args": vars(args), "config": _to_jsonable(optimizer_cfg)})

    print(f"\n🧬 进化开始... (目标预算: {args.budget} | 批次大小: {args.batch_size})")
    start_time = time.time()

    final_pool = optimizer.evolve(init_population=init_pop, eval_fn=train_eval_fn, log_fn=log_fn)

    best_train_ind = max(final_pool, key=lambda g: float(g.mu()[0]))
    best_prompt_text = best_train_ind.prompt_text()

    evolution_time = time.time() - start_time
    print(f"\n✅ 进化阶段结束! (耗时: {evolution_time:.1f}s)")
    print(f"🏆 训练集最强精英: {best_train_ind.uid} | 训练集准确率(抽样): {best_train_ind.mu()[0]:.2%}")

    with open("spe_main_best_prompt.txt", "w", encoding="utf-8") as f:
        f.write(best_prompt_text)

    # ---------------------------------------------------------
    # 5. 终极考验
    # ---------------------------------------------------------
    print(f"\n🏁 正在进行全量测试集大考 (共 {test_total} 题，启用并发提速)...")
    import threading
    import concurrent.futures

    test_correct = 0
    test_processed = 0
    test_start_time = time.time()
    detailed_test_results = []

    test_lock = threading.Lock()

    def _test_single_task(idx: int):
        res = test_evaluator.evaluate_by_index(kernel=kernel, prompt=best_prompt_text, index=idx)
        is_correct = res.get('y', [0.0])[0] > 0.5

        nonlocal test_correct, test_processed
        with test_lock:
            if is_correct:
                test_correct += 1
            test_processed += 1

            detailed_test_results.append({
                "idx": idx,
                "task": res.get("task", "unknown"),
                "correct": is_correct
            })

            if test_processed % 50 == 0 or test_processed == test_total:
                elapsed = time.time() - test_start_time
                avg_time = elapsed / test_processed
                eta = avg_time * (test_total - test_processed)
                acc = test_correct / test_processed
                print(f"   进度: [{test_processed}/{test_total}] | 实时准确率: {acc:.2%} | 预计剩余: {eta / 60:.1f}min")

    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(_test_single_task, idx) for idx in range(test_total)]
        concurrent.futures.wait(futures)

    final_test_accuracy = test_correct / test_total if test_total > 0 else 0

    # ---------------------------------------------------------
    # 6. 实验收尾与总结
    # ---------------------------------------------------------
    print(f"\n" + "🎉" * 20)
    print(f"   主实验总结报告: SPE Baseline (弱种子启动)")
    print(f"   - 训练准确率 (Batch Avg): {best_train_ind.mu()[0]:.2%}")
    print(f"   - 测试准确率 (全量大考): {final_test_accuracy:.2%}")
    print(f"   - 最终字符数: {len(best_prompt_text)} (预估 {len(best_prompt_text) / 4:.0f} Tokens)")
    print(f"   - 进化耗时: {evolution_time / 60:.1f} min")
    print(f"🎉" * 20 + "\n")

    final_summary = {
        "phase": "experiment_end",
        "best_prompt": best_prompt_text,
        "best_prompt_length": len(best_prompt_text),
        "estimated_tokens": len(best_prompt_text) / 4.0,
        "best_train_acc": best_train_ind.mu()[0],
        "final_test_acc": final_test_accuracy,
        "total_budget_used": optimizer.used_budget,
        "detailed_results": detailed_test_results
    }

    with open("spe_main_result.json", "w", encoding="utf-8") as f:
        json.dump(final_summary, f, ensure_ascii=False, indent=4)

    log_fn(_to_jsonable(final_summary))
    print(f"📝 详细实验日志已保存至: {log_path} 和 spe_main_result.json")


if __name__ == "__main__":
    main()