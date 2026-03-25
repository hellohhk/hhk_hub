import argparse
import json
import time
import dataclasses
from pathlib import Path
import numpy as np
import threading
import concurrent.futures

# 导入核心组件
from new_spe.utils.config_loader import load_deepseek_config
from new_spe.models.deepseek_kernel import DeepSeekKernel
from new_spe.evaluation.bbh_evaluator import BBHEvaluator
from new_spe.search.embedding import HashingNgramEmbedder
from new_spe.search.optimizer import SPEOptimizer, SPEOptimizerConfig
from new_spe.core.genome import StructuredGenome


# ==========================================
# 工具函数 (完全保留)
# ==========================================
def _jsonl_logger(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    def log_fn(obj):
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    return log_fn


def _to_jsonable(x):
    if dataclasses.is_dataclass(x): return _to_jsonable(dataclasses.asdict(x))
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, dict): return {str(k): _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [_to_jsonable(v) for v in x]
    return x


# ==========================================
# 参数解析 (完全保留，仅调小了默认预算以适配消融实验)
# ==========================================
def parse_args():
    p = argparse.ArgumentParser(description="Ablation: SPE w/o Structure")
    p.add_argument("--train_json", type=str, default="data/bbh/merged_train.json")
    p.add_argument("--test_json", type=str, default="data/bbh/merged_test.json")
    p.add_argument("--bbh_n_shot", type=int, default=3)
    p.add_argument("--budget", type=int, default=500)  # 消融实验预算调小
    p.add_argument("--batch_size", type=int, default=10)
    p.add_argument("--gens", type=int, default=10)
    p.add_argument("--mu", type=int, default=4)
    p.add_argument("--lambda_", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_dir", type=str, default="logs")
    p.add_argument("--role", type=str, default="You are a brilliant mathematician and logician.")
    p.add_argument("--instruct", type=str, default="Solve the problem step by step, using clear logical deduction.")
    p.add_argument("--const", type=str,
                   default="Put your final answer in the format required by the problem. Do not add conversational filler.")
    p.add_argument("--style", type=str, default="Rigorous and analytical")
    return p.parse_args()


# ==========================================
# 主流程
# ==========================================
def main():
    args = parse_args()
    print("\n" + "=" * 60)
    print("🔬 消融实验 A: SPE w/o Structure (退化为扁平化变异)")
    print("=" * 60 + "\n")

    # 1. 初始化模型内核与向量化工具 (完全保留原路径)
    cfg = load_deepseek_config("config/apikey.txt")
    kernel = DeepSeekKernel(cfg, verbose=False)
    embedder = HashingNgramEmbedder()

    # 2. 初始化合并数据集评测器 (完全保留原路径)
    print(f"📖 正在加载合并数据集...")
    train_evaluator = BBHEvaluator(path=args.train_json, n_shot=args.bbh_n_shot, seed=args.seed)
    test_evaluator = BBHEvaluator(path=args.test_json, n_shot=args.bbh_n_shot, seed=args.seed + 1)

    train_total = len(train_evaluator.flat_dataset)
    test_total = len(test_evaluator.flat_dataset)
    print(f"✅ 加载完成 | 训练样本: {train_total} | 测试样本 (全量大考): {test_total}")

    def train_eval_fn(prompt: str):
        return train_evaluator.evaluate_once(kernel=kernel, prompt=prompt, embedder=embedder)

    # 3. 初始化 SPE 优化器
    run_id = time.strftime("%Y%m%d_%H%M%S")
    log_path = Path(args.log_dir) / f"ablation_no_structure_{run_id}.jsonl"
    log_fn = _jsonl_logger(log_path)

    # 【消融核心修改 1】：清空 invariant_loci 基因锁
    optimizer_cfg = SPEOptimizerConfig(
        budget=args.budget, mu=args.mu, lambd=args.lambda_,
        gens=args.gens, batch_size=args.batch_size, seed=args.seed,
        invariant_loci=()  # <--- 干掉所有结构化保护
    )
    optimizer = SPEOptimizer(kernel=kernel, cfg=optimizer_cfg, embedder=embedder)

    # 4. 进化循环
    # 【消融核心修改 2】：将四个结构化基因揉成一段扁平文本
    flat_prompt_text = f"{args.role} {args.instruct} {args.const} Style: {args.style}"
    init_pop = []
    for i in range(args.mu):
        loci = {"L_flat": flat_prompt_text}  # <--- 只有一个扁平的基因位
        init_pop.append(StructuredGenome(loci=loci, uid=f"init_flat_{i}", operator="init"))

    log_fn({"phase": "experiment_start", "args": vars(args), "config": _to_jsonable(optimizer_cfg)})

    print(f"\n🧬 扁平化无结构进化开始... (目标预算: {args.budget} | 批次大小: {args.batch_size})")
    start_time = time.time()
    final_pool = optimizer.evolve(init_population=init_pop, eval_fn=train_eval_fn, log_fn=log_fn)

    best_train_ind = max(final_pool, key=lambda g: float(g.mu()[0]))
    best_prompt_text = best_train_ind.prompt_text()
    evolution_time = time.time() - start_time

    print(f"\n✅ 进化阶段结束! (耗时: {evolution_time:.1f}s)")
    print(f"🏆 训练集最强精英: {best_train_ind.uid} | 训练集准确率: {best_train_ind.mu()[0]:.2%}")

    # 保存残缺版最佳 Prompt
    with open("ablation_no_structure_best.txt", "w", encoding="utf-8") as f:
        f.write(best_prompt_text)

    # 5. 终极考验：多线程并发对测试集进行全量测试
    print(f"\n🏁 正在使用无结构化最佳 Prompt 进行全量测试集大考 (共 {test_total} 题)...")
    test_correct = 0
    test_processed = 0
    test_start_time = time.time()
    test_lock = threading.Lock()

    def _test_single_task(idx: int):
        res = test_evaluator.evaluate_by_index(kernel=kernel, prompt=best_prompt_text, index=idx)
        is_correct = res.get('y', [0.0])[0] > 0.5
        nonlocal test_correct, test_processed
        with test_lock:
            if is_correct: test_correct += 1
            test_processed += 1
            if test_processed % 50 == 0 or test_processed == test_total:
                print(f"   进度: [{test_processed}/{test_total}] | 实时准确率: {test_correct / test_processed:.2%}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(_test_single_task, idx) for idx in range(test_total)]
        concurrent.futures.wait(futures)

    final_test_accuracy = test_correct / test_total if test_total > 0 else 0

    # 6. 实验收尾
    print(f"\n" + "🎉" * 20)
    print(f"   实验总结报告: SPE w/o Structure")
    print(f"   - 训练准确率 (Batch Avg): {best_train_ind.mu()[0]:.2%}")
    print(f"   - 测试准确率 (全量大考): {final_test_accuracy:.2%}")
    print(f"🎉" * 20 + "\n")

    # 写入 JSON，方便明天查阅
    result_data = {
        "ablation_type": "SPE w/o Structure",
        "train_score": best_train_ind.mu()[0],
        "final_test_accuracy": final_test_accuracy,
        "best_prompt": best_prompt_text
    }
    with open("ablation_no_structure_result.json", "w", encoding="utf-8") as f:
        json.dump(result_data, f, ensure_ascii=False, indent=4)
    print("💾 完整测试报告已保存至: ablation_no_structure_result.json")


if __name__ == "__main__":
    main()