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
# 【重构版】：无历史档案库优化器
# ==========================================
class SPEOptimizerNoArchive(SPEOptimizer):
    def _update_elite_pool(self, candidates):
        """
        核心覆盖：真正的纯代际替换。
        不论历史数据有多好，我们只从当前这一代 (candidates) 中选拔 Top-K，直接覆盖。
        """
        if not candidates:
            return

        # 1. 对当前传入的个体按准确率降序排列
        sorted_cands = sorted(candidates, key=lambda g: float(g.mu()[0]), reverse=True)

        # 2. 截取前 mu 个作为新的精英池，抛弃历史档案
        self.elite_pool = sorted_cands[:self.cfg.mu]


# ==========================================
# 参数解析
# ==========================================
def parse_args():
    p = argparse.ArgumentParser(description="Ablation: SPE w/o Archive")
    p.add_argument("--train_json", type=str, default="data/bbh/merged_train.json")
    p.add_argument("--test_json", type=str, default="data/bbh/merged_test.json")
    p.add_argument("--bbh_n_shot", type=int, default=3)
    p.add_argument("--budget", type=int, default=500)
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
    from my_api_key import inject_api_key
    inject_api_key()

    args = parse_args()
    print("\n" + "=" * 60)
    print("🔬 消融实验 B: SPE w/o Archive (纯代际替换，极易退化)")
    print("=" * 60 + "\n")

    cfg = load_deepseek_config("config/apikey.txt")
    kernel = DeepSeekKernel(cfg, verbose=False)
    embedder = HashingNgramEmbedder()

    print(f"📖 正在加载合并数据集...")
    train_evaluator = BBHEvaluator(path=args.train_json, n_shot=args.bbh_n_shot, seed=args.seed)
    test_evaluator = BBHEvaluator(path=args.test_json, n_shot=args.bbh_n_shot, seed=args.seed + 1)

    train_total = len(train_evaluator.flat_dataset)
    test_total = len(test_evaluator.flat_dataset)
    print(f"✅ 加载完成 | 训练样本: {train_total} | 测试样本: {test_total}")

    # 恢复纯净版的评估函数
    def train_eval_fn(prompt: str):
        return train_evaluator.evaluate_once(kernel=kernel, prompt=prompt, embedder=embedder)

    run_id = time.strftime("%Y%m%d_%H%M%S")
    log_path = Path(args.log_dir) / f"ablation_no_archive_{run_id}.jsonl"
    log_fn = _jsonl_logger(log_path)

    optimizer_cfg = SPEOptimizerConfig(
        budget=args.budget, mu=args.mu, lambd=args.lambda_,
        gens=args.gens, batch_size=args.batch_size, seed=args.seed
    )

    optimizer = SPEOptimizerNoArchive(kernel=kernel, cfg=optimizer_cfg, embedder=embedder)

    init_pop = []
    for i in range(args.mu):
        loci = {"L_role": args.role, "L_instruct": args.instruct, "L_const": args.const, "L_style": args.style}
        init_pop.append(StructuredGenome(loci=loci, uid=f"init_{i}", operator="init"))

    log_fn({"phase": "experiment_start", "args": vars(args), "config": _to_jsonable(optimizer_cfg)})

    print(f"\n🧬 无保护进化开始... (目标预算: {args.budget})")
    start_time = time.time()
    final_pool = optimizer.evolve(init_population=init_pop, eval_fn=train_eval_fn, log_fn=log_fn)

    best_train_ind = max(final_pool, key=lambda g: float(g.mu()[0]))
    best_prompt_text = best_train_ind.prompt_text()
    evolution_time = time.time() - start_time

    print(f"\n✅ 进化阶段结束! (耗时: {evolution_time:.1f}s)")
    print(f"🏆 残存最强个体: {best_train_ind.uid} | 训练集准确率: {best_train_ind.mu()[0]:.2%}")

    with open("ablation_no_archive_best.txt", "w", encoding="utf-8") as f:
        f.write(best_prompt_text)

    print(f"\n🏁 正在进行全量测试集大考 (共 {test_total} 题)...")
    import threading
    import concurrent.futures

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

    print(f"\n" + "🎉" * 20)
    print(f"   实验总结报告: SPE w/o Archive")
    print(f"   - 训练准确率: {best_train_ind.mu()[0]:.2%}")
    print(f"   - 测试准确率: {final_test_accuracy:.2%}")
    print(f"🎉" * 20 + "\n")

    result_data = {
        "ablation_type": "SPE w/o Archive",
        "train_score": best_train_ind.mu()[0],
        "final_test_accuracy": final_test_accuracy,
        "best_prompt": best_prompt_text
    }
    with open("ablation_no_archive_result.json", "w", encoding="utf-8") as f:
        json.dump(result_data, f, ensure_ascii=False, indent=4)
    print("💾 完整测试报告已保存至: ablation_no_archive_result.json")


if __name__ == "__main__":
    main()