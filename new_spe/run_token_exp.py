import argparse
import json
import time
from pathlib import Path

# 导入你现有的基础组件
from new_spe.utils.config_loader import load_deepseek_config
from new_spe.evaluation.bbh_evaluator import BBHEvaluator
from new_spe.search.embedding import HashingNgramEmbedder
from new_spe.search.optimizer import SPEOptimizerConfig
from new_spe.core.genome import StructuredGenome

# 导入 SPE 独家算子
from new_spe.operators.spe_operators import (
    IntraLocusRewrite, IntraLocusRefine, LocusCrossover,
    SemanticInterpolation, ErrorDrivenRefine, CompactnessPruner
)

# 🌟 导入我们刚写好的独立实验组件
from new_spe.models.token_kernel import TokenTrackedKernel
from new_spe.search.token_optimizer import TokenBoundedSPEOptimizer


def parse_args():
    p = argparse.ArgumentParser(description="SPE vs Baseline: Token Efficiency Experiment")

    # 核心参数：方法与统一预算
    p.add_argument("--method", type=str, default="spe", choices=["spe", "opro", "ape", "evoprompt"],
                   help="选择要评估的算法基线")
    p.add_argument("--token_budget", type=int, default=500000, help="全局 Token 消耗上限 (硬性熔断点)")

    # 实验配置参数
    p.add_argument("--batch_size", type=int, default=20, help="每次测验的题目数量")
    p.add_argument("--train_json", type=str, default="data/bbh/merged_train.json")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_dir", type=str, default="logs_token_exp")

    return p.parse_args()


def main():
    # 注入 API Key
    from my_api_key import inject_api_key
    inject_api_key()

    args = parse_args()
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print(f"🚀 Token 效率公平对比实验启动")
    print(f"   - 测试方法: [{args.method.upper()}]")
    print(f"   - 预算上限: {args.token_budget} Tokens")
    print(f"   - 批次大小: {args.batch_size} 题/次")
    print("=" * 60 + "\n")

    # 1. 初始化底层 API 配置
    cfg = load_deepseek_config("config/apikey.txt")

    # 🌟 2. 实例化带财务监控的 Kernel (所有请求的咽喉)
    kernel = TokenTrackedKernel(config=cfg, token_budget=args.token_budget, verbose=False)
    embedder = HashingNgramEmbedder()

    # 3. 初始化评估器
    print(f"📖 正在加载合并数据集 ({args.train_json})...")
    evaluator = BBHEvaluator(path=args.train_json, n_shot=3, seed=args.seed)

    def eval_fn(prompt: str):
        # 注意：这里传入的 kernel 是带 Token 阻断功能的
        return evaluator.evaluate_once(kernel=kernel, prompt=prompt, embedder=embedder)

    final_pool = []

    # ==========================================
    # 分支执行逻辑
    # ==========================================
    if args.method == "spe":
        # SPE 专属配置
        custom_operators = [
            IntraLocusRewrite(), IntraLocusRefine(), LocusCrossover(),
            SemanticInterpolation(), ErrorDrivenRefine(), CompactnessPruner()
        ]
        # 严格遵守主实验的算力分配
        custom_probs = [0.15, 0.1, 0.15, 0.1, 0.35, 0.15]

        # 这里的 budget (999999) 和 gens 已经失效，实际生命周期由 Token 决定
        optimizer_cfg = SPEOptimizerConfig(
            budget=999999, mu=3, lambd=4, batch_size=args.batch_size, seed=args.seed
        )

        optimizer = TokenBoundedSPEOptimizer(
            kernel=kernel, cfg=optimizer_cfg, embedder=embedder,
            operators=custom_operators, operator_probs=custom_probs
        )

        # 单点种子冷启动
        init_pop = [StructuredGenome(
            loci={"L_role": "You are a brilliant mathematician and logician.",
                  "L_instruct": "Solve the problem.",
                  "L_const": "",
                  "L_style": ""},
            uid="init_0", operator="init"
        )]

        start_time = time.time()
        final_pool = optimizer.evolve(init_population=init_pop, eval_fn=eval_fn)
        run_time = time.time() - start_time

    elif args.method in ["opro", "ape", "evoprompt"]:
        print(f"🚧 [{args.method.upper()}] 的适配逻辑暂未接入。")
        print(f"👉 后续如需测试，请将对应的 Baseline 逻辑搬移至此，并将它们的底层 API 请求替换为 `kernel` 即可。")
        return

    # ==========================================
    # 实验收尾与数据导出 (核心产出)
    # ==========================================
    best_ind = max(final_pool, key=lambda g: float(g.mu()[0])) if final_pool else None

    if best_ind:
        print("\n" + "🎉" * 20)
        print(f"   实验总结报告: {args.method.upper()}")
        print(f"   - 最终消耗 Token: {kernel.total_tokens_consumed}")
        print(f"   - 最佳训练准确率: {best_ind.mu()[0]:.2%}")
        print(f"   - 运行总耗时: {run_time / 60:.1f} min")
        print("\n   👑 [最终最佳提示词 (Best Prompt)] 👑")
        print("-" * 50)
        print(best_ind.prompt_text())  # 🌟 在终端直接打印出最终的 Prompt
        print("-" * 50)
        print("🎉" * 20 + "\n")

    # 🌟 将曲线数据和最佳提示词一起保存，用于论文画图和记录
    run_id = time.strftime("%Y%m%d_%H%M%S")
    save_path = Path(args.log_dir) / f"{args.method}_curve_budget_{args.token_budget}_{run_id}.json"

    curve_data = {
        "experiment_meta": {
            "method": args.method,
            "token_budget": args.token_budget,
            "batch_size": args.batch_size,
            "final_tokens_used": kernel.total_tokens_consumed,
            "best_acc_found": best_ind.mu()[0] if best_ind else 0.0,
            # 🌟 新增：把最佳提示词的完整文本和结构化基因座都存进 JSON 里
            "best_prompt_text": best_ind.prompt_text() if best_ind else "",
            "best_prompt_loci": best_ind.loci if best_ind else {}
        },
        "performance_curve": kernel.performance_curve
    }

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(curve_data, f, ensure_ascii=False, indent=4)

    print(f"📈 珍贵的【Token-性能曲线】与【最佳提示词】数据已安全保存至: {save_path}")


if __name__ == "__main__":
    main()
