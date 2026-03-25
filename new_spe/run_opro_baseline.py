import time
import threading
import concurrent.futures
import json
from pathlib import Path

# 导入你项目中的核心组件
from new_spe.utils.config_loader import load_deepseek_config
from new_spe.models.deepseek_kernel import DeepSeekKernel
from new_spe.evaluation.bbh_evaluator import BBHEvaluator

# ==========================================
# OPRO 最佳提示词 (Hardcoded Baseline)
# ==========================================
OPRO_BEST_PROMPT = (
    "Thoroughly understand the problem by identifying all given information, constraints, "
    "and the objective. Determine the appropriate reasoning type (logical, mathematical, "
    "common sense, or a blend). Break the problem into explicit, manageable steps, applying "
    "precise reasoning at each step and verifying correctness. After deriving a solution, "
    "critically review the entire process to ensure all conditions are met and no errors "
    "remain. Finally, state the final answer directly and concisely."
)

# 测试集路径
TEST_JSON_PATH = "data/bbh/merged_test.json"


def main():
    print("\n" + "=" * 60)
    print("🚀 OPRO BASELINE EVALUATION (Full Test Set)")
    print("=" * 60 + "\n")

    # 1. 初始化模型内核
    print("🔧 正在初始化 DeepSeek 内核...")
    cfg = load_deepseek_config("config/apikey.txt")
    kernel = DeepSeekKernel(cfg, verbose=False)

    # 2. 加载测试集 (合并后的全任务测试集)
    print(f"📖 正在加载测试集: {TEST_JSON_PATH}")
    test_evaluator = BBHEvaluator(path=TEST_JSON_PATH, n_shot=3, seed=42)
    test_total = len(test_evaluator.flat_dataset)
    print(f"✅ 加载完成 | 测试样本总量: {test_total}")

    # 3. 开始多线程全量并发测试
    print(f"\n🏁 正在使用 OPRO 提示词进行并发测试...")
    print(f"📝 Prompt 内容: \"{OPRO_BEST_PROMPT[:80]}...\"")

    test_correct = 0
    test_processed = 0
    test_start_time = time.time()

    # 建立测试数据统计锁
    test_lock = threading.Lock()

    def _test_single_task(idx: int):
        # 传递 OPRO 提示词给评测器
        res = test_evaluator.evaluate_by_index(kernel=kernel, prompt=OPRO_BEST_PROMPT, index=idx)
        is_correct = res.get('y', [0.0])[0] > 0.5

        # 拿回结果后排队登记
        nonlocal test_correct, test_processed
        with test_lock:
            if is_correct:
                test_correct += 1
            test_processed += 1

            # 实时进度条
            if test_processed % 50 == 0 or test_processed == test_total:
                elapsed = time.time() - test_start_time
                avg_time = elapsed / test_processed
                eta = avg_time * (test_total - test_processed)
                acc = test_correct / test_processed
                print(f"   [进度 {test_processed}/{test_total}] | 实时准确率: {acc:.2%} | 预计剩余: {eta / 60:.1f}min")

    # 开辟 20 个并发线程同时大考
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(_test_single_task, idx) for idx in range(test_total)]
        concurrent.futures.wait(futures)

    final_test_accuracy = test_correct / test_total if test_total > 0 else 0
    test_time = time.time() - test_start_time

    # 4. 打印最终报告
    print(f"\n" + "🎉" * 20)
    print(f"   OPRO 基础基线 (Baseline) 测试报告")
    print(f"   - 测试集准确率 (全量): {final_test_accuracy:.2%}")
    print(f"   - 测试总耗时: {test_time / 60:.1f} min")
    print(f"🎉" * 20 + "\n")

    # 保存结果到单独的 JSON 文件
    result_data = {
        "method": "OPRO (Flat Natural Language)",
        "prompt": OPRO_BEST_PROMPT,
        "test_total": test_total,
        "final_test_accuracy": final_test_accuracy,
        "test_time_seconds": test_time
    }

    Path("logs").mkdir(exist_ok=True)
    with open("logs/opro_baseline_result.json", "w", encoding="utf-8") as f:
        json.dump(result_data, f, ensure_ascii=False, indent=4)
    print(f"📝 结果已保存至: logs/opro_baseline_result.json")


if __name__ == "__main__":
    main()