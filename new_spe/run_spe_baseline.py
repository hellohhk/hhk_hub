import argparse
import json
import time
import threading
import concurrent.futures
from pathlib import Path

# 导入核心组件
from new_spe.utils.config_loader import load_deepseek_config
from new_spe.models.deepseek_kernel import DeepSeekKernel
from new_spe.evaluation.bbh_evaluator import BBHEvaluator


def parse_args():
    p = argparse.ArgumentParser(description="SPE Standalone Test (Run Best Prompt on Test Set)")
    p.add_argument("--test_json", type=str, default="data/bbh/merged_test.json")
    p.add_argument("--prompt_file", type=str, default="spe_main_best_prompt.txt")
    p.add_argument("--bbh_n_shot", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    # 1. 注入 API Key
    from my_api_key import inject_api_key
    inject_api_key()

    args = parse_args()
    print("\n" + "=" * 60)
    print("🎯 SPE 独立测试模式: 验证最强提示词的最终实力")
    print("=" * 60 + "\n")

    # 2. 读取我们辛辛苦苦进化出来的 Best Prompt
    prompt_path = Path(args.prompt_file)
    if not prompt_path.exists():
        print(f"❌ 找不到提示词文件: {args.prompt_file}，请确认文件路径是否正确！")
        return

    with open(prompt_path, "r", encoding="utf-8") as f:
        best_prompt_text = f.read()

    print(f"📄 成功加载目标提示词 (长度: {len(best_prompt_text)} 字符)")

    # 3. 初始化大模型与评测器
    cfg = load_deepseek_config("config/apikey.txt")
    kernel = DeepSeekKernel(cfg, verbose=False)

    print(f"📖 正在加载测试集: {args.test_json}")
    # 注意：这里的 seed 保持和主实验 test_evaluator 一致 (seed + 1)
    test_evaluator = BBHEvaluator(path=args.test_json, n_shot=args.bbh_n_shot, seed=args.seed + 1)
    test_total = len(test_evaluator.flat_dataset)
    print(f"✅ 加载完成 | 测试样本总数: {test_total}")

    # 4. 开启多线程大考
    print(f"\n🏁 正在进行全量测试集大考 (共 {test_total} 题，启用多线程并发并发提速)...")

    test_correct = 0
    test_processed = 0
    test_start_time = time.time()
    detailed_test_results = []
    test_lock = threading.Lock()

    def _test_single_task(idx: int):
        # 调用大模型进行推理
        res = test_evaluator.evaluate_by_index(kernel=kernel, prompt=best_prompt_text, index=idx)
        # 只要 y 向量的第一位大于 0.5 即判定为正确
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

            # 每处理 50 题打印一次进度
            if test_processed % 50 == 0 or test_processed == test_total:
                elapsed = time.time() - test_start_time
                avg_time = elapsed / test_processed
                eta = avg_time * (test_total - test_processed)
                acc = test_correct / test_processed
                print(f"   进度: [{test_processed}/{test_total}] | 实时准确率: {acc:.2%} | 预计剩余: {eta / 60:.1f}min")

    # 启用 20 个并发线程拉满速度
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(_test_single_task, idx) for idx in range(test_total)]
        concurrent.futures.wait(futures)

    final_test_accuracy = test_correct / test_total if test_total > 0 else 0
    total_time = time.time() - test_start_time

    # 5. 打印并保存总结报告
    print(f"\n" + "🎉" * 20)
    print(f"   独立测试总结报告: 终极精英 Prompt 大考")
    print(f"   - 最终测试准确率: {final_test_accuracy:.2%}")
    print(f"   - 测试集总耗时: {total_time / 60:.1f} min")
    print(f"🎉" * 20 + "\n")

    final_summary = {
        "phase": "standalone_test_end",
        "best_prompt": best_prompt_text,
        "final_test_acc": final_test_accuracy,
        "test_total_time_seconds": total_time,
        "detailed_results": detailed_test_results
    }

    # 存入专门的结果文件，避免覆盖之前的进化记录
    result_file = "standalone_test_result.json"
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(final_summary, f, ensure_ascii=False, indent=4)

    print(f"📝 详细测试结果已保存至: {result_file}")


if __name__ == "__main__":
    main()