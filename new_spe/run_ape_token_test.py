import argparse
import json
import time
import threading
import concurrent.futures
import random
from pathlib import Path

# 导入核心组件
from new_spe.utils.config_loader import load_deepseek_config
from new_spe.models.deepseek_kernel import DeepSeekKernel
from new_spe.evaluation.bbh_evaluator import BBHEvaluator


def parse_args():
    p = argparse.ArgumentParser(description="Custom Prompt Standalone Test")
    p.add_argument("--test_json", type=str, default="data/bbh/merged_test.json")
    p.add_argument("--bbh_n_shot", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    # 1. 注入 API Key
    try:
        from new_spe.my_api_key import inject_api_key
        inject_api_key()
    except ImportError:
        import os
        # 兼容性备用 Key 注入
        os.environ["DEEPSEEK_API_KEY"] = "sk-9c6929df4f5541eb94ad3af0c77ddfcc"

    args = parse_args()
    print("\n" + "=" * 60)
    print("🎯 独立测试模式: 验证精炼逻辑引导提示词")
    print("=" * 60 + "\n")

    # 2. 🌟 注入你要测试的目标提示词
    best_prompt_text = "First, consider the logical structure carefully."

    print(f"📄 成功加载目标提示词:\n")
    print("-" * 50)
    print(best_prompt_text)
    print("-" * 50 + "\n")

    # 3. 初始化大模型与评测器
    # 自动识别 config 文件的正确路径
    cfg_path = "new_spe/config/apikey.txt" if Path("new_spe/config/apikey.txt").exists() else "config/apikey.txt"
    cfg = load_deepseek_config(cfg_path)
    kernel = DeepSeekKernel(cfg, verbose=False)

    print(f"📖 正在加载测试集: {args.test_json}")
    # 注意：这里的 seed 保持和主实验 test_evaluator 一致 (seed + 1)
    test_evaluator = BBHEvaluator(path=args.test_json, n_shot=args.bbh_n_shot, seed=args.seed + 1)

    # 兼容处理获取长度
    test_total = len(getattr(test_evaluator, 'flat_dataset', []))
    if test_total == 0:
        print("❌ 测试集加载失败或为空，请检查路径！")
        return

    print(f"✅ 加载完成 | 测试样本总数: {test_total}")

    # ==========================================
    # 🌟 关键稳定配置区
    # ==========================================
    MAX_WORKERS = 8  # 并发数保持 8，保护 API 稳定连接
    MAX_RETRIES = 3  # 遇到超时/限流时的最大重试次数
    # ==========================================

    print(f"\n🏁 正在进行全量测试集大考 (共 {test_total} 题，最大并发数: {MAX_WORKERS})...")

    test_correct = 0
    test_processed = 0
    test_start_time = time.time()
    detailed_test_results = []
    test_lock = threading.Lock()

    def _test_single_task(idx: int):
        is_correct = False
        res = {"task": "error"}

        # 🌟 指数退避重试机制
        for attempt in range(MAX_RETRIES):
            try:
                res = test_evaluator.evaluate_by_index(kernel=kernel, prompt=best_prompt_text, index=idx)
                # BBH Evaluator 默认返回的 y 是对答案预测的概率或 One-hot 向量，取第一位判断对错
                is_correct = res.get('y', [0.0])[0] > 0.5
                break  # 如果成功没有报错，跳出重试循环

            except Exception as e:
                wait_time = (2 ** attempt) + random.uniform(0.5, 1.5)
                print(
                    f"   ⚠️ 第 {idx} 题 API 超时/限流 (尝试 {attempt + 1}/{MAX_RETRIES})。等待 {wait_time:.1f} 秒后重试...")
                time.sleep(wait_time)

                if attempt == MAX_RETRIES - 1:
                    print(f"   ❌ 第 {idx} 题连续 {MAX_RETRIES} 次失败，彻底放弃。最后一次报错: {e}")

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

            # 每处理 20 题打印一次进度
            if test_processed % 20 == 0 or test_processed == test_total:
                elapsed = time.time() - test_start_time
                avg_time = elapsed / test_processed
                eta = avg_time * (test_total - test_processed)
                acc = test_correct / test_processed
                print(f"   进度: [{test_processed}/{test_total}] | 实时准确率: {acc:.2%} | 预计剩余: {eta / 60:.1f}min")

    # 使用保护性的并发数启动线程池
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(_test_single_task, idx) for idx in range(test_total)]
        concurrent.futures.wait(futures)

    final_test_accuracy = test_correct / test_total if test_total > 0 else 0
    total_time = time.time() - test_start_time

    # 5. 打印并保存总结报告
    print(f"\n" + "🎉" * 20)
    print(f"   独立测试总结报告")
    print(f"   - 最终测试准确率: {final_test_accuracy:.2%}")
    print(f"   - 测试集总耗时: {total_time / 60:.1f} min")
    print(f"🎉" * 20 + "\n")

    final_summary = {
        "phase": "custom_prompt_test_end",
        "best_prompt": best_prompt_text,
        "final_test_acc": final_test_accuracy,
        "test_total_time_seconds": total_time,
        "detailed_results": detailed_test_results
    }

    # 存入专门的结果文件，避免覆盖之前的测试记录
    result_file = "custom_prompt_test_result.json"
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(final_summary, f, ensure_ascii=False, indent=4)

    print(f"📝 详细测试结果已安全保存至: {result_file}")


if __name__ == "__main__":
    main()