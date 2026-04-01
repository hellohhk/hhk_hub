import argparse
import json
import random
import os
import re
import csv
import time
import statistics
import threading
import concurrent.futures
from pathlib import Path
from tqdm import tqdm

try:
    from colorama import Fore, Style, init

    init(autoreset=True)
except ImportError:
    class Fore:
        GREEN = ""; RED = ""; CYAN = ""; YELLOW = ""; RESET = ""


    class Style:
        BRIGHT = ""; RESET_ALL = ""

# 🌟 导入统一的底层通信和计费组件
from new_spe.utils.config_loader import load_deepseek_config
from new_spe.models.token_kernel import TokenTrackedKernel


def parse_args():
    p = argparse.ArgumentParser(description="APE Token Efficiency Experiment")
    p.add_argument("--token_budget", type=int, default=500000, help="全局 Token 消耗上限")
    p.add_argument("--batch_size", type=int, default=15, help="每个 Prompt 测 15 题")
    p.add_argument("--init_candidates", type=int, default=10, help="初始 Prompt 数量")
    p.add_argument("--top_k", type=int, default=3, help="每轮保留前 K")
    p.add_argument("--variants", type=int, default=3, help="每个 Top-K 的变异数量")
    p.add_argument("--train_json", type=str, default="data/bbh/merged_train.json")
    p.add_argument("--log_dir", type=str, default="logs_token_exp")
    return p.parse_args()


class APEEvaluator:
    def __init__(self, kernel: TokenTrackedKernel):
        self.kernel = kernel

    def extract_answer(self, text):
        text = text.strip()
        match = re.findall(r'(?:answer|option) is[:\s]*(?:\()?([a-zA-Z0-9]+)', text, re.IGNORECASE)
        if match: return match[-1].lower()
        match = re.findall(r'\\boxed\{([a-zA-Z0-9]+)\}', text)
        if match: return match[-1].lower()
        match = re.findall(r'\(([A-Z])\)', text)
        if match: return match[-1].upper()
        if 'yes' in text.lower()[:10]: return 'yes'
        if 'no' in text.lower()[:10]: return 'no'
        return text.strip()

    def evaluate(self, prompt_text, data_pool, batch_size):
        if self.kernel.is_budget_exhausted() or not data_pool:
            return 0.0, []

        actual_batch = min(batch_size, len(data_pool))
        batch = random.sample(data_pool, actual_batch)

        correct = 0
        details = []
        lock = threading.Lock()

        def _eval_single(item):
            if self.kernel.is_budget_exhausted(): return
            try:
                full_input = f"Question: {item['input']}\nAnswer: {prompt_text}"
                res = self.kernel.chat(
                    system_msg="You are a logical reasoning expert. Solve the problem step-by-step.",
                    user_msg=full_input,
                    expect_json=False,
                    temperature=0.0
                )
                if not res.content: return

                raw_output = res.content
                pred = self.extract_answer(raw_output)
                target = str(item['target']).strip().lower()
                pred_clean = pred.replace('(', '').replace(')', '').lower()
                target_clean = target.replace('(', '').replace(')', '').lower()

                is_correct = (pred_clean == target_clean) or (target_clean in pred_clean.split())

                with lock:
                    nonlocal correct
                    if is_correct: correct += 1
                    # 保存详情
                    details.append({
                        "question": item['input'],
                        "target": target,
                        "pred": pred,
                        "correct": is_correct,
                        "reasoning": raw_output
                    })
            except Exception as e:
                print(f"   ❌ 测验异常: {e}")

        # 并发评估提速
        max_workers = min(10, actual_batch)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_eval_single, item) for item in batch]
            concurrent.futures.wait(futures, timeout=120)

        score = (correct / actual_batch) * 100 if actual_batch > 0 else 0.0
        return score, details


class APETokenDriven:
    def __init__(self, kernel: TokenTrackedKernel, evaluator: APEEvaluator, train_data: list, args):
        self.kernel = kernel
        self.evaluator = evaluator
        self.train_data = train_data
        self.args = args
        self.history = set()

        self.DETAIL_LOG = "ape_evolution_details.txt"
        with open(self.DETAIL_LOG, 'w', encoding='utf-8') as f:
            f.write(f"=== APE Evolution Log Started at {time.strftime('%H:%M:%S')} ===\n\n")

    def log_details(self, iteration, prompt, score, snapshots):
        with open(self.DETAIL_LOG, 'a', encoding='utf-8') as f:
            f.write(f"{'=' * 20} Iteration {iteration} (Score: {score:.1f}%) {'=' * 20}\n")
            f.write(f"PROMPT: {prompt}\n\n")

            correct_sample = next((x for x in snapshots if x.get('correct')), None)
            wrong_sample = next((x for x in snapshots if not x.get('correct')), None)

            if correct_sample:
                f.write(
                    f"[✅ SUCCESS CASE]\nQ: {correct_sample['question'][:100]}...\nA: {correct_sample['target']}\nModel CoT:\n{correct_sample['reasoning']}\n\n")
            if wrong_sample:
                f.write(
                    f"[❌ FAILURE CASE]\nQ: {wrong_sample['question'][:100]}...\nA: {wrong_sample['target']} | Pred: {wrong_sample['pred']}\nModel CoT:\n{wrong_sample['reasoning']}\n\n")
            f.write("\n")

    def generate_candidates(self, num):
        if self.kernel.is_budget_exhausted(): return []
        print(f"{Fore.BLUE}🧠 [Generating] Creating {num} initial candidates...{Fore.RESET}")

        # 🌟 修改点：强行注入与 SPE 一致的冷启动种子，保证实验起跑线公平！
        spe_seed = "Role: You are a brilliant mathematician and logician.\nTask: Solve the problem."
        candidates = {spe_seed}
        self.history.add(spe_seed)

        meta_prompt = (
            f"Task: Generate {num - 1} concise 'Chain-of-Thought' trigger phrases.\n"
            "Goal: These phrases will be appended to a hard logic question to force the AI to think step-by-step.\n"
            "Constraint: Output ONLY the phrases, one per line."
        )
        res = self.kernel.chat(system_msg="You are a helpful assistant.", user_msg=meta_prompt, expect_json=False,
                               temperature=1.0)

        if res.content:
            for line in res.content.split('\n'):
                clean = re.sub(r'^[\d\-\.\)\s]+', '', line).strip().strip('"')
                if len(clean) > 5 and clean not in self.history:
                    candidates.add(clean)
                    self.history.add(clean)
        return list(candidates)[:num]

    def mutate(self, best_prompts):
        print(f"{Fore.BLUE}🧬 [Mutating] Evolving top candidates...{Fore.RESET}")
        new_cands = set()
        for p in best_prompts:
            if self.kernel.is_budget_exhausted(): break
            msg = f"Rewrite this prompt to be more effective/concise/logical:\n\"{p}\"\nOutput ONLY the rewritten version."

            for _ in range(self.args.variants):
                if self.kernel.is_budget_exhausted(): break
                res = self.kernel.chat(system_msg="You are a helpful assistant.", user_msg=msg, expect_json=False,
                                       temperature=0.9)
                if res.content:
                    clean = res.content.strip().strip('"')
                    if clean and clean not in self.history:
                        new_cands.add(clean)
                        self.history.add(clean)
        return list(new_cands)

    def run(self):
        pool = self.train_data
        candidates = self.generate_candidates(self.args.init_candidates)

        best_prompt, best_score = "", 0.0
        generation = 0
        top_k_prompts = []

        print("\n🚀 [Token 驱动进化] APE 开始演化...")
        # 🌟 Token 驱动循环核心
        while not self.kernel.is_budget_exhausted():
            generation += 1
            print(
                f"\n{Style.BRIGHT}=== Iteration {generation} | 当前 Token: {self.kernel.total_tokens_consumed}/{self.kernel.token_budget} ==={Style.RESET_ALL}")

            iteration_results = []

            pbar = tqdm(candidates, desc="Evaluating Prompts", unit="prompt")
            for p in pbar:
                if self.kernel.is_budget_exhausted():
                    print("\n⚠️ Token耗尽，中断当前代评估。")
                    break

                score, details = self.evaluator.evaluate(p, pool, self.args.batch_size)
                iteration_results.append((p, score, details))

                if score > best_score:
                    best_score = score
                    best_prompt = p
                    pbar.set_postfix({"Best": f"{best_score:.0f}%"})

            if not iteration_results:
                break

            iteration_results.sort(key=lambda x: x[1], reverse=True)
            scores = [x[1] for x in iteration_results]

            avg_score = statistics.mean(scores)
            max_score = max(scores)
            print(
                f"\n{Fore.CYAN}📊 Round {generation} Stats:{Fore.RESET} Max: {Fore.GREEN}{max_score:.1f}%{Fore.RESET} | Avg: {Fore.YELLOW}{avg_score:.1f}%{Fore.RESET}")

            top_p, top_s, top_d = iteration_results[0]
            print(f"{Fore.GREEN}🏆 Round Best: \"{top_p}\" ({top_s:.1f}%){Fore.RESET}")

            self.log_details(generation, top_p, top_s, top_d)

            # 🌟 记录当前最佳分数用于画图
            self.kernel.log_performance(best_score)

            # 突变繁育下一代
            if not self.kernel.is_budget_exhausted():
                top_k_prompts = [x[0] for x in iteration_results[:self.args.top_k]]
                candidates = self.mutate(top_k_prompts)

                # 随机填充以维持种群多样性
                if len(candidates) < self.args.top_k * self.args.variants and not self.kernel.is_budget_exhausted():
                    candidates.extend(self.generate_candidates(self.args.init_candidates // 2))

        return best_prompt, best_score


def main():
    import os
    from pathlib import Path

    # 兼容路径导入
    try:
        from new_spe.my_api_key import inject_api_key
        inject_api_key()
    except ImportError:
        os.environ["DEEPSEEK_API_KEY"] = "sk-9c6929df4f5541eb94ad3af0c77ddfcc"

    args = parse_args()
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print(f"🚀 Token 效率公平对比实验启动")
    print(f"   - 测试基线: [APE (Automatic Prompt Engineer)]")
    print(f"   - 预算上限: {args.token_budget} Tokens")
    print("=" * 60 + "\n")

    cfg_path = "new_spe/config/apikey.txt" if Path("new_spe/config/apikey.txt").exists() else "config/apikey.txt"
    cfg = load_deepseek_config(cfg_path)

    kernel = TokenTrackedKernel(config=cfg, token_budget=args.token_budget, verbose=False)

    print(f"📖 正在加载训练集 ({args.train_json})...")
    with open(args.train_json, 'r', encoding='utf-8') as f:
        train_data = json.load(f)

    evaluator = APEEvaluator(kernel)
    ape_runner = APETokenDriven(kernel, evaluator, train_data, args)

    start_time = time.time()
    best_prompt, best_score = ape_runner.run()
    run_time = time.time() - start_time

    print("\n" + "🎉" * 20)
    print(f"   实验总结报告: APE")
    print(f"   - 最终消耗 Token: {kernel.total_tokens_consumed}")
    print(f"   - 最佳训练准确率: {best_score:.2f}%")
    print(f"   - 运行总耗时: {run_time / 60:.1f} min")
    print("\n   👑 [最终最佳提示词 (Best Prompt)] 👑")
    print("-" * 50)
    print(best_prompt)
    print("-" * 50)
    print("🎉" * 20 + "\n")

    # 导出 JSON 曲线数据
    run_id = time.strftime("%Y%m%d_%H%M%S")
    save_path = Path(args.log_dir) / f"ape_curve_budget_{args.token_budget}_{run_id}.json"

    curve_data = {
        "experiment_meta": {
            "method": "ape",
            "token_budget": args.token_budget,
            "final_tokens_used": kernel.total_tokens_consumed,
            "best_acc_found": best_score,
            "best_prompt_text": best_prompt
        },
        "performance_curve": kernel.performance_curve
    }

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(curve_data, f, ensure_ascii=False, indent=4)

    print(f"📈 珍贵的【Token-性能曲线】与【最佳提示词】数据已安全保存至: {save_path}")


if __name__ == "__main__":
    main()