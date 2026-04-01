import argparse
import json
import random
import re
import time
import statistics
import threading
import concurrent.futures
from pathlib import Path

# 🌟 导入统一的底层通信和计费组件
from new_spe.utils.config_loader import load_deepseek_config
from new_spe.models.token_kernel import TokenTrackedKernel


def parse_args():
    p = argparse.ArgumentParser(description="EvoPrompt Token Efficiency Experiment")
    p.add_argument("--token_budget", type=int, default=500000, help="全局 Token 消耗上限")
    p.add_argument("--batch_size", type=int, default=20, help="每次测验的题目数量")
    p.add_argument("--pop_size", type=int, default=10, help="EvoPrompt 种群大小")
    p.add_argument("--train_json", type=str, default="data/bbh/merged_train.json")
    p.add_argument("--log_dir", type=str, default="logs_token_exp")
    return p.parse_args()


class EvoPromptEvaluator:
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
            return 0.0

        actual_batch = min(batch_size, len(data_pool))
        batch = random.sample(data_pool, actual_batch)

        correct = 0
        lock = threading.Lock()

        def _eval_single(item):
            if self.kernel.is_budget_exhausted(): return
            try:
                full_input = f"Question: {item['input']}\nAnswer: {prompt_text}"
                # 调用带记账功能的内核
                res = self.kernel.chat(
                    system_msg="You are a logical reasoning expert. Solve the problem step-by-step.",
                    user_msg=full_input,
                    expect_json=False,
                    temperature=0.0
                )

                if not res.content: return

                pred = self.extract_answer(res.content)
                target = str(item['target']).strip().lower()
                pred_clean = pred.replace('(', '').replace(')', '').lower()
                target_clean = target.replace('(', '').replace(')', '').lower()

                is_correct = (pred_clean == target_clean) or (target_clean in pred_clean.split())

                with lock:
                    nonlocal correct
                    if is_correct: correct += 1
            except Exception as e:
                print(f"   ❌ 测验异常: {e}")

        # 并发评估，加入超时强杀保护
        max_workers = min(10, actual_batch)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_eval_single, item) for item in batch]
            concurrent.futures.wait(futures, timeout=120)

        return correct / actual_batch if actual_batch > 0 else 0.0


class EvoPromptTokenDriven:
    def __init__(self, kernel: TokenTrackedKernel, evaluator: EvoPromptEvaluator, train_data: list, args):
        self.kernel = kernel
        self.evaluator = evaluator
        self.train_data = train_data
        self.args = args
        self.history = set()

    def generate_initial_population(self, num):
        if self.kernel.is_budget_exhausted(): return []
        print(f"\n🌱 [Init] Calling LLM to create {num} initial prompts...")

        meta_prompt = (
            f"Task: Generate {num} distinct 'Chain-of-Thought' trigger phrases for complex reasoning tasks.\n"
            "Constraint: Output ONLY the phrases, one per line."
        )
        population = []
        res = self.kernel.chat(system_msg="You are a helpful assistant.", user_msg=meta_prompt, expect_json=False,
                               temperature=1.0)

        if res.content:
            for line in res.content.split('\n'):
                clean = re.sub(r'^[\d\-\.\)\s]+', '', line).strip().strip('"')
                if len(clean) > 5 and clean not in self.history:
                    population.append(clean)
                    self.history.add(clean)
        return population[:num]

    def roulette_wheel_selection(self, population_with_scores):
        total_score = sum(s for _, s in population_with_scores)
        if total_score == 0:
            return random.choice(population_with_scores)[0]
        pick = random.uniform(0, total_score)
        current = 0
        for p, s in population_with_scores:
            current += s
            if current > pick: return p
        return population_with_scores[-1][0]

    def evolve_ga(self, prompt1, prompt2):
        if self.kernel.is_budget_exhausted(): return None

        ga_instruction = (
            "Please follow the instruction step-by-step to generate a better prompt.\n"
            "1. Cross over the following prompts and generate a new prompt:\n"
            f"Prompt 1: {prompt1}\n"
            f"Prompt 2: {prompt2}\n"
            "2. Mutate the prompt generated in Step 1 and generate a final prompt bracketed with <prompt> and </prompt>."
        )
        res = self.kernel.chat(system_msg="You are a helpful assistant.", user_msg=ga_instruction, expect_json=False,
                               temperature=0.9)

        if not res.content: return None

        match = re.search(r'<prompt>(.*?)</prompt>', res.content, re.DOTALL | re.IGNORECASE)
        clean = match.group(1).strip() if match else res.content.split('\n')[-1].strip().replace('<prompt>',
                                                                                                 '').replace(
            '</prompt>', '')

        if clean and clean not in self.history:
            self.history.add(clean)
            return clean
        return None

    def run(self):
        # 1. 初始化
        candidates = self.generate_initial_population(self.args.pop_size)
        population_results = []

        print("\n=== Evaluating Initial Population ===")
        for p in candidates:
            if self.kernel.is_budget_exhausted(): break
            score = self.evaluator.evaluate(p, self.train_data, self.args.batch_size)
            population_results.append((p, score))

        best_score = max([s for _, s in population_results] + [0.0])
        self.kernel.log_performance(best_score)
        print(f"✅ 初始种群评估完毕 | 最高准确率: {best_score:.2%}")

        generation = 0
        best_prompt = ""

        # 2. Token 驱动循环
        print("\n🚀 [Token 驱动进化] 开始跨代繁殖...")
        while not self.kernel.is_budget_exhausted():
            generation += 1
            print(
                f"\n--- [ 进化代数 {generation} | 当前 Token 消耗: {self.kernel.total_tokens_consumed} / {self.kernel.token_budget} ] ---")

            offspring_results = []
            for _ in range(self.args.pop_size):
                if self.kernel.is_budget_exhausted(): break

                p1 = self.roulette_wheel_selection(population_results)
                p2 = self.roulette_wheel_selection(population_results)

                print("   🧬 触发交叉变异算子...")
                child_prompt = self.evolve_ga(p1, p2)

                if child_prompt and not self.kernel.is_budget_exhausted():
                    score = self.evaluator.evaluate(child_prompt, self.train_data, self.args.batch_size)
                    offspring_results.append((child_prompt, score))
                    print(f"      📊 子代得分: {score:.2%}")

            if not offspring_results: break

            # 合并并选择 Top-N
            combined_pool = population_results + offspring_results
            combined_pool.sort(key=lambda x: x[1], reverse=True)
            population_results = combined_pool[:self.args.pop_size]

            scores = [x[1] for x in population_results]
            current_best = max(scores) if scores else 0.0

            # 🌟 记录当前代数的最优表现，用于画图
            self.kernel.log_performance(current_best)

            best_prompt = population_results[0][0]
            print(f"🏆 第 {generation} 代结束 | 全局最高准确率: {current_best:.2%} | Avg: {statistics.mean(scores):.2%}")

        return best_prompt, current_best


def main():
    try:
        from my_api_key import inject_api_key
        inject_api_key()
    except ImportError:
        pass

    args = parse_args()
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print(f"🚀 Token 效率公平对比实验启动")
    print(f"   - 测试基线: [EVOPROMPT]")
    print(f"   - 预算上限: {args.token_budget} Tokens")
    print("=" * 60 + "\n")

    # 🌟 实例化受 Token 限制的核心引擎
    cfg = load_deepseek_config("config/apikey.txt")
    kernel = TokenTrackedKernel(config=cfg, token_budget=args.token_budget, verbose=True)

    print(f"📖 正在加载训练集 ({args.train_json})...")
    with open(args.train_json, 'r', encoding='utf-8') as f:
        train_data = json.load(f)

    evaluator = EvoPromptEvaluator(kernel)
    evoprompt_runner = EvoPromptTokenDriven(kernel, evaluator, train_data, args)

    # 运行演化
    start_time = time.time()
    best_prompt, best_score = evoprompt_runner.run()
    run_time = time.time() - start_time

    # 实验收尾
    print("\n" + "🎉" * 20)
    print(f"   实验总结报告: EVOPROMPT")
    print(f"   - 最终消耗 Token: {kernel.total_tokens_consumed}")
    print(f"   - 最佳训练准确率: {best_score:.2%}")
    print(f"   - 运行总耗时: {run_time / 60:.1f} min")
    print("\n   👑 [最终最佳提示词 (Best Prompt)] 👑")
    print("-" * 50)
    print(best_prompt)
    print("-" * 50)
    print("🎉" * 20 + "\n")

    # 保存用于画图的 JSON 曲线数据
    run_id = time.strftime("%Y%m%d_%H%M%S")
    save_path = Path(args.log_dir) / f"evoprompt_curve_budget_{args.token_budget}_{run_id}.json"

    curve_data = {
        "experiment_meta": {
            "method": "evoprompt",
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