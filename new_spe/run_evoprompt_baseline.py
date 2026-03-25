import json
import random
import os
import re
import csv
import time
import statistics
import threading
import concurrent.futures
from openai import OpenAI

try:
    from colorama import Fore, Style, init

    init(autoreset=True)
except ImportError:
    class Fore:
        GREEN = ""; RED = ""; CYAN = ""; YELLOW = ""; BLUE = ""; RESET = ""


    class Style:
        BRIGHT = ""; RESET_ALL = ""

# ================= 绝对公平的对比配置 =================
API_BASE = "https://api.deepseek.com"
API_KEY = "sk-9c6929df4f5541eb94ad3af0c77ddfcc"  # 你的 Key
MODEL_NAME = "deepseek-chat"

# 强制对齐 SPE 和 APE 的合并数据集
TRAIN_PATH = "data/bbh/merged_train.json"
TEST_PATH = "data/bbh/merged_test.json"

# [统一资源约束 - 2500 预算]
BUDGET = 2000
BATCH_SIZE = 20
POPULATION_SIZE = 10  # EvoPrompt 论文推荐的种群大小 N=10

# [输出文件]
OUTPUT_FILE = "evoprompt_best_prompt.json"
LOG_FILE = "evoprompt_train_log.csv"
DETAIL_LOG = "evoprompt_evolution_details.txt"


# ====================================================

class DeepSeekClient:
    def __init__(self):
        self.client = OpenAI(base_url=API_BASE, api_key=API_KEY)

    def generate(self, prompt, temperature=0.7, max_tokens=1024, system_prompt="You are a helpful assistant."):
        for _ in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=False
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                time.sleep(2)
        return ""


class Evaluator:
    def __init__(self, client):
        self.client = client

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

    def evaluate(self, prompt_text, data_pool, remaining_budget):
        """【多线程并发 + 预算控制评估】"""
        if remaining_budget <= 0 or not data_pool:
            return 0.0, [], 0

        actual_batch = min(BATCH_SIZE, remaining_budget, len(data_pool))
        batch = random.sample(data_pool, actual_batch)

        correct = 0
        details = []
        lock = threading.Lock()

        def _eval_single(item):
            full_input = f"Question: {item['input']}\nAnswer: {prompt_text}"
            raw_output = self.client.generate(
                full_input,
                temperature=0,
                max_tokens=512,
                system_prompt="You are a logical reasoning expert. Solve the problem step-by-step."
            )

            pred = self.extract_answer(raw_output)
            target = str(item['target']).strip().lower()
            pred_clean = pred.replace('(', '').replace(')', '').lower()
            target_clean = target.replace('(', '').replace(')', '').lower()

            is_correct = (pred_clean == target_clean) or (target_clean in pred_clean.split())

            with lock:
                nonlocal correct
                if is_correct: correct += 1
                details.append({
                    "question": item['input'], "target": target, "pred": pred,
                    "correct": is_correct, "reasoning": raw_output
                })

        with concurrent.futures.ThreadPoolExecutor(max_workers=min(20, actual_batch)) as executor:
            futures = [executor.submit(_eval_single, item) for item in batch]
            concurrent.futures.wait(futures)

        score = (correct / actual_batch) * 100 if actual_batch > 0 else 0
        return score, details, actual_batch


class EvoPrompt:
    def __init__(self, train_data):
        self.client = DeepSeekClient()
        self.evaluator = Evaluator(self.client)
        self.train_data = train_data
        self.history = set()
        self.used_budget = 0

        with open(LOG_FILE, 'w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(["BudgetUsed", "Source", "Score", "Prompt"])
        with open(DETAIL_LOG, 'w', encoding='utf-8') as f:
            f.write(f"=== EvoPrompt (GA) BASELINE Started at {time.strftime('%H:%M:%S')} ===\n\n")

    def generate_initial_population(self, num):
        """生成初始种群"""
        if self.used_budget >= BUDGET: return []
        print(f"{Fore.BLUE}🧠 [Init] Calling LLM to create {num} initial prompts...{Fore.RESET}")
        self.used_budget += 1

        meta_prompt = (
            f"Task: Generate {num} distinct 'Chain-of-Thought' trigger phrases for complex reasoning tasks.\n"
            "Constraint: Output ONLY the phrases, one per line."
        )
        population = []
        res = self.client.generate(meta_prompt, temperature=1.0)
        for line in res.split('\n'):
            clean = re.sub(r'^[\d\-\.\)\s]+', '', line).strip().strip('"')
            if len(clean) > 5 and clean not in self.history:
                population.append(clean)
                self.history.add(clean)
        return population[:num]

    def roulette_wheel_selection(self, population_with_scores):
        """【核心复现】轮盘赌选择：分数越高，被选中的概率越大"""
        total_score = sum(s for _, s, _ in population_with_scores)
        if total_score == 0:
            # 如果全军覆没，随机选
            return random.choice(population_with_scores)[0]

        pick = random.uniform(0, total_score)
        current = 0
        for p, s, _ in population_with_scores:
            current += s
            if current > pick:
                return p
        return population_with_scores[-1][0]

    def evolve_ga(self, prompt1, prompt2):
        """【核心复现】调用 LLM 执行 EvoPrompt 论文图1的交叉与变异算子"""
        if self.used_budget >= BUDGET: return None

        # 严格复现 EvoPrompt 论文 Figure 1 的指令
        ga_instruction = (
            "Please follow the instruction step-by-step to generate a better prompt.\n"
            "1. Cross over the following prompts and generate a new prompt:\n"
            f"Prompt 1: {prompt1}\n"
            f"Prompt 2: {prompt2}\n"
            "2. Mutate the prompt generated in Step 1 and generate a final prompt bracketed with <prompt> and </prompt>."
        )

        self.used_budget += 1
        res = self.client.generate(ga_instruction, temperature=0.9)

        # 提取被 <prompt> 包裹的最终结果
        match = re.search(r'<prompt>(.*?)</prompt>', res, re.DOTALL | re.IGNORECASE)
        if match:
            clean = match.group(1).strip()
        else:
            # 如果模型没按格式输出，提取最后一行作为容错
            clean = res.split('\n')[-1].strip().replace('<prompt>', '').replace('</prompt>', '')

        if clean and clean not in self.history:
            self.history.add(clean)
            return clean
        return None

    def run(self):
        pool = self.train_data

        # 1. 初始种群评估
        candidates = self.generate_initial_population(POPULATION_SIZE)
        population_results = []

        print(f"\n{Style.BRIGHT}=== Evaluating Initial Population ==={Style.RESET_ALL}")
        for p in candidates:
            if self.used_budget >= BUDGET: break
            score, details, cost = self.evaluator.evaluate(p, pool, BUDGET - self.used_budget)
            self.used_budget += cost
            population_results.append((p, score, details))

        generation = 0
        best_prompt, best_score = "", 0.0

        # 2. 进化主循环 (Algorithm 2 in EvoPrompt paper)
        while self.used_budget < BUDGET:
            generation += 1
            print(
                f"\n{Style.BRIGHT}=== Generation {generation} | Budget: {self.used_budget}/{BUDGET} ==={Style.RESET_ALL}")

            offspring_results = []

            # 为当前种群的每个位置生成一个新后代
            for _ in range(POPULATION_SIZE):
                if self.used_budget >= BUDGET: break

                # 轮盘赌选父母
                p1 = self.roulette_wheel_selection(population_results)
                p2 = self.roulette_wheel_selection(population_results)

                # 交叉+变异
                child_prompt = self.evolve_ga(p1, p2)

                # 评估后代
                if child_prompt:
                    score, details, cost = self.evaluator.evaluate(child_prompt, pool, BUDGET - self.used_budget)
                    self.used_budget += cost
                    offspring_results.append((child_prompt, score, details))

            if not offspring_results: break

            # 【核心复现】将父代与子代混合，根据分数取 Top-N 作为下一代
            combined_pool = population_results + offspring_results
            combined_pool.sort(key=lambda x: x[1], reverse=True)

            # 更新种群 (Top-N survival)
            population_results = combined_pool[:POPULATION_SIZE]

            scores = [x[1] for x in population_results]
            avg_score = statistics.mean(scores)
            print(
                f"{Fore.CYAN}📊 Stats:{Fore.RESET} Max: {Fore.GREEN}{max(scores):.1f}%{Fore.RESET} | Avg: {Fore.YELLOW}{avg_score:.1f}%{Fore.RESET} | Budget: {self.used_budget}")

            top_p, top_s, top_d = population_results[0]
            print(f"{Fore.GREEN}🏆 Gen Best: \"{top_p}\" ({top_s:.1f}%){Fore.RESET}")

            if top_s > best_score:
                best_score = top_s
                best_prompt = top_p

        return best_prompt, best_score


def test_full_dataset(client, prompt, test_data):
    """【多线程并发大考】"""
    print(f"\n🏁 正在进行最终全量测试集大考 (共 {len(test_data)} 题)...")
    correct = 0
    test_processed = 0
    start_time = time.time()
    lock = threading.Lock()
    evaluator = Evaluator(client)

    def _test_single(item):
        full_input = f"Question: {item['input']}\nAnswer: {prompt}"
        raw_output = client.generate(full_input, temperature=0, max_tokens=512, system_prompt="Solve step-by-step.")
        pred = evaluator.extract_answer(raw_output)
        target = str(item['target']).strip().lower()
        pred_clean = pred.replace('(', '').replace(')', '').lower()
        target_clean = target.replace('(', '').replace(')', '').lower()

        is_correct = (pred_clean == target_clean) or (target_clean in pred_clean.split())

        with lock:
            nonlocal correct, test_processed
            if is_correct: correct += 1
            test_processed += 1

            if test_processed % 50 == 0 or test_processed == len(test_data):
                elapsed = time.time() - start_time
                avg_time = elapsed / test_processed
                eta = avg_time * (len(test_data) - test_processed)
                print(
                    f"   [进度 {test_processed}/{len(test_data)}] | 实时准确率: {correct / test_processed:.2%} | 预计剩余: {eta / 60:.1f}min")

    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(_test_single, item) for item in test_data]
        concurrent.futures.wait(futures)

    return correct / len(test_data) if len(test_data) > 0 else 0


def main():
    print(f"{Style.BRIGHT}🚀 EvoPrompt (GA) BASELINE Started...{Style.RESET_ALL}")

    if not os.path.exists(TRAIN_PATH) or not os.path.exists(TEST_PATH):
        print("❌ 找不到合并后的数据文件，请先运行合并脚本。")
        return

    with open(TRAIN_PATH, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    with open(TEST_PATH, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    # 1. 执行 EvoPrompt 算法
    evoprompt = EvoPrompt(train_data)
    best_prompt, train_score = evoprompt.run()

    print("\n" + "=" * 60)
    print(f"{Fore.GREEN}🏆 FINAL CHAMPION PROMPT:\n\"{best_prompt}\"{Fore.RESET}")
    print(f"🎯 Train Max Score (Sampled): {train_score:.1f}%")

    # 2. 全量测试
    final_test_acc = test_full_dataset(evoprompt.client, best_prompt, test_data)

    print(f"\n🎉 实验总结报告 (EvoPrompt Baseline)")
    print(f"   - 消耗总预算: {evoprompt.used_budget}/{BUDGET}")
    print(f"   - 训练集表现: {train_score:.1f}%")
    print(f"   - 测试集准确率: {final_test_acc:.2%}")
    print("=" * 60)

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump({
            "method": "EvoPrompt (GA)",
            "budget": evoprompt.used_budget,
            "best_prompt": best_prompt,
            "train_score": train_score,
            "final_test_acc": final_test_acc
        }, f, indent=4)


if __name__ == "__main__":
    main()