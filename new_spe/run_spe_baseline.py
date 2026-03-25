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
from tqdm import tqdm

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

# 强制对齐 SPE 的合并数据集
TRAIN_PATH = "data/bbh/merged_train.json"
TEST_PATH = "data/bbh/merged_test.json"

# [统一资源约束 - 与 SPE 2500 实验完全一致]
BUDGET = 2500  # 总预算 (评估1题=1，调用大模型生成变异=1)
BATCH_SIZE = 20  # 每次测 20 题
INITIAL_CANDIDATES = 10  # 初始生成的候选数量
TOP_K = 3  # 每次保留前 3 名
VARIANTS = 3  # 每个精英变异 3 次

# [输出文件]
OUTPUT_FILE = "ape_best_prompt.json"
LOG_FILE = "ape_train_log.csv"
DETAIL_LOG = "ape_evolution_details.txt"


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
                # 隐藏过多报错信息，避免刷屏
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
        """【核心改造：多线程并发 + 预算控制】"""
        if remaining_budget <= 0 or not data_pool:
            return 0.0, [], 0

        # 确保不超预算
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

        # 并发执行 batch 评估
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(20, actual_batch)) as executor:
            futures = [executor.submit(_eval_single, item) for item in batch]
            concurrent.futures.wait(futures)

        score = (correct / actual_batch) * 100 if actual_batch > 0 else 0
        return score, details, actual_batch  # 返回消耗的预算


class APE:
    def __init__(self, train_data):
        self.client = DeepSeekClient()
        self.evaluator = Evaluator(self.client)
        self.train_data = train_data
        self.history = set()
        self.used_budget = 0  # 绝对公平的计费器

        with open(LOG_FILE, 'w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(["BudgetUsed", "Source", "Score", "Prompt"])
        with open(DETAIL_LOG, 'w', encoding='utf-8') as f:
            f.write(f"=== FAIR APE BASELINE Started at {time.strftime('%H:%M:%S')} ===\n\n")

    def log_details(self, budget, prompt, score, snapshots):
        with open(DETAIL_LOG, 'a', encoding='utf-8') as f:
            f.write(f"{'=' * 20} Budget Used {budget}/{BUDGET} (Score: {score:.1f}%) {'=' * 20}\n")
            f.write(f"PROMPT: {prompt}\n\n")
            correct_sample = next((x for x in snapshots if x['correct']), None)
            wrong_sample = next((x for x in snapshots if not x['correct']), None)
            if correct_sample:
                f.write(
                    f"[✅ SUCCESS CASE]\nQ: {correct_sample['question'][:100]}...\nA: {correct_sample['target']}\nModel CoT:\n{correct_sample['reasoning']}\n\n")
            if wrong_sample:
                f.write(
                    f"[❌ FAILURE CASE]\nQ: {wrong_sample['question'][:100]}...\nA: {wrong_sample['target']} | Pred: {wrong_sample['pred']}\nModel CoT:\n{wrong_sample['reasoning']}\n\n")

    def generate_candidates(self, num):
        if self.used_budget >= BUDGET: return []
        print(f"{Fore.BLUE}🧠 [Generating] Calling LLM to create {num} initial seeds...{Fore.RESET}")

        self.used_budget += 1  # 生成一批初始 Prompt 算 1 次调用

        meta_prompt = (
            f"Task: Generate {num} concise 'Chain-of-Thought' trigger phrases.\n"
            "Goal: These phrases will be appended to a logic question to force the AI to think step-by-step.\n"
            "Constraint: Output ONLY the phrases, one per line."
        )
        candidates = set()
        res = self.client.generate(meta_prompt, temperature=1.0)
        for line in res.split('\n'):
            clean = re.sub(r'^[\d\-\.\)\s]+', '', line).strip().strip('"')
            if len(clean) > 5 and clean not in self.history:
                candidates.add(clean)
                self.history.add(clean)
        return list(candidates)[:num]

    def mutate(self, best_prompts):
        print(f"{Fore.BLUE}🧬 [Mutating] Calling LLM to evolve top {len(best_prompts)} candidates...{Fore.RESET}")
        new_cands = set()
        for p in best_prompts:
            for _ in range(VARIANTS):
                if self.used_budget >= BUDGET: break
                msg = f"Rewrite this prompt to be more effective/concise/logical:\n\"{p}\"\nOutput ONLY the rewritten version."
                res = self.client.generate(msg, temperature=0.9)
                self.used_budget += 1  # 每次变异消耗 1 点预算

                clean = res.strip().strip('"')
                if clean and clean not in self.history:
                    new_cands.add(clean)
                    self.history.add(clean)
        return list(new_cands)

    def run(self):
        pool = self.train_data
        candidates = self.generate_candidates(INITIAL_CANDIDATES)
        best_prompt, best_score = "", 0.0
        generation = 0

        while self.used_budget < BUDGET and candidates:
            generation += 1
            print(
                f"\n{Style.BRIGHT}=== Generation {generation} | Budget: {self.used_budget}/{BUDGET} ==={Style.RESET_ALL}")

            iteration_results = []

            # 评估当前所有候选
            for p in candidates:
                if self.used_budget >= BUDGET: break

                score, details, cost = self.evaluator.evaluate(p, pool, BUDGET - self.used_budget)
                self.used_budget += cost  # 扣除做题费

                iteration_results.append((p, score, details))
                if score > best_score:
                    best_score = score
                    best_prompt = p

            if not iteration_results: break

            # 排序筛选
            iteration_results.sort(key=lambda x: x[1], reverse=True)
            scores = [x[1] for x in iteration_results]

            avg_score = statistics.mean(scores)
            print(
                f"{Fore.CYAN}📊 Stats:{Fore.RESET} Max: {Fore.GREEN}{max(scores):.1f}%{Fore.RESET} | Avg: {Fore.YELLOW}{avg_score:.1f}%{Fore.RESET} | Used Budget: {self.used_budget}")

            top_p, top_s, top_d = iteration_results[0]
            print(f"{Fore.GREEN}🏆 Gen Best: \"{top_p}\" ({top_s:.1f}%){Fore.RESET}")

            for p, s, d in iteration_results:
                with open(LOG_FILE, 'a', newline='', encoding='utf-8') as f:
                    csv.writer(f).writerow([self.used_budget, "Eval", f"{s:.1f}", p])
            self.log_details(self.used_budget, top_p, top_s, top_d)

            # 如果没钱了，直接退出
            if self.used_budget >= BUDGET: break

            # 生成下一代
            top_k_prompts = [x[0] for x in iteration_results[:TOP_K]]
            candidates = self.mutate(top_k_prompts)

        return best_prompt, best_score


def test_full_dataset(client, prompt, test_data):
    """【新增：多线程进行全量测试集大考，对齐 SPE】"""
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
    print(f"{Style.BRIGHT}🚀 APE FAIR BASELINE Started...{Style.RESET_ALL}")

    if not os.path.exists(TRAIN_PATH) or not os.path.exists(TEST_PATH):
        print("❌ 找不到合并后的数据文件，请先运行合并脚本。")
        return

    with open(TRAIN_PATH, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    with open(TEST_PATH, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    # 1. 严格预算下的进化
    ape = APE(train_data)
    best_prompt, train_score = ape.run()

    print("\n" + "=" * 60)
    print(f"{Fore.GREEN}🏆 FINAL CHAMPION PROMPT:\n\"{best_prompt}\"{Fore.RESET}")
    print(f"🎯 Train Max Score (Sampled): {train_score:.1f}%")

    # 2. 最终大考 (不计入预算)
    final_test_acc = test_full_dataset(ape.client, best_prompt, test_data)

    print(f"\n🎉 实验总结报告 (APE Baseline)")
    print(f"   - 消耗总预算: {ape.used_budget}/{BUDGET}")
    print(f"   - 训练集表现: {train_score:.1f}%")
    print(f"   - 测试集准确率: {final_test_acc:.2%}")
    print("=" * 60)

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump({
            "method": "APE (Flat Context)",
            "budget": ape.used_budget,
            "best_prompt": best_prompt,
            "train_score": train_score,
            "final_test_acc": final_test_acc
        }, f, indent=4)


if __name__ == "__main__":
    main()