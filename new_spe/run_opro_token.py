import argparse
import json
import random
import re
import time
import threading
import concurrent.futures
import numpy as np
from pathlib import Path
import pandas as pd

try:
    from colorama import Fore, Style, init

    init(autoreset=True)
except ImportError:
    class Fore:
        GREEN = ""; RED = ""; CYAN = ""; YELLOW = ""; RESET = ""


    class Style:
        BRIGHT = ""; RESET_ALL = ""

# 🌟 导入统一的底层通信、计费和判分组件
from new_spe.utils.config_loader import load_deepseek_config
from new_spe.models.token_kernel import TokenTrackedKernel
from new_spe.evaluation.bbh_evaluator import BBHEvaluator


def parse_args():
    p = argparse.ArgumentParser(description="OPRO Token Efficiency Experiment")
    p.add_argument("--token_budget", type=int, default=500000, help="全局 Token 消耗上限")
    p.add_argument("--batch_size", type=int, default=30, help="每次评估使用的样本数 (OPRO 建议 20-30)")
    p.add_argument("--train_json", type=str, default="data/bbh/merged_train.json")
    p.add_argument("--log_dir", type=str, default="logs_token_exp")
    return p.parse_args()


class OPROEvaluator:
    """完美复刻你原版 OPRO 的评估器：生成 -> 提取 -> 判分"""

    def __init__(self, kernel: TokenTrackedKernel):
        self.kernel = kernel

    def gen_prompt(self, question, instruction):
        """构造 OPRO 特有的带指令输入格式"""
        prompt = f"Q: {question}\n\nA:"
        if instruction:
            prompt += f" {instruction}"
        return prompt

    def batch_extract_answers(self, questions, raw_answers, task_configs):
        """用 LLM 从长回复中提取最终答案 (严格计费)"""
        if self.kernel.is_budget_exhausted(): return [""] * len(questions)

        extracted_results = [""] * len(questions)
        lock = threading.Lock()

        def _extract_single(idx, q, ans, t_conf):
            if self.kernel.is_budget_exhausted() or not ans: return
            is_bool, is_num = t_conf

            extract_prompt = (
                "You are an expert answer extractor. Your goal is to extract the FINAL ANSWER from the Model's Response.\n"
                "----------------\n"
                f"Original Question: {q}\n"
                "----------------\n"
                f"Model Response: {ans}\n"
                "----------------\n"
                "Instructions:\n"
            )

            if is_bool:
                extract_prompt += "1. Output ONLY 'Yes', 'No', 'True', or 'False'.\n"
            elif is_num:
                extract_prompt += "1. Output ONLY the number (e.g., 7, 3.14).\n"
            else:
                extract_prompt += "1. Output ONLY the option letter enclosed in parentheses like '(A)', '(B)' or short phrase.\n"

            extract_prompt += "2. Do NOT output any reasoning or labels. Just the value.\n\nExtracted Answer:"

            try:
                # 这一步同样计入 Token 消耗
                res = self.kernel.chat(system_msg="You are an expert answer extractor.", user_msg=extract_prompt,
                                       expect_json=False, temperature=0.0)
                with lock:
                    extracted_results[idx] = res.content.strip()
            except:
                pass

        # 并发提取
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(_extract_single, i, questions[i], raw_answers[i], task_configs[i]) for i in
                       range(len(questions))]
            concurrent.futures.wait(futures, timeout=60)

        return extracted_results

    def get_normalized_prediction(self, pred, is_num, is_bool):
        """基础清洗"""
        p = pred.strip()
        if is_bool:
            p = re.sub(r'[^a-zA-Z]', '', p).lower()
            if 'yes' in p or 'true' in p: return 'yes'
            if 'no' in p or 'false' in p: return 'no'
        elif is_num:
            m = re.search(r'-?\d+\.?\d*', p)
            if m: return m.group(0)
        else:
            m = re.findall(r'\(([A-Z])\)', p)
            if m: return m[-1].upper()
            m = re.findall(r'(?:answer|option) is[:\s]*(?:\()?([a-zA-Z0-9]+)', p, re.IGNORECASE)
            if m: return m[-1].lower()
            m = re.findall(r'\\boxed\{([a-zA-Z0-9]+)\}', p)
            if m: return m[-1].lower()
        return p.lower()

    def evaluate(self, instruction, data_pool, batch_size):
        if self.kernel.is_budget_exhausted() or not data_pool:
            return 0.0

        actual_batch = min(batch_size, len(data_pool))
        batch = random.sample(data_pool, actual_batch)

        questions = [item["input"] for item in batch]
        true_answers = [str(item["target"]).strip().lower() for item in batch]

        # OPRO 特有：区分题目类型以供提取器使用
        boolean_tasks = {"boolean_expressions", "causal_judgement", "formal_fallacies", "navigate",
                         "sports_understanding", "web_of_lies"}
        numeric_tasks = {"object_counting", "multistep_arithmetic_two"}
        task_configs = [(item.get("task", "") in boolean_tasks, item.get("task", "") in numeric_tasks) for item in
                        batch]

        # 1. 并发生成长回答 (Generation)
        raw_answers = [""] * len(batch)
        lock = threading.Lock()

        def _gen_single(idx, item):
            if self.kernel.is_budget_exhausted(): return
            try:
                prompt = self.gen_prompt(item["input"], instruction)
                res = self.kernel.chat(system_msg="You are a logical reasoning expert.", user_msg=prompt,
                                       expect_json=False, temperature=0.0)
                with lock:
                    raw_answers[idx] = res.content
            except:
                pass

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(_gen_single, i, batch[i]) for i in range(len(batch))]
            concurrent.futures.wait(futures, timeout=120)

        # 2. 并发提取最终答案 (Extraction)
        extracted_answers = self.batch_extract_answers(questions, raw_answers, task_configs)

        # 3. 本地清洗判分 (Scoring)
        correct = 0
        for i in range(len(batch)):
            is_bool, is_num = task_configs[i]
            norm_pred = self.get_normalized_prediction(extracted_answers[i], is_num, is_bool)
            norm_target = true_answers[i].replace('(', '').replace(')', '')

            if is_num:
                try:
                    if abs(float(norm_pred) - float(norm_target)) < 1e-5: correct += 1
                except:
                    if norm_pred == norm_target: correct += 1
            else:
                if norm_pred == norm_target or norm_target in norm_pred.split():
                    correct += 1

        return (correct / actual_batch) * 100 if actual_batch > 0 else 0.0


class OPROTokenDriven:
    def __init__(self, kernel: TokenTrackedKernel, evaluator: OPROEvaluator, train_data: list, args):
        self.kernel = kernel
        self.evaluator = evaluator
        self.train_data = train_data
        self.args = args
        self.history = []

    def gen_meta_prompt(self, few_shot_count=3):
        meta_prompt = (
            "Your task is to generate a UNIVERSAL system instruction <INS> that helps a model solve "
            "diverse logic, math, and common sense reasoning problems.\n"
            "The goal is to find one instruction that works well across ALL these different tasks.\n"
            "Below are some previous instructions with their average scores (0 to 100).\n"
        )

        sorted_ins = sorted(self.history, key=lambda x: x[1])
        for ins, score in sorted_ins[-15:]:
            meta_prompt += f"\ntext:\n{ins}\nscore:\n{score:.2f}\n"

        meta_prompt += "\nBelow are some examples of problems from different tasks. "
        meta_prompt += "Your instruction will be inserted at <INS>.\n"

        few_shots = random.sample(self.train_data, min(few_shot_count, len(self.train_data)))
        for item in few_shots:
            q = item.get("input", "")
            a = item.get("target", "")
            task = item.get("task", "Unknown")
            meta_prompt += f"\n[Task: {task}]\ninput:\nQ: {q}\nA: <INS>\noutput:\n{a}\n"

        meta_prompt += (
            "\n\nGenerate a NEW, general-purpose instruction <INS> that improves accuracy across all tasks. "
            "The instruction should begin with <INS> and end with </INS>."
        )
        return meta_prompt

    def run(self):
        print("\n🚀 [Token 驱动进化] OPRO 开始演化...")

        # 🌟 强行注入 SPE 的起跑线种子，确保实验绝对公平
        initial_instructions = [
            "Role: You are a brilliant mathematician and logician.\nTask: Solve the problem.",
            "Let's think step by step.",
            "Answer the question directly and concisely."
        ]

        best_prompt, best_score = "", 0.0

        print(f"{Fore.CYAN}--- Evaluating Initial Instructions ---{Fore.RESET}")
        for ins in initial_instructions:
            if self.kernel.is_budget_exhausted(): break
            score = self.evaluator.evaluate(ins, self.train_data, self.args.batch_size)
            self.history.append((ins, score))
            print(f"Initial: \"{ins[:50]}...\" | Score: {score:.1f}%")
            if score > best_score:
                best_score = score
                best_prompt = ins

        self.kernel.log_performance(best_score)

        generation = 0
        while not self.kernel.is_budget_exhausted():
            generation += 1
            print(
                f"\n{Style.BRIGHT}=== Iteration {generation} | 当前 Token: {self.kernel.total_tokens_consumed}/{self.kernel.token_budget} ==={Style.RESET_ALL}")

            meta_prompt = self.gen_meta_prompt()

            print(f"{Fore.BLUE}🧠 [Generating] LLM is thinking of a new instruction...{Fore.RESET}")
            res = self.kernel.chat(system_msg="You are an expert prompt optimizer.", user_msg=meta_prompt,
                                   expect_json=False, temperature=0.9)

            if self.kernel.is_budget_exhausted() or not res.content:
                break

            clean_output = re.sub(r'<think>.*?</think>', '', res.content, flags=re.DOTALL)
            generated_instructions = re.findall(r"<INS>(.*?)</INS>", clean_output, re.DOTALL)

            if not generated_instructions:
                generated_instructions = [clean_output.strip().split('\n')[-1]]

            for new_ins in generated_instructions:
                new_ins = new_ins.strip()
                if len(new_ins) < 3 or any(new_ins == x[0] for x in self.history) or self.kernel.is_budget_exhausted():
                    continue

                print(f"Evaluating New Prompt: \"{new_ins}\"")
                score = self.evaluator.evaluate(new_ins, self.train_data, self.args.batch_size)
                self.history.append((new_ins, score))
                print(f"  -> Score: {Fore.GREEN}{score:.1f}%{Fore.RESET}")

                if score > best_score:
                    best_score = score
                    best_prompt = new_ins
                    print(f"  {Fore.YELLOW}🎉 New Best Found!{Fore.RESET}")

            self.kernel.log_performance(best_score)
            print(f"{Fore.CYAN}📊 Current Global Best: {best_score:.1f}%{Fore.RESET}")

        return best_prompt, best_score


def main():
    import os
    try:
        from new_spe.my_api_key import inject_api_key
        inject_api_key()
    except ImportError:
        os.environ["DEEPSEEK_API_KEY"] = "sk-9c6929df4f5541eb94ad3af0c77ddfcc"

    args = parse_args()
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print(f"🚀 Token 效率公平对比实验启动")
    print(f"   - 测试基线: [OPRO (Optimization by PROmpting)]")
    print(f"   - 预算上限: {args.token_budget} Tokens")
    print("=" * 60 + "\n")

    cfg_path = "new_spe/config/apikey.txt" if Path("new_spe/config/apikey.txt").exists() else "config/apikey.txt"
    cfg = load_deepseek_config(cfg_path)

    kernel = TokenTrackedKernel(config=cfg, token_budget=args.token_budget, verbose=False)

    print(f"📖 正在加载混合训练集 ({args.train_json})...")
    with open(args.train_json, 'r', encoding='utf-8') as f:
        train_data = json.load(f)

    evaluator = OPROEvaluator(kernel)
    opro_runner = OPROTokenDriven(kernel, evaluator, train_data, args)

    start_time = time.time()
    best_prompt, best_score = opro_runner.run()
    run_time = time.time() - start_time

    print("\n" + "🎉" * 20)
    print(f"   实验总结报告: OPRO")
    print(f"   - 最终消耗 Token: {kernel.total_tokens_consumed}")
    print(f"   - 最佳训练准确率: {best_score:.2f}%")
    print(f"   - 运行总耗时: {run_time / 60:.1f} min")
    print("\n   👑 [最终最佳提示词 (Best Prompt)] 👑")
    print("-" * 50)
    print(best_prompt)
    print("-" * 50)
    print("🎉" * 20 + "\n")

    run_id = time.strftime("%Y%m%d_%H%M%S")
    save_path = Path(args.log_dir) / f"opro_curve_budget_{args.token_budget}_{run_id}.json"

    curve_data = {
        "experiment_meta": {
            "method": "opro",
            "token_budget": args.token_budget,
            "final_tokens_used": kernel.total_tokens_consumed,
            "best_acc_found": best_score,
            "best_prompt_text": best_prompt
        },
        "performance_curve": kernel.performance_curve
    }

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(curve_data, f, ensure_ascii=False, indent=4)

    print(f"📈 珍贵的【Token-性能曲线】数据已安全保存至: {save_path}")


if __name__ == "__main__":
    main()