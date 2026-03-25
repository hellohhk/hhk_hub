# evolver.py
import random
import copy
from typing import List, Dict
from prompt_genome import PromptGenome
from llm_api import TeacherOracle

class PromptEvolver:
    """
    提示词进化引擎。
    对应论文 3.4 节：Evolutionary Operators (进化算子)
    包含核心的“名师订正”逻辑和“模块化交叉”逻辑。
    """
    def __init__(self, teacher_model: TeacherOracle):
        self.teacher = teacher_model

    def diagnose_and_mutate(self, parent_genome: PromptGenome, failure_cases: List[Dict]) -> PromptGenome:
        """
        核心算子 1：Diagnose-and-Refine Mutation (诊断-修正变异)
        拿着 Student 做错的题目，请 Teacher 进行诊断，并重写 Instruction。
        """
        if not failure_cases:
            # 如果没有错题（满分），则只做轻微的同义词变异，或者直接保留
            return copy.deepcopy(parent_genome)

        # 1. 采样最具代表性的错题（为了节省 Token，通常每次只挑 1-2 道错题给老师看）
        sample_fail = random.choice(failure_cases)
        failed_q = sample_fail['question']
        wrong_a = sample_fail['wrong_output']
        correct_a = sample_fail['ground_truth']

        print(f"[Mutation] Teacher 正在诊断错题...\n题目: {failed_q[:50]}...")

        # 2. 调用 Teacher 进行诊断和重写
        new_instruction = self.teacher.diagnose_and_refine(
            current_instruction=parent_genome.instruction,
            failed_question=failed_q,
            wrong_answer=wrong_a,
            correct_answer=correct_a
        )

        # 3. 创建新的后代基因组
        child_genome = copy.deepcopy(parent_genome)
        if new_instruction:
            # 只更新 Instruction 基因位点，其他保持不变（结构化进化的优势）
            child_genome.instruction = new_instruction
            
        # 清空历史得分，因为这是一个全新的个体
        child_genome.history_scores = []
        child_genome.evaluated_count = 0
        child_genome.average_score = 0.0

        return child_genome

    def distill_few_shot_examples(self, target_genome: PromptGenome, train_dataset: List[Dict], num_shots: int = 3):
        """
        核心算子 2：Contextual Knowledge Distillation (上下文知识蒸馏)
        让 Teacher 模型生成完美的思维链 (CoT) 解析，作为 Examples 注入到基因组中。
        """
        print(f"[Distillation] 正在抽取 {num_shots} 道题生成完美 CoT 范例...")
        sampled_data = random.sample(train_dataset, num_shots)
        new_examples = []

        for item in sampled_data:
            q = item['question']
            ans = item['answer']
            
            # 让老师写出完美的解答步骤
            perfect_cot = self.teacher.generate_few_shot_cot(question=q, correct_answer=ans)
            
            if perfect_cot:
                new_examples.append({
                    "question": q,
                    "answer": perfect_cot # 此时的 answer 已经包含了详细的推理过程
                })

        # 更新基因位点
        target_genome.examples = new_examples
        print("[Distillation] 范例注入完成！")

    def crossover(self, genome_a: PromptGenome, genome_b: PromptGenome) -> PromptGenome:
        """
        附加算子：Crossover (模块化交叉)
        从两个高分个体中，随机交换部分基因位点（比如把 A 的 Role 和 B 的 Instruction 拼在一起）。
        这是纯文本优化（如 OPRO）做不到的，只有结构化基因才能做！
        """
        child = copy.deepcopy(genome_a)
        
        # 抛硬币决定每个基因位点继承自哪一个父代
        if random.random() > 0.5:
            child.role = genome_b.role
        if random.random() > 0.5:
            child.instruction = genome_b.instruction
        if random.random() > 0.5:
            child.style = genome_b.style
            
        # 注意：交叉产生的新个体也需要清空历史得分
        child.history_scores = []
        child.evaluated_count = 0
        child.average_score = 0.0
        
        return