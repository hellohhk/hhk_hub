# prompt_genome.py
import json
from typing import List, Dict

class PromptGenome:
    """
    结构化提示词基因组 (Structured Prompt Genome)
    对应论文 3.3 节：将 Prompt 解耦为 Role, Instruction, Examples, Style 四个基因位点。
    """
    def __init__(self, role: str = "", instruction: str = "", examples: List[Dict] = None, style: str = ""):
        self.role = role
        self.instruction = instruction
        self.examples = examples if examples is not None else [] # 格式: [{"question": "...", "answer": "..."}]
        self.style = style
        
        # 记录该个体的表现（用于 UCB 调度和进化选择）
        self.history_scores = [] 
        self.evaluated_count = 0
        self.average_score = 0.0

    def update_score(self, new_score: float):
        """更新该基因组的准确率得分，用于 UCB 计算"""
        self.history_scores.append(new_score)
        self.evaluated_count += 1
        self.average_score = sum(self.history_scores) / self.evaluated_count

    def render_examples(self) -> str:
        """将 Few-shot 样例渲染为字符串"""
        if not self.examples:
            return ""
        
        example_str = "Here are some examples for your reference:\n"
        for i, ex in enumerate(self.examples):
            example_str += f"--- Example {i+1} ---\nQuestion: {ex['question']}\nAnswer: {ex['answer']}\n\n"
        return example_str

    def build_prompt(self, target_question: str) -> str:
        """
        组装最终发送给大模型的字符串
        """
        components = []
        
        if self.role:
            components.append(f"System Role: {self.role}")
        
        if self.instruction:
            components.append(f"Instruction: {self.instruction}")
            
        if self.style:
            components.append(f"Output Style: {self.style}")

        examples_text = self.render_examples()
        if examples_text:
            components.append(examples_text)

        # 拼接当前需要解答的问题
        components.append(f"Now, please solve the following target question:\nQuestion: {target_question}\nAnswer:")

        # 用换行符连接各个模块
        return "\n\n".join(components)

    def to_dict(self) -> Dict:
        """序列化，方便保存实验日志"""
        return {
            "role": self.role,
            "instruction": self.instruction,
            "examples": self.examples,
            "style": self.style,
            "evaluated_count": self.evaluated_count,
            "average_score": self.average_score
        }
        
    def __repr__(self):
        return f"<PromptGenome | Score: {self.average_score:.2f} | Evals: {self.evaluated_count}>"

# 测试一下是否能正常工作
if __name__ == "__main__":
    # 初始化一个种子基因组 (Seed Genome)
    seed_genome = PromptGenome(
        role="You are an expert Math Olympiad Tutor.",
        instruction="Think step-by-step. List known variables first, then set up the equations.",
        examples=[{"question": "What is 2+2?", "answer": "Let's think step by step. 2 + 2 equals 4. The answer is 4."}],
        style="Be concise and precise."
    )
    
    # 模拟给模型发问题
    final_prompt = seed_genome.build_prompt("A train travels at 60mph. How far does it go in 2 hours?")
    print("=== 拼装好的最终 Prompt ===")
    print(final_prompt)