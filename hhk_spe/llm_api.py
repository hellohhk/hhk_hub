# llm_api.py
import json
import time
from typing import Optional
import openai # 假设使用兼容 OpenAI 格式的接口 (如 vLLM, DeepSeek API)

class LLMClient:
    """
    大模型 API 封装类。
    处理网络请求、超时重试以及结构化输出的解析。
    """
    def __init__(self, api_key: str, base_url: str, model_name: str, is_teacher: bool = False):
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name
        self.is_teacher = is_teacher # 标记是否为 Oracle 模型

    def generate(self, prompt: str, temperature: float = 0.0, max_tokens: int = 1024, max_retries: int = 3) -> Optional[str]:
        """基础的文本生成方法，带重试机制"""
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                print(f"[API Error] {self.model_name} 请求失败 (尝试 {attempt+1}/{max_retries}): {e}")
                time.sleep(2 ** attempt) # 指数退避重试
        return None

class TeacherOracle(LLMClient):
    """
    导师模型专用类 (如 DeepSeek-V3)
    对应论文 3.4.1: Diagnose-and-Refine Mutation 和 3.4.2: Contextual Knowledge Distillation
    """
    def __init__(self, api_key: str, base_url: str, model_name: str = "deepseek-chat"):
        super().__init__(api_key, base_url, model_name, is_teacher=True)

    def diagnose_and_refine(self, current_instruction: str, failed_question: str, wrong_answer: str, correct_answer: str) -> str:
        """
        核心算子 1：诊断并修正指令 (Error-Driven)
        """
        prompt = f"""You are an expert Prompt Engineer and Math Teacher.
A student model (8B parameters) used the following instruction but failed a math problem.

[Current Instruction]:
{current_instruction}

[Failed Question]: {failed_question}
[Student's Wrong Output]: {wrong_answer}
[Correct Ground Truth]: {correct_answer}

Task:
1. Diagnose WHY the student failed (e.g., missed a hidden condition, calculation error).
2. Rewrite the [Current Instruction] to prevent this specific type of error in the future. The new instruction should be generalizable, not just for this specific question.
3. OUTPUT ONLY THE REWRITTEN INSTRUCTION. Do not include your analysis.
"""
        # Teacher 需要一定的 temperature 来产生多样性的变异
        return self.generate(prompt, temperature=0.7)

    def generate_few_shot_cot(self, question: str, correct_answer: str) -> str:
        """
        核心算子 2：上下文知识蒸馏 (生成高质量的思维链解析)
        """
        prompt = f"""You are an expert Math Teacher.
Please provide a flawless, step-by-step Chain-of-Thought (CoT) solution for the following math problem.
The final answer MUST strictly be: {correct_answer}.

[Question]: {question}

Output your step-by-step reasoning clearly.
"""
        return self.generate(prompt, temperature=0.0) # 蒸馏知识需要严谨，温度设为0

class StudentModel(LLMClient):
    """
    学生模型专用类 (如 Llama-3-8B-Instruct)
    负责在 Evaluator 中做题。
    """
    def __init__(self, api_key: str, base_url: str, model_name: str = "llama-3-8b-instruct"):
        super().__init__(api_key, base_url, model_name, is_teacher=False)

    def solve_math_problem(self, assembled_prompt: str) -> str:
        """接收组装好的 prompt (来自 PromptGenome)，输出解答"""
        # 学生做题时也需要严密，温度通常设为 0 (Greedy Decoding)
        return self.generate(assembled_prompt, temperature=0.0, max_tokens=512)

# ==========================================
# 太爷，测试代码在下面，您可以单独运行这个文件试试！
if __name__ == "__main__":
    # 【注意】运行前需要填入您真实的 API Key 和 Base URL
    # 如果 Llama-3 是本地 vLLM 部署，URL 可能是 http://localhost:8000/v1
    
    # teacher = TeacherOracle(api_key="your_deepseek_key", base_url="https://api.deepseek.com/v1")
    # student = StudentModel(api_key="your_llama_key", base_url="http://localhost:8000/v1")
    
    print("API 模块加载完毕。等待太爷指示！")