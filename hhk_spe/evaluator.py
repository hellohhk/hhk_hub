# evaluator.py
import re
from typing import List, Dict, Tuple
from prompt_genome import PromptGenome
from llm_api import StudentModel

class GSM8KEvaluator:
    """
    GSM8K 数据集评估器。
    负责在验证集/训练集上测试 Prompt 的效果，并收集错题。
    """
    def __init__(self, student_model: StudentModel):
        self.student = student_model

    def extract_answer(self, response_text: str) -> str:
        """
        从模型输出的思维链 (CoT) 中提取最终的数学答案。
        (实际论文代码中，这里通常用更复杂的正则匹配，这里做一个简化版的演示)
        """
        # 假设模型被 Instruction 约束为以 "The answer is: [数字]" 结尾
        match = re.search(r'The answer is[:\s]*([0-9\.\-\/]+)', response_text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        # Fallback: 提取文本中的最后一个数字
        numbers = re.findall(r'-?\d+(?:\.\d+)?', response_text)
        if numbers:
            return numbers[-1]
        return ""

    def check_exact_match(self, prediction: str, ground_truth: str) -> bool:
        """
        论文 3.1 节定义的数学等价性判定 (Mathematical Equivalence)
        """
        # 去除多余的空格和逗号 (例如 1,000 -> 1000)
        pred = prediction.replace(",", "").strip()
        gt = ground_truth.replace(",", "").strip()
        
        try:
            # 尝试转换为浮点数进行精确比对，避免 "2.0" 和 "2" 匹配失败
            return abs(float(pred) - float(gt)) < 1e-6
        except ValueError:
            # 如果不是纯数字，则退化为字符串比对
            return pred.lower() == gt.lower()

    def evaluate_genome(self, genome: PromptGenome, dataset: List[Dict], sample_size: int = 10) -> Tuple[float, List[Dict]]:
        """
        核心方法：评估一个基因组，返回准确率和错题本。
        
        Args:
            genome: 待评估的提示词个体
            dataset: GSM8K 数据集列表 [{"question": "...", "answer": "123"}]
            sample_size: UCB 调度时，每次评估的 Batch Size（为了省钱，通常是小批量测试）
            
        Returns:
            accuracy (float): 准确率 [0.0, 1.0]
            failure_cases (List[Dict]): 错题本列表
        """
        correct_count = 0
        failure_cases = []
        
        # 实际代码中，应该随机打乱或根据 UCB 策略选择未测过的样本
        test_batch = dataset[:sample_size] 
        
        print(f"开始评估个体... Batch Size: {sample_size}")
        
        for item in test_batch:
            question = item['question']
            ground_truth_answer = item['answer']
            
            # 1. 组装 Prompt
            prompt = genome.build_prompt(question)
            
            # 2. 调用学生模型解答
            model_output = self.student.solve_math_problem(prompt)
            if not model_output:
                continue # API 失败跳过
                
            # 3. 提取答案并比对
            predicted_answer = self.extract_answer(model_output)
            is_correct = self.check_exact_match(predicted_answer, ground_truth_answer)
            
            if is_correct:
                correct_count += 1
            else:
                # 4. 收集错题 (极其关键！这是下一步 Error-Driven 的燃料)
                failure_cases.append({
                    "question": question,
                    "wrong_output": model_output,
                    "ground_truth": ground_truth_answer
                })
                
        # 5. 计算得分并更新基因组的历史记录
        accuracy = correct_count / sample_size if sample_size > 0 else 0.0
        genome.update_score(accuracy)
        
        return accuracy, failure_cases

# ==========================================
# 太爷，测试代码在下面！
if __name__ == "__main__":
    # 模拟一个极小的 GSM8K 验证集
    mock_dataset = [
        {"question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?", "answer": "72"},
        {"question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?", "answer": "10"}
    ]
    
    # 模拟不需要真实调用的 Student，只看逻辑通不通
    class MockStudent(StudentModel):
        def solve_math_problem(self, prompt: str) -> str:
            # 故意答错一题，测试错题收集功能
            if "Natalia" in prompt:
                return "Let's think. 48 + 48/2 = 72. The answer is 72." # 正确
            else:
                return "12 * 50 = 600. The answer is 600." # 错误逻辑
                
    dummy_student = MockStudent(api_key="fake", base_url="fake")
    evaluator = GSM8KEvaluator(dummy_student)
    
    # 拿之前写好的基因组来测
    test_genome = PromptGenome(instruction="Solve the math problem.")
    
    acc, failures = evaluator.evaluate_genome(test_genome, mock_dataset, sample_size=2)
    
    print(f"\n评估完成！")
    print(f"准确率: {acc*100}%")
    print(f"收集到错题数量: {len(failures)}")
    if failures:
        print(f"第一道错题的错误输出: {failures[0]['wrong_output']}")