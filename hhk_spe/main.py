# main.py
from prompt_genome import PromptGenome
from llm_api import StudentModel, TeacherOracle
from evaluator import GSM8KEvaluator
from evolver import PromptEvolver
from scheduler import UCBScheduler

def load_mock_dataset():
    """模拟加载 GSM8K 训练集"""
    return [
        {"question": "If John has 5 apples and eats 2, how many are left?", "answer": "3"},
        {"question": "A car travels 60 miles in 1 hour. How far in 2.5 hours?", "answer": "150"},
        {"question": "What is 15% of 200?", "answer": "30"},
        {"question": "If 3x = 12, what is x?", "answer": "4"},
        {"question": "The perimeter of a square is 20. What is its area?", "answer": "25"}
    ]

def run_spe_experiment(max_api_budget: int = 50, batch_size: int = 2):
    """
    SPE 框架主循环
    """
    print("=== 🚀 启动结构化提示词进化 (SPE) 实验 ===")
    
    # 1. 初始化模型与各个模块
    # 注意：实际运行需填入真实的 API 密钥
    student = StudentModel(api_key="fake_key", base_url="fake_url", model_name="llama-3-8b")
    teacher = TeacherOracle(api_key="fake_key", base_url="fake_url", model_name="deepseek-chat")
    
    evaluator = GSM8KEvaluator(student)
    evolver = PromptEvolver(teacher)
    scheduler = UCBScheduler(c_param=0.5)
    
    dataset = load_mock_dataset()
    
    # 2. 初始化种子基因组 (Initialization)
    seed_genome = PromptGenome(
        role="You are a meticulous Math Expert.",
        instruction="Please solve the math problem clearly. End your response with 'The answer is: [number]'.",
        style="Clear and step-by-step."
    )
    
    # 可选：蒸馏几道题作为 Few-Shot Examples (知识蒸馏)
    # evolver.distill_few_shot_examples(seed_genome, dataset, num_shots=1)
    
    scheduler.add_genome(seed_genome)
    
    current_budget_used = 0
    generation = 1
    
    # 3. 核心进化循环 (基于预算约束)
    while current_budget_used < max_api_budget:
        print(f"\n--- [Generation {generation} | Budget Used: {current_budget_used}/{max_api_budget}] ---")
        
        # A. 调度器选择下一个评估对象 (UCB Selection)
        target_genome = scheduler.select_next()
        print(f"[{'探索' if target_genome.evaluated_count==0 else '利用'}] 选中基因组，当前均分: {target_genome.average_score:.2f}")
        
        # B. 评估与错题收集 (Evaluation)
        acc, failures = evaluator.evaluate_genome(target_genome, dataset, sample_size=batch_size)
        scheduler.update_global_step(batch_size)
        current_budget_used += batch_size
        print(f"评估完成！本批次准确率: {acc*100}%，收集到 {len(failures)} 道错题。")
        
        # C. 错误驱动变异 (Evolution)
        if failures:
            # 如果有错题，让老师去诊断并生成新的提示词
            child_genome = evolver.diagnose_and_mutate(target_genome, failures)
            if child_genome.instruction != target_genome.instruction:
                scheduler.add_genome(child_genome)
                print(f"🎉 成功繁育新子代加入种群！新指令预览: {child_genome.instruction[:50]}...")
        
        generation += 1

    # 4. 实验结束，输出最优结果
    best_genome = scheduler.get_best_genome()
    print("\n" + "="*50)
    print("🏆 进化完成！预算已耗尽。")
    print(f"最优 Prompt 平均准确率: {best_genome.average_score*100:.1f}%")
    print("最优 Prompt 结构:")
    print(f"- Role: {best_genome.role}")
    print(f"- Instruction: {best_genome.instruction}")
    print("="*50)

if __name__ == "__main__":
    # 为了跑通测试流程，您可以在本地将 llm_api.py 里的接口替换成模拟返回，或者填入真实 API 测试。
    run_spe_experiment(max_api_budget=10, batch_size=2)