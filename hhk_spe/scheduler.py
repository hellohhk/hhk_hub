# scheduler.py
import math
from typing import List
from prompt_genome import PromptGenome

class UCBScheduler:
    """
    基于 UCB (Upper Confidence Bound) 的置信度感知调度器。
    用于在有限预算下，平衡“探索（没怎么测过的提示词）”与“利用（已经表现很好的提示词）”。
    """
    def __init__(self, c_param: float = 0.5):
        self.population: List[PromptGenome] = []
        self.total_evaluations = 0
        self.c_param = c_param # UCB 公式中的探索常数 C

    def add_genome(self, genome: PromptGenome):
        """将新的基因组（比如刚变异出来的子代）加入种群"""
        self.population.append(genome)

    def select_next(self) -> PromptGenome:
        """
        核心算子：根据 UCB 公式选择下一个最值得评估的个体。
        公式: S_i = v_i + C * sqrt( (2 * ln(N)) / n_i )
        """
        if not self.population:
            raise ValueError("种群为空，无法选择！")

        best_score = -float('inf')
        best_genome = None

        for genome in self.population:
            # 如果这个个体一次都没测过，赋予无限大的优先级，强制探索
            if genome.evaluated_count == 0:
                return genome

            # 计算 Exploitation (利用) 项: 历史平均准确率
            exploitation = genome.average_score
            
            # 计算 Exploration (探索) 项: 测得越少，该项越大
            exploration = self.c_param * math.sqrt(
                (2 * math.log(self.total_evaluations)) / genome.evaluated_count
            )
            
            # 综合得分
            ucb_score = exploitation + exploration
            
            if ucb_score > best_score:
                best_score = ucb_score
                best_genome = genome

        return best_genome

    def update_global_step(self, batch_size: int):
        """每次评估完一个 Batch，更新全局评估次数 N"""
        self.total_evaluations += batch_size

    def get_best_genome(self) -> PromptGenome:
        """返回目前为止平均分最高的个体（仅看实际得分，不加探索项）"""
        if not self.population:
            return None
        # 过滤掉还没测过的
        evaluated_genomes = [g for g in self.population if g.evaluated_count > 0]
        if not evaluated_genomes:
            return None
        return max(evaluated_genomes, key=lambda g: g.average_score)

# ==========================================
# 太爷，测试代码在下面！
if __name__ == "__main__":
    scheduler = UCBScheduler(c_param=0.5)
    
    # 模拟三个提示词
    g1 = PromptGenome(instruction="Prompt A")
    g1.evaluated_count = 10
    g1.average_score = 0.8  # 分数高，测得多
    
    g2 = PromptGenome(instruction="Prompt B")
    g2.evaluated_count = 2
    g2.average_score = 0.6  # 分数低，测得少
    
    g3 = PromptGenome(instruction="Prompt C") 
    # g3 刚出生，没测过
    
    scheduler.add_genome(g1)
    scheduler.add_genome(g2)
    scheduler.add_genome(g3)
    scheduler.total_evaluations = 12
    
    print("第一次选择 (应该选 C，因为没测过):", scheduler.select_next().instruction)
    
    # 假装 C 测完了，得了 0.5 分
    g3.evaluated_count = 1
    g3.average_score = 0.5
    scheduler.update_global_step(1)
    
    print("第二次选择 (UCB 算法开始权衡 A 和 B):", scheduler.select_next().instruction)