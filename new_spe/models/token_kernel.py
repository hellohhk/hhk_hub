import threading
from typing import Any, Dict, Optional

# 导入你现有的基础组件
from new_spe.models.deepseek_kernel import DeepSeekKernel, KernelCallResult
from new_spe.utils.config_loader import DeepSeekConfig


class TokenTrackedKernel(DeepSeekKernel):
    """
    继承自 DeepSeekKernel，专门用于 Token 效率对比实验。
    内置线程安全的 Token 累加器和硬性预算阻断（熔断）机制。
    """

    def __init__(self, config: DeepSeekConfig, token_budget: int, verbose: bool = False):
        # 初始化父类，继承所有基础通信能力
        super().__init__(config, verbose)

        # 🌟 Token 追踪核心组件
        self.token_budget = token_budget
        self.total_tokens_consumed = 0
        self.token_lock = threading.Lock()

        # 🌟 用于记录【消耗的 Token vs 当前最高准确率】的曲线数据
        # 数据格式预期: [{"tokens_used": 15000, "best_acc": 0.45}, ...]
        self.performance_curve = []

    def is_budget_exhausted(self) -> bool:
        """检查 Token 预算是否已经耗尽"""
        with self.token_lock:
            return self.total_tokens_consumed >= self.token_budget

    def log_performance(self, current_best_acc: float):
        """记录当前消耗下的最佳性能，用于后期绘制 Token-效率 曲线图"""
        with self.token_lock:
            self.performance_curve.append({
                "tokens_used": self.total_tokens_consumed,
                "best_acc": current_best_acc
            })

    def chat(
            self,
            system_msg: str,
            user_msg: str,
            *,
            expect_json: bool,
            stream: bool = True,
            temperature: Optional[float] = None,
            extra: Optional[Dict[str, Any]] = None,
    ) -> KernelCallResult:
        """重写 chat 方法，在发起请求前后拦截并记录 Token 消耗"""

        # 1. 拦截器：发起请求前，检查预算是否已经枯竭
        if self.is_budget_exhausted():
            if self.verbose:
                print("⚠️ [拦截] Token 预算已耗尽，拒绝发起新的 API 请求。")
            return KernelCallResult(content="", raw={"error": "Token Budget Exhausted!"})

        # 2. 核心调用：完美复用父类原本的请求逻辑 (包含你的流式获取和 JSON 解析)
        result = super().chat(
            system_msg=system_msg,
            user_msg=user_msg,
            expect_json=expect_json,
            stream=stream,
            temperature=temperature,
            extra=extra
        )

        # 3. 记账员：从返回结果中提取 usage 并累加 total_tokens
        # （得益于你之前在 DeepSeekKernel 中精准拦截了 usage 数据）
        if result.usage and "total_tokens" in result.usage:
            with self.token_lock:
                self.total_tokens_consumed += result.usage["total_tokens"]

        return result