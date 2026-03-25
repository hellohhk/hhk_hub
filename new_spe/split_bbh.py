import json
import os
import random
from pathlib import Path


def split_bbh_dataset(source_dir, train_dir, test_dir, train_ratio=0.7, seed=42):
    """
    按比例拆分 BBH 数据集，并保持原有的 JSON 格式
    """
    source_path = Path(source_dir)
    train_path = Path(train_dir)
    test_path = Path(test_dir)

    # 1. 如果原始数据不存在，提示错误
    if not source_path.exists():
        print(f"❌ 错误: 找不到源数据目录 '{source_path}'")
        print("请确认 BBH 数据是否已经下载并解压到该位置。")
        # 导师的代码通常会把数据下到 data/bbh/bbh
        if Path("data/bbh/bbh").exists():
            print("💡 提示: 发现数据在 'data/bbh/bbh'，请修改 SOURCE_DIR 参数。")
        return

    # 2. 创建输出目录
    train_path.mkdir(parents=True, exist_ok=True)
    test_path.mkdir(parents=True, exist_ok=True)

    json_files = list(source_path.glob("*.json"))
    if not json_files:
        print(f"❌ 错误: 在 '{source_path}' 目录下没有找到任何 .json 文件。")
        return

    print(
        f"📂 找到 {len(json_files)} 个任务文件，准备按 {train_ratio * 100:.0f}:{(1 - train_ratio) * 100:.0f} 比例拆分...")

    # 固定随机种子，保证每次拆分的结果完全一致，这对学术实验非常重要
    random.seed(seed)
    total_train = 0
    total_test = 0

    # 3. 遍历每一个任务文件
    for file in json_files:
        with open(file, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"⚠️ 警告: 文件 {file.name} 不是合法的 JSON，已跳过。")
                continue

        examples = data.get("examples", [])
        if not examples:
            continue

        # 核心：打乱数据
        random.shuffle(examples)

        # 核心：计算切分点
        split_idx = int(len(examples) * train_ratio)

        train_examples = examples[:split_idx]
        test_examples = examples[split_idx:]

        total_train += len(train_examples)
        total_test += len(test_examples)

        # 构造新的 JSON 数据 (保留 name 和 description，替换 examples)
        train_data = {
            "name": data.get("name", file.stem),
            "description": data.get("description", ""),
            "examples": train_examples
        }

        test_data = {
            "name": data.get("name", file.stem),
            "description": data.get("description", ""),
            "examples": test_examples
        }

        # 4. 保存到各自的目录
        with open(train_path / file.name, 'w', encoding='utf-8') as f:
            # indent=4 保证输出的 JSON 格式漂亮且易读
            json.dump(train_data, f, ensure_ascii=False, indent=4)

        with open(test_path / file.name, 'w', encoding='utf-8') as f:
            json.dump(test_data, f, ensure_ascii=False, indent=4)

    print("-" * 50)
    print("✅ BBH 数据集物理拆分完成！")
    print(f"   🔹 训练集 (Train) 题目总数: {total_train} -> 保存在 '{train_path}'")
    print(f"   🔹 测试集 (Test)  题目总数: {total_test} -> 保存在 '{test_path}'")
    print("-" * 50)


if __name__ == "__main__":
    # 配置路径：根据你实际的数据存放位置调整
    # 导师之前的脚本下载解压后，路径通常是 data/bbh/bbh
    SOURCE_DIR = "data/bbh"

    # 拆分后的存放位置
    TRAIN_DIR = "data/bbh/train"
    TEST_DIR = "data/bbh/test"

    split_bbh_dataset(SOURCE_DIR, TRAIN_DIR, TEST_DIR, train_ratio=0.7)