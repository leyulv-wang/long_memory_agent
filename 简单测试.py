import os
import pandas as pd

# 设定要搜索的文件名
target_filename = "create_final_entities.parquet"

print(f"正在当前目录下搜索 {target_filename} ...")

found_paths = []

# 遍历当前目录及所有子目录
for root, dirs, files in os.walk("."):
    if target_filename in files:
        full_path = os.path.join(root, target_filename)
        found_paths.append(full_path)
        print(f"✅ 找到了！路径在: {full_path}")

if not found_paths:
    print("❌ 未找到文件。请确认你是否已经运行过 GraphRAG 的 indexing (索引) 步骤？")
else:
    # 如果找到了，自动读取第一个并打印列名
    print("\n" + "="*30)
    print("正在尝试读取第一个找到的文件...")
    try:
        df = pd.read_parquet(found_paths[0])
        print("【列名 (Columns)】:")
        print(list(df.columns))
        print("\n【前 2 行数据】:")
        print(df.head(2))
    except Exception as e:
        print(f"读取失败: {e}")