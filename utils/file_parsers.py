import json
from pathlib import Path


def parse_book_to_dict(file_path: str) -> dict:
    """
    读取具有特定键/值格式的 .txt 文件并将其转换为字典。

    文件格式规则:
    - 键(key)占一行。
    - 值(value)是接下来的所有行，直到遇到一个空行。
    - 多行的值将被存储为字符串列表。
    - 单行的值将被存储为单个字符串。

    Args:
        file_path (str): 要解析的文件的路径。

    Returns:
        dict: 从文件中解析出的结构化数据。
    """
    # 使用 pathlib 读取文件，更现代且健壮
    content = Path(file_path).read_text(encoding='utf-8').strip()
    structured_data = {}

    # 使用两个换行符（即一个空行）来分割各个数据块
    blocks = content.split('\n\n')

    for block in blocks:
        lines = block.strip().split('\n')
        key = lines[0].strip()
        # 获取所有非空的 value 行
        values = [line.strip() for line in lines[1:] if line.strip()]

        # 根据 value 的行数决定是存为字符串还是列表
        if len(values) == 1:
            structured_data[key] = values[0]
        else:
            structured_data[key] = values

    return structured_data
