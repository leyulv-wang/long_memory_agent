import os


def print_tree(startpath):
    # topdown=True 是默认值，但在修改 dirs 时必须显式保证是从上到下遍历
    for root, dirs, files in os.walk(startpath, topdown=True):

        # --- 修改部分开始 ---
        # 1. 过滤目录：移除 __pycache__ (以及 .git, .idea 等其他你想隐藏的目录)
        # 注意：必须使用 dirs[:] = ... 进行原地修改，这样 os.walk 才会跳过这些目录
        dirs[:] = [d for d in dirs if d not in ['__pycache__', '.git', '.idea']]

        # 2. 过滤文件：不显示 .pyc 文件 (可选，让视图更干净)
        files = [f for f in files if not f.endswith('.pyc')]
        # --- 修改部分结束 ---

        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * level  # 我稍微增加了缩进宽度(4空格)，看起来更清晰

        # 打印当前文件夹名称
        print(f'{indent}{os.path.basename(root)}/')

        subindent = ' ' * 4 * (level + 1)
        # 打印文件
        for f in files:
            print(f'{subindent}├── {f}')


# 替换成你的路径
path = r"D:\python\智能体记忆框架"
print_tree(path)