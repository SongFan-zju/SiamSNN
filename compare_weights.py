import pickle
import re
import sys


def read_numbers_from_txt(file_path):
    """
    从 .txt 文件中读取数字。
    格式为 "1,数字"，只提取逗号后的数字。
    """
    numbers = []
    print(f"1. 正在处理 .txt 文件: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_number, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue  # 跳过空行

                # 预期格式是 "1,数字"，使用 split(',') 来分割
                parts = line.split(',', 1)
                if len(parts) == 2:
                    # 提取逗号后的部分，并尝试转换为浮点数
                    number_str = parts[1].strip()
                    try:
                        numbers.append(float(number_str))
                    except ValueError:
                        # 忽略无法转换为数字的行
                        print(f"   警告: 第 {line_number} 行中 '{number_str}' 不是有效的数字，已跳过。")
                else:
                    # 格式不符的行，例如没有逗号
                    print(f"   警告: 第 {line_number} 行格式不正确 ('{line}')，已跳过。")

    except FileNotFoundError:
        print(f"错误: 找不到 .txt 文件 {file_path}")
        sys.exit(1)
    except Exception as e:
        print(f"处理 .txt 文件时发生错误: {e}")
        sys.exit(1)

    print(f"   成功从 .txt 文件中提取 {len(numbers)} 个数字。")
    return numbers


def extract_numbers_from_pkl(file_path):
    """
    加载 .pkl 文件，并从其内容中提取所有数字（整数和浮点数）。
    """
    numbers = []
    print(f"\n2. 正在处理 .pkl 文件: {file_path}")

    # 2.1 加载 .pkl 文件
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

    except FileNotFoundError:
        print(f"错误: 找不到 .pkl 文件 {file_path}")
        sys.exit(1)
    except pickle.UnpicklingError as e:
        print(f"错误: 无法解封 .pkl 文件，可能文件已损坏或使用了不兼容的 Python 版本/库。详情: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"加载 .pkl 文件时发生未知错误: {e}")
        sys.exit(1)

    # 2.2 将数据转换为字符串并提取数字
    # 注意：这种通过将整个数据结构转为字符串再用正则提取的方法，
    # 可能会在复杂结构（如嵌套字典或列表）中遗漏数字，或提取到非期望的数字（如索引、键名）。
    # 如果数据结构已知，直接遍历结构会更精确。
    data_str = str(data)

    # 正则表达式用于匹配整数和浮点数（包括负数）
    number_pattern = r'-?\d+\.?\d*|\-?\.?\d+'
    extracted_numbers_str = re.findall(number_pattern, data_str)

    # 清理并转换为浮点数
    for num_str in extracted_numbers_str:
        # 排除只匹配到小数点或负号的空字符串
        if num_str in ['.', '-', '', '..']:
            continue
        try:
            numbers.append(float(num_str))
        except ValueError:
            pass

    print(f"   成功从 .pkl 文件中提取 {len(numbers)} 个数字。")
    return numbers


def compare_number_lists(list1, list2):
    """
    比较两个数字列表，判断它们是否完全相等。
    """
    print("\n3. 正在比较两个数字列表...")

    # 首先检查长度是否一致
    if len(list1) != len(list2):
        print(f"   列表长度不一致: .txt 提取 {len(list1)} 个, .pkl 提取 {len(list2)} 个。")
        return 1  # 长度不一致，肯定不同

    # 逐个比较数字。由于是浮点数，需要考虑到精度问题，使用容忍度（tolerance）进行比较
    TOLERANCE = 1e-9

    for i in range(len(list1)):
        num1 = list1[i]
        num2 = list2[i]

        # 使用 abs(a - b) > TOLERANCE 来判断两个浮点数是否"不相等"
        if abs(num1 - num2) > TOLERANCE:
            print(f"   发现不一致: 第 {i+1} 个数字不同:")
            print(f"     .txt 文件中的值: {num1}")
            print(f"     .pkl 文件中的值: {num2}")
            return 1  # 发现任何不一致，立即返回 1

    return 0  # 所有数字都一致


# --- 主程序入口 (预设路径版) ---
if __name__ == "__main__":

    # =========================================================================
    """ 
    🚨 请替换为你的实际文件路径 
    使用 r'...' 格式避免 Windows 路径反斜杠问题
    """
    TXT_FILE = r'./pcss_output/saimsnn_wgts.txt'
    PKL_FILE = r'track_weight.pkl'

    print("--- 文件数字对比程序启动 ---")

    # 1. 从 .txt 文件提取数字
    txt_numbers = read_numbers_from_txt(TXT_FILE)

    # 2. 从 .pkl 文件提取数字
    pkl_numbers = extract_numbers_from_pkl(PKL_FILE)

    # 3. 比较结果
    result_code = compare_number_lists(txt_numbers, pkl_numbers)

    print("\n--- 对比结果 ---")
    if result_code == 0:
        print("WDR=0: 两个文件中的所有对应数字都相等。")
    else:
        print("WDR=1: 发现不一致的数字或列表长度不同。")

    # sys.exit(result_code) # 根据需要决定是否让程序以 DIFF 状态码退出
