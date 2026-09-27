import os
import csv
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


def to_hex2(val):
    """
    将数值转换为 2 位大写十六进制字符串，不足前面补 0。
    支持整数或可以转换为整数的字符串。
    """
    i = int(val)
    if not (0 <= i <= 0xFF):
        raise ValueError(f"值 {i} 超出 0–255 范围，无法用 2 位十六进制表示")
    return f"{i:02X}"


def process_file(csv_path, txt_path):
    """
    对单个 CSV 文件进行处理，按规则写入 TXT。
    """
    os.makedirs(os.path.dirname(txt_path), exist_ok=True)
    with open(csv_path, newline='', encoding='utf-8') as fin, \
         open(txt_path, 'w', encoding='utf-8') as fout:
        reader = csv.reader(fin)
        for row in reader:
            if len(row) < 4:
                print(f"[{threading.current_thread().name}] xxxxxxx (file: {os.path.basename(csv_path)})")
                continue  # 跳过不完整行
            tik, z_str, y_str, x_str = row[0], row[1], row[2], row[3]
            # 转成 2 位十六进制
            hx = to_hex2(x_str)
            hy = to_hex2(y_str)
            hz = to_hex2(z_str)
            # 构建两行
            line1 = f"00090ff000{hz}{hy}{hx}"
            # line2 = f"0009083000{hz}{hy}{hx}"
            fout.write(line1 + "\n")
            # fout.write(line2 + "\n")
        fout.write('0300000000100000\n')
        fout.write('0300000100100000')


def get_csv_files(input_root):
    """获取所有CSV文件路径"""
    csv_files = []
    for dirpath, dirnames, filenames in os.walk(input_root):
        for fname in filenames:
            if fname.lower().endswith('.csv'):
                csv_files.append(os.path.join(dirpath, fname))
    return csv_files


def process_single_file(args):
    """处理单个文件的包装函数，用于线程池"""
    in_csv, out_txt = args
    try:
        process_file(in_csv, out_txt)
        return f"[OK] {in_csv} → {out_txt}"
    except Exception as e:
        return f"[ERROR] {in_csv} → {out_txt}, 错误: {str(e)}"


def main(input_root, output_root, max_workers=4):
    # 获取所有CSV文件
    csv_files = get_csv_files(input_root)

    if not csv_files:
        print("未找到任何CSV文件")
        return

    print(f"找到 {len(csv_files)} 个CSV文件，开始并行处理...")

    # 准备参数列表
    tasks = []
    for in_csv in csv_files:
        rel = os.path.relpath(in_csv, input_root)
        out_txt = os.path.splitext(os.path.join(output_root, rel))[0] + '.txt'
        tasks.append((in_csv, out_txt))

    # 使用线程池并行处理
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_args = {executor.submit(process_single_file, args): args for args in tasks}

        # 收集结果
        for future in as_completed(future_to_args):
            result = future.result()
            print(result)

    print("全部处理完成！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="递归处理 input_root 下所有 CSV，生成相同目录结构的 TXT（并行处理）。")
    parser.add_argument("--input_root", default=r'spike/video3', help="输入根目录，包含 0~10 子文件夹")
    parser.add_argument("--output_root", default=r'fpga/video3', help="输出根目录，会在此重建相同的目录结构")
    parser.add_argument("--workers", type=int, default=6, help="并行处理线程数，默认为4")

    args = parser.parse_args()

    main(args.input_root, args.output_root, args.workers)
    print("全部处理完成！")
