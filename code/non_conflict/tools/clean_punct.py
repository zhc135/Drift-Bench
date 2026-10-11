

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import json
import re
from typing import Any

# 可自定义的标点字符集合（包含中英文常见标点）
PUNCT_CHARS = "，,。.；;:：！？?!、…·•-—()（）\"'‘’“”…"

# 正则：匹配由上述标点任意字符组成且长度 >= 2 的连续序列（如 "，，，，"、",,,"、"，，。,"）
RE_REPEAT_PUNCT = re.compile(r"[{}]{{2,}}".format(re.escape(PUNCT_CHARS)))

# 仅合并除换行之外的多余空白（保留连续换行作为段落分隔）
RE_MULTISPACE = re.compile(r"[^\S\n]{2,}")  # 匹配连续两个及以上的空白，但不包含换行符

input_path="data/non_conflict/all_rewritten/summary_finance.json"

def clean_string(s: str) -> str:
    """
    清理字符串：
      - 删除连续两个及以上的标点序列（默认行为）
      - 合并多余的空格/制表符，但保留换行和段落分隔（\n\n）
      - 去除字符串前后多余的空白和标点（但不去掉前后换行）
    若想把连续标点合并为单个标点而不是删除，见注释替代方案。
    """
    if not s:
        return s

    # 统一换行为 '\n'（防止 Windows/mac 不同换行干扰）
    s = s.replace("\r\n", "\n").replace("\r", "\n")

    # 1) 删除连续的标点序列（长度 >= 2）
    s = RE_REPEAT_PUNCT.sub("", s)

    # 若你想保留一个标点（把连续多重标点合并为单个相同标点），
    # 可以使用下面这段替代上面一行：
    # def _collapse(match):
    #     seq = match.group(0)
    #     # 以序列中的最后一个字符作为代表（也可以选第一个）
    #     return seq[-1]
    # s = RE_REPEAT_PUNCT.sub(_collapse, s)

    # 2) 合并多余空白为单个空格，但保留换行（不把 \n\n 压成一个空格）
    s = RE_MULTISPACE.sub(" ", s)

    # 3) 去除开头或结尾多余的标点或空白（注意：不去除换行符，以保留段落分隔）
    s = s.strip(" \t" + PUNCT_CHARS)

    return s

def clean_json(obj: Any) -> Any:
    """
    递归遍历 JSON 对象，清理所有字符串值。
    """
    if isinstance(obj, dict):
        return {k: clean_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_json(v) for v in obj]
    elif isinstance(obj, str):
        return clean_string(obj)
    else:
        return obj

def main(input_path: str, output_path: str):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    cleaned = clean_json(data)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=2)

    print(f"Cleaned JSON written to: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python clean_punct.py input.json output.json")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])

