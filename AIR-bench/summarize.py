#!/usr/bin/env python3
"""AIR-Bench 官方口径汇总(与 cal_score.py 同逻辑):WinRate = llm_score>gpt_score 比例。
用法: python3 summarize.py <batch_run_output_xxx.jsonl>
"""
import json
import re
import sys

def parse_scores(gen_text):
    masked = re.sub(r'Assistant\s*\d+', 'Assistant', gen_text, flags=re.I)
    nums = [int(n) for n in re.findall(r'\d+', masked) if 0 <= int(n) <= 10]
    if len(nums) >= 2:
        return nums[-2], nums[-1]
    return None, None

input_file = sys.argv[1]
total, fail, win = 0, 0, 0
sum_g = sum_l = 0.0
for line in open(input_file, encoding='utf-8'):
    rec = json.loads(line)
    g, l = parse_scores(rec.get('gen', ''))
    if g is None or l is None:
        fail += 1
        print(f"[unparsed] {rec.get('gen', '')[:80]!r}")
        continue
    if g > 10 or l > 10:
        fail += 1
        continue
    total += 1
    if l > g:
        win += 1
    sum_g += g
    sum_l += l
print(f"input: {input_file}")
print(f"Sum={total}, Win_Rate={win/total if total else 0:.4f}, "
      f"gpt4_avg_score={sum_g/total if total else 0:.4f}, "
      f"llm_avg_score={sum_l/total if total else 0:.4f}")
print(f"fail_num: {fail}, success_num: {total}, "
      f"percentage: {total/(total+fail) if (total+fail) else 0:.4f}")
