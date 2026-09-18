#!/usr/bin/env python3
"""组装 AIR-Bench Chat_result:eval 输出 + Chat_meta 回填 meta_info/task_name/dataset_name。
用法: python3 assemble.py <eval_resp.jsonl> <Chat_meta.json> <out Chat_result.jsonl>
"""
import json
import sys

resp_path, meta_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
meta = {str(r['uniq_id']): r for r in json.load(open(meta_path, encoding='utf-8'))}
n = 0
with open(out_path, 'w', encoding='utf-8') as out:
    for line in open(resp_path, encoding='utf-8'):
        r = json.loads(line)
        m = meta.get(str(r['uniq_id']))
        if m is None:
            print(f"[skip] uniq_id {r['uniq_id']} 不在 meta 中")
            continue
        out.write(json.dumps({
            'meta_info': json.dumps(m['meta_info'], ensure_ascii=False),
            'question': r['question'], 'answer_gt': r['answer_gt'],
            'path': r['path'], 'task_name': m['task_name'],
            'dataset_name': m['dataset_name'], 'response': r['response'],
            'uniq_id': r['uniq_id'],
        }, ensure_ascii=False) + '\n')
        n += 1
print(f"[assemble] {n} 条 -> {out_path}")
