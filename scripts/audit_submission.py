"""Audit downloaded evidence without pretending missing training metadata exists.

Run: python scripts/audit_submission.py (requires pyarrow and Pillow).
This supplements, never replaces, scripts/verify.py.
"""
import hashlib
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lab22 import data as D, judge as J


def main():
    train = pq.read_table(ROOT / 'data/pref/train.parquet').to_pylist()
    held = pq.read_table(ROOT / 'data/pref/eval.parquet').to_pylist()
    D.assert_disjoint(train, held)
    assert len(train) == 800 and len(held) == 100
    path = ROOT / 'data/eval/side_by_side.jsonl'
    outputs = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
    summary = json.loads((ROOT / 'data/eval/judge_summary.json').read_text(encoding='utf-8'))
    judged = json.loads((ROOT / 'data/eval/judge_results_rm.json').read_text(encoding='utf-8'))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == summary['outputs_sha256'] == judged['outputs_sha256']
    assert len(outputs) == len(judged['records']) == 58
    eval_prompts = {D.normalize_prompt(r['prompt'][0]['content']) for r in held}
    eval_outputs = [r for r in outputs if r['category'] == 'heldout']
    assert len({D.normalize_prompt(r['prompt']) for r in eval_outputs}) == 50
    assert all(D.normalize_prompt(r['prompt']) in eval_prompts for r in eval_outputs)
    for original, verdict in zip(outputs, judged['records']):
        assert all(original[k] == verdict[k] for k in ('id', 'prompt', 'category', 'sft', 'dpo'))
    names = list(judged['per_judge'])
    assert len(names) == 2
    for i, row in enumerate(judged['records']):
        expected = J.panel_record([judged['per_judge'][name][i] for name in names])
        assert expected['winner'] == row['winner']
    for category in ['overall', 'heldout', 'helpfulness', 'safety']:
        rows = judged['records'] if category == 'overall' else [r for r in judged['records'] if r['category'] == category]
        recalculated = J.summarize(rows, seed=42)
        for key in ('n', 'dpo_wins', 'sft_wins', 'ties', 'dpo_win_rate', 'win_rate_ci95', 'length_matched_n', 'length_matched_win_rate', 'mean_chars_sft', 'mean_chars_dpo'):
            assert recalculated[key] == summary[category][key], (category, key)
    assert J.agreement(*[judged['per_judge'][name] for name in names])['agreement'] == summary['judge_agreement']['agreement']
    for name in ['02-sft-loss', '02b-pref-length', '03-dpo-reward-curves', '04-side-by-side-table']:
        with Image.open(ROOT / f'submission/screenshots/{name}.png') as image:
            image.verify()
    print('PASS evidence: 800/100 disjoint; 50 distinct evaluated held-out prompts; output SHA256 matches')
    print('PASS judge: 58 original outputs preserved; panel votes/counts/rates/length statistics/agreement match')
    print('PASS images: all four mandatory images decode successfully')
    assert D.split_mismatch(ROOT / 'data/pref', ROOT / 'adapters/dpo') is None
    assert (ROOT / 'adapters/sft-mini/adapter_config.json').exists()
    assert (ROOT / 'models/sft-merged/config.json').exists()
    print('PASS supplied split fingerprint and SFT metadata presence')
    print('NOT VERIFIED: full clean GPU rerun; use scripts/verify.py for remaining gaps')


if __name__ == '__main__':
    main()
