"""Case-grouped cross-validation helper."""
from collections import defaultdict

import numpy as np
from sklearn.model_selection import StratifiedKFold

from parse_slide_ids import norm_key

STATE = {'case': None}


def make_case_grouped_folds(records, n_folds, seed):
    """
    与 final_train.make_source_stratified_folds 的规则一致，
    只把切分单位从 slide 换成 **case（标本号）**：
      · 以 (source, slide_id) 聚合文件（orig + aug 同折，与原逻辑相同）
      · 再以 (source, case) 把切片聚合成组
      · 在**组**层面按 (source, label) 双层分层做 KFold
    """
    cmap = STATE['case'] or {}

    slide_map = {}
    for rec in records:
        key = (rec['source'], rec['slide_id'])
        slide_map.setdefault(key, {'label': rec['label'], 'source': rec['source'],
                                   'slide_id': rec['slide_id'], 'files': []})
        slide_map[key]['files'].append(rec)
    slides = list(slide_map.values())

    # 切片 → 组
    groups = {}
    for sl in slides:
        case = cmap.get(norm_key(sl['slide_id']), sl['slide_id'])
        gk = (sl['source'], case)
        groups.setdefault(gk, {'source': sl['source'], 'case': case,
                               'labels': set(), 'slides': []})
        groups[gk]['labels'].add(sl['label'])
        groups[gk]['slides'].append(sl)
    glist = list(groups.values())

    multi = [g for g in glist if len(g['labels']) > 1]
    if multi:
        print(f"  [warn] {len(multi)} 个标本组内标签不一致，按多数票分层：")
        for g in multi[:5]:
            print(f"         {g['source']}/{g['case']}  labels={sorted(g['labels'])}")
    for g in glist:
        g['label'] = int(round(np.mean([s['label'] for s in g['slides']])))

    n_slide, n_group = len(slides), len(glist)
    dup = [g for g in glist if len(g['slides']) > 1]
    print(f"\n  Data pool (case level):  {n_slide} slides → {n_group} cases"
          f"   (多切片标本 {len(dup)} 个)")
    for g in dup:
        print(f"    {g['source']}/{g['case']}  x{len(g['slides'])}: "
              + ', '.join(s['slide_id'][:34] for s in g['slides']))
    src_summary = defaultdict(lambda: [0, 0])
    for g in glist:
        src_summary[g['source']][g['label']] += 1
    for src, (neg, pos) in sorted(src_summary.items()):
        print(f"    {src:12s}  cases: pos={pos:4d}  neg={neg:4d}")

    by_src_lbl = defaultdict(list)
    for g in glist:
        by_src_lbl[(g['source'], g['label'])].append(g)

    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_groups = [[] for _ in range(n_folds)]
    for (src, lbl), gs in by_src_lbl.items():
        arr = np.array(gs, dtype=object)
        if len(arr) < n_folds:
            for i, g in enumerate(arr):
                fold_groups[i % n_folds].append(g)
        else:
            y = np.zeros(len(arr), dtype=int)
            for fi, (_, vi) in enumerate(kf.split(arr, y)):
                for g in arr[vi]:
                    fold_groups[fi].append(g)

    folds = []
    for k in range(n_folds):
        val_recs = [r for g in fold_groups[k] for s in g['slides'] for r in s['files']]
        tr_recs  = [r for j in range(n_folds) if j != k
                    for g in fold_groups[j] for s in g['slides'] for r in s['files']]
        folds.append((tr_recs, val_recs))

    print(f"\n  Per-fold split (cases / slides / files):")
    for k, (tr, va) in enumerate(folds):
        va_c = len(fold_groups[k])
        tr_c = sum(len(fold_groups[j]) for j in range(n_folds) if j != k)
        va_s = sum(len(g['slides']) for g in fold_groups[k])
        tr_s = sum(len(g['slides']) for j in range(n_folds) if j != k
                   for g in fold_groups[j])
        va_pos = sum(r['label'] for r in va)
        print(f"    Fold {k+1}  train {tr_c:3d} cases /{tr_s:4d} slides /{len(tr):4d} files"
              f"   val {va_c:3d} cases /{va_s:4d} slides /{len(va):4d} files"
              f"  (+{va_pos}/-{len(va)-va_pos})")

    # 泄漏自检
    for k, (tr, va) in enumerate(folds):
        tc = {cmap.get(norm_key(r['slide_id']), r['slide_id']) for r in tr}
        vc = {cmap.get(norm_key(r['slide_id']), r['slide_id']) for r in va}
        inter = tc & vc
        assert not inter, f"Fold {k+1} 标本跨折泄漏: {sorted(inter)[:5]}"
    print(f"  ✓ 自检通过：5 折之间无标本级重叠")
    return folds
