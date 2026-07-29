"""
step2 — 반응 분포 산출 (발명 명세서 5.(4) 기준선 / 5.(5) 대상, 수식 ①②③)

이미 추론이 끝난 `*_me5_sigmoid.jsonl`(레코드마다 sigmoid 확률 dict 보유)을 받아
라벨별 반응 분포 p(l)을 산출한다. 기준선(baseline)과 대상(target)은 같은 함수·같은
임계값·같은 정규화로 처리되므로 그대로 대비 가능하다.

    ① hit(c,l) = 1[ σ(z_c,l) ≥ τ ]        (기본 τ = 0.80)
    ② H(l) = Σ_c hit(c,l),  N = Σ_{l∈L} H(l)
    ③ p(l) = H(l) / N

주의 — 분모는 '댓글 수'가 아니라 '분석 대상 라벨의 전체 히트 수 N'이다. 멀티라벨이라
한 댓글이 여러 라벨에 동시에 히트할 수 있어 Σ p(l) = 1 이 되도록 하는 정규화다.

사용 예:
  # 기준선 (8팀 전체 코퍼스)
  python gap_analysis/step2_build_reaction_vector.py \
      --sigmoid 04_reaction/전체/OUTPUT/corpus_me5_sigmoid.jsonl \
      --out gap_analysis/out/corpus_reaction_080_9label.json

  # 대상 (환절기) + 일별 추이
  python gap_analysis/step2_build_reaction_vector.py \
      --sigmoid 04_reaction/환절기/output/hwanjeolgi_me5_sigmoid.jsonl \
      --out gap_analysis/out/hwanjeolgi_reaction_080_9label.json --by-date
"""
import argparse
import json
import os
from collections import Counter, defaultdict

# 분석 대상 라벨 집합 L — 노이즈 라벨과 분석 제외 라벨을 뺀 9라벨 (명세서 실시예)
LABELS_9 = ["장기_팬덤", "보컬_라이브", "신규_유입", "밴드_정체성", "위로_공감",
            "연주_악기", "이별_감성", "청량_여름", "음악성"]
# 초기 실험용 대안 집합 (최종 결과는 9label 기준)
LABELS_8 = ["비주얼_멤버매력", "보컬_라이브", "밴드_정체성", "위로_공감",
            "연주_악기", "이별_감성", "청량_여름", "음악성"]
LABELS_7 = [l for l in LABELS_8 if l != "비주얼_멤버매력"]
LABEL_SETS = {"9label": LABELS_9, "8label": LABELS_8, "7label": LABELS_7}


def load_sigmoid(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if not rows:
        raise ValueError(f"{path} 에서 읽은 레코드가 0개")
    if "sigmoid" not in rows[0]:
        raise KeyError(f"{path} 레코드에 'sigmoid' 필드가 없음 — run_*_sigmoid.py 산출물이 맞는지 확인")
    return rows


def distribution(rows, labels, tau):
    """수식 ①②③ — 히트 판정 → 히트 수 집계 → 전체 히트 수로 정규화."""
    hits, n_hit, n_multi = [], 0, 0
    for r in rows:
        sig = r.get("sigmoid", {})
        matched = [l for l in labels if sig.get(l, 0.0) >= tau]   # ① ≥ τ
        if matched:
            hits.extend(matched)
            n_hit += 1
            if len(matched) >= 2:
                n_multi += 1

    counts = Counter(hits)
    total = len(hits)                                            # ② N
    if total == 0:
        raise ValueError(f"임계값 {tau} 에서 히트가 하나도 없음 — τ를 낮추세요")
    pct = {l: round(counts.get(l, 0) / total, 4) for l in labels}  # ③ p(l)

    return {
        "method": f"sigmoid_threshold_{tau}",
        "threshold": tau,
        "n_total": len(rows),
        "n_comments_hit": n_hit,
        "n_multi_hit": n_multi,
        "n_label_hits": total,
        "labels": labels,
        "reaction_pct": pct,
        "reaction_pct_percent": {l: round(v * 100, 2) for l, v in pct.items()},
        "hit_counts": {l: counts.get(l, 0) for l in labels},
    }


def by_date(rows, labels, tau):
    """일별 라벨 히트 수 — 명세서 5.(7)의 '게시 시점 메타데이터를 이용한 일별 반응 추이'."""
    daily = defaultdict(lambda: {l: 0 for l in labels})
    skipped = 0
    for r in rows:
        stamp = r.get("published_at") or r.get("date")
        if not stamp:
            skipped += 1
            continue
        day = str(stamp)[:10]
        sig = r.get("sigmoid", {})
        for l in labels:
            if sig.get(l, 0.0) >= tau:
                daily[day][l] += 1
    if skipped:
        print(f"  ! published_at 없는 레코드 {skipped:,}개는 일별 추이에서 제외")
    return {"threshold": tau, "labels": labels, "by_date": dict(sorted(daily.items()))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sigmoid", required=True, help="*_me5_sigmoid.jsonl 경로")
    p.add_argument("--out", required=True, help="출력 JSON 경로")
    p.add_argument("--tau", type=float, default=0.80, help="확률 임계값 τ (기본 0.80)")
    p.add_argument("--label-set", choices=sorted(LABEL_SETS), default="9label")
    p.add_argument("--by-date", action="store_true", help="일별 추이 JSON도 함께 저장")
    args = p.parse_args()

    labels = LABEL_SETS[args.label_set]
    rows = load_sigmoid(args.sigmoid)
    print(f"{os.path.basename(args.sigmoid)} — {len(rows):,}개 레코드, τ={args.tau}, {args.label_set}")

    result = distribution(rows, labels, args.tau)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"✓ {args.out}")
    print(f"  히트 댓글 {result['n_comments_hit']:,} / 전체 히트 N={result['n_label_hits']:,}")
    for l, v in sorted(result["reaction_pct_percent"].items(), key=lambda x: -x[1]):
        print(f"    {l:18s} {v:5.2f}%  {'█' * int(v / 2)}")

    if args.by_date:
        d_path = args.out.replace(".json", "_by_date.json")
        with open(d_path, "w", encoding="utf-8") as f:
            json.dump(by_date(rows, labels, args.tau), f, ensure_ascii=False, indent=2)
        print(f"✓ {d_path}")


if __name__ == "__main__":
    main()
