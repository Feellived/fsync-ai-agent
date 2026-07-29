"""
step3 — 괴리도(Gap) 산출 (발명 명세서 5.(7), 수식 ④, 청구항 1·2·3)

라벨별 상대 강도(lift)를 산출하고, 의도 라벨을 기준으로 라벨 단위 판정을 내린다.

    ④ lift(l) = p_target(l) / p_baseline(l)

    lift > 1 → 해당 반응이 통상 수준보다 강하게 발생
    lift < 1 → 통상 수준에 미달

판정 (청구항 3):
    의도 라벨   & lift ≥ TAU_ALIGNED   → ✅ 정렬 : 의도한 반응이 전달됨
    의도 라벨   & lift <  TAU_ALIGNED  → 🔴 누수 : 의도했으나 전달되지 못함
    비의도 라벨 & lift ≥ TAU_EMERGENT  → 🟡 자생 : 의도하지 않았으나 강하게 발생
    비의도 라벨 & lift ≥ TAU_LOW       →    보통
    비의도 라벨 & lift <  TAU_LOW      →    낮음

※ 청구항 3은 단일 '기준값'으로 기술되어 있으나, 실제 운용에서는 의도 라벨의 전달
   판정 기준(TAU_ALIGNED)보다 비의도 라벨의 자생 판정 기준(TAU_EMERGENT)을 높게 둔다.
   의도하지 않은 반응을 '자생'이라 부르려면 통상 대비 더 뚜렷한 초과가 필요하기 때문.
   둘 다 CLI로 조정 가능하며, 값을 바꿔도 판정만 달라지고 lift 자체는 불변이다.

이전 버전은 의도 강도를 float으로 환산해 코사인 유사도 하나로 '전체 일치율'을 뽑고
반응−의도 차분을 gap으로 삼았으나, 명세서 어디에도 그런 산출식은 없다. 단일 스칼라
일치율은 라벨별로 방향이 반대인 괴리(누수와 자생)를 상쇄해 버려 해석을 오히려 망친다.
그 경로는 폐기하고 명세서대로 라벨 단위 lift 판정으로 대체했다.

사용 예:
  python gap_analysis/step3_compute_gap.py --release hwanjeolgi \
      --baseline "05_최종결과/corpus_reaction_08_9label.json" \
      --target   "05_최종결과/hwanjeolgi_reaction_080_9label.json"
"""
import argparse
import json
import math
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "out")

# 판정 기준값 (기본값은 최종 발표에 쓰인 설정)
TAU_ALIGNED = 1.2    # 의도 라벨: 이 이상이면 전달된 것으로 판정
TAU_EMERGENT = 1.5   # 비의도 라벨: 이 이상이면 자생 반응으로 판정
TAU_LOW = 0.8        # 비의도 라벨: 이 미만이면 통상 미달

VERDICT_ALIGNED = "✅ 정렬"
VERDICT_LEAK = "🔴 누수"
VERDICT_EMERGENT = "🟡 자생"
VERDICT_NORMAL = "  보통"
VERDICT_LOW = "  낮음"


def judge(lift, intended, tau_aligned, tau_emergent, tau_low):
    if intended:
        return VERDICT_ALIGNED if lift >= tau_aligned else VERDICT_LEAK
    if lift >= tau_emergent:
        return VERDICT_EMERGENT
    if lift >= tau_low:
        return VERDICT_NORMAL
    return VERDICT_LOW


def compute_gap(release, baseline, target, intent,
                tau_aligned=TAU_ALIGNED, tau_emergent=TAU_EMERGENT, tau_low=TAU_LOW):
    """baseline/target 은 step2 산출물, intent 는 step1 산출물의 intent dict."""
    labels = baseline["labels"]
    if labels != target["labels"]:
        raise ValueError(
            "기준선과 대상의 라벨 집합이 다름 — 동일 라벨 체계에서만 대비 가능\n"
            f"  baseline: {labels}\n  target  : {target['labels']}"
        )
    if baseline["threshold"] != target["threshold"]:
        raise ValueError(
            f"임계값이 다름 (baseline τ={baseline['threshold']}, target τ={target['threshold']}) "
            "— 동일 임계값에서만 대비 가능"
        )

    b_pct = baseline["reaction_pct"]
    t_pct = target["reaction_pct"]

    out_labels = {}
    n_success = 0
    n_intended = 0
    for l in labels:
        b = b_pct.get(l, 0.0)
        t = t_pct.get(l, 0.0)
        if b == 0:
            raise ZeroDivisionError(f"기준선 p({l}) = 0 — lift 정의 불가. 기준선 코퍼스를 늘리세요")

        lift = t / b                                        # ④
        level = intent.get(l, "none")
        intended = level != "none"
        verdict = judge(lift, intended, tau_aligned, tau_emergent, tau_low)

        if intended:
            n_intended += 1
            if verdict == VERDICT_ALIGNED:
                n_success += 1

        out_labels[l] = {
            "baseline": round(b, 4),
            release: round(t, 4),
            "lift": round(lift, 3),
            "log2_lift": round(math.log2(lift), 3),
            "intent": level,
            "verdict": verdict,
        }

    if n_intended == 0:
        raise ValueError(f"[{release}] 의도 라벨이 하나도 없음 — step1 산출물을 확인하세요")

    return {
        "release": release,
        "method": f"sigmoid_{baseline['threshold']:.2f}_{len(labels)}label",
        "thresholds": {"aligned": tau_aligned, "emergent": tau_emergent, "low": tau_low},
        "intent": {l: v for l, v in intent.items() if v != "none"},
        # 의도한 라벨 중 전달에 실패한 비율. 라벨별 판정의 요약일 뿐이며,
        # 이 값 하나가 '정합도'를 대표하지는 않는다 (누수와 자생은 상쇄되지 않는다).
        "intent_miss_rate": round(1 - n_success / n_intended, 4),
        "n_success": n_success,
        "n_intended": n_intended,
        "labels": out_labels,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--release", required=True, help="intent_labels.json 의 릴리즈 키 (예: hwanjeolgi)")
    p.add_argument("--baseline", required=True, help="기준선 반응 분포 JSON (step2 산출물)")
    p.add_argument("--target", required=True, help="대상 반응 분포 JSON (step2 산출물)")
    p.add_argument("--intent", default=None,
                   help="intent JSON (기본: gap_analysis/out/{release}_intent.json)")
    p.add_argument("--out", default=None)
    p.add_argument("--tau-aligned", type=float, default=TAU_ALIGNED)
    p.add_argument("--tau-emergent", type=float, default=TAU_EMERGENT)
    p.add_argument("--tau-low", type=float, default=TAU_LOW)
    args = p.parse_args()

    intent_path = args.intent or os.path.join(OUT_DIR, f"{args.release}_intent.json")
    if not os.path.exists(intent_path):
        raise FileNotFoundError(
            f"{intent_path} 없음 — step1_build_intent_vectors.py 를 먼저 실행하세요")

    with open(args.baseline, encoding="utf-8") as f:
        baseline = json.load(f)
    with open(args.target, encoding="utf-8") as f:
        target = json.load(f)
    with open(intent_path, encoding="utf-8") as f:
        intent = json.load(f)["intent"]

    report = compute_gap(args.release, baseline, target, intent,
                         args.tau_aligned, args.tau_emergent, args.tau_low)

    out_path = args.out or os.path.join(OUT_DIR, f"{args.release}_gap_lift_{report['method']}.json")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    w = 58
    print(f"\n{'=' * w}\n  Gap Report — {args.release}  ({report['method']})\n{'=' * w}")
    print(f"  의도 라벨 {report['n_intended']}개 중 {report['n_success']}개 전달  "
          f"(미달률 {report['intent_miss_rate']:.1%})\n")
    print(f"  {'라벨':16s} {'기준선':>8} {'대상':>8} {'lift':>7}  판정")
    print(f"  {'-' * (w - 4)}")
    for l, v in sorted(report["labels"].items(), key=lambda x: -x[1]["lift"]):
        print(f"  {l:16s} {v['baseline']:>8.4f} {v[args.release]:>8.4f} "
              f"{v['lift']:>7.3f}  {v['verdict']}")
    print(f"\n✓ {out_path}")


if __name__ == "__main__":
    main()
