"""
step1 — 의도 라벨 추출 (발명 명세서 5.(6) 단계)

기획 문서에서 읽어낸 의도 라벨과 강조 강도를 반응 라벨 체계와 동일한 라벨 공간에
대응시켜 릴리즈별 intent JSON으로 떨어뜨린다.

입력 : gap_analysis/intent_labels.json  (라벨 + 강도만. 기획 문서 원문은 대외비라 미포함)
출력 : gap_analysis/out/{release}_intent.json

이전 버전은 의도 강도를 float 스코어(high=1.0, med=0.67 …)로 환산해 코사인 비교에
넣었으나, 명세서의 괴리도 판정(청구항 3)은 강도를 float으로 쓰지 않고 '의도 라벨인가
아닌가'의 기준으로만 사용한다. 따라서 등급 문자열을 그대로 보존한다.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "model_training_script"))
import config

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(BASE_DIR, "intent_labels.json")
OUT_DIR = os.path.join(BASE_DIR, "out")

VALID_LEVELS = {"high", "med", "low", "none"}


def build(release=None):
    with open(SRC, encoding="utf-8") as f:
        spec = json.load(f)

    releases = spec["releases"]
    targets = [release] if release else list(releases)

    os.makedirs(OUT_DIR, exist_ok=True)
    for rel in targets:
        if rel not in releases:
            raise KeyError(f"'{rel}' 릴리즈가 intent_labels.json 에 없음 (가능: {list(releases)})")
        entry = releases[rel]
        raw = entry["intent"]

        unknown = set(raw) - set(config.LABELS)
        if unknown:
            raise ValueError(f"[{rel}] 반응 라벨 체계에 없는 라벨: {sorted(unknown)}")
        bad = {k: v for k, v in raw.items() if v not in VALID_LEVELS}
        if bad:
            raise ValueError(f"[{rel}] 허용되지 않은 강도값: {bad}")

        # 명시되지 않은 라벨은 의도 없음(none)
        intent = {lbl: raw.get(lbl, "none") for lbl in config.LABELS}

        out = {
            "release": rel,
            "album_title": entry.get("album_title"),
            "release_date": entry.get("release_date"),
            "intent": intent,
            "intended_labels": [l for l in config.LABELS if intent[l] != "none"],
        }
        out_path = os.path.join(OUT_DIR, f"{rel}_intent.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)

        print(f"✓ {rel}_intent.json  (의도 라벨 {len(out['intended_labels'])}개)")
        for lbl in out["intended_labels"]:
            print(f"   {lbl:18s} {intent[lbl]}")
        print()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--release", default=None, help="미지정 시 전체 릴리즈")
    build(p.parse_args().release)
