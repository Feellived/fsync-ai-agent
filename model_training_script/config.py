from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

LABELS = [
    "기타_노이즈",
    "비주얼_멤버매력",
    "장기_팬덤",
    "역주행_기대",
    "보컬_라이브",
    "신규_유입",
    "밴드_정체성",
    "위로_공감",
    "연주_악기",
    "이별_감성",
    "청량_여름",
    "음악성",
]
NUM_LABELS = len(LABELS)

# 최종 채택 백본 (명세서 5.(3) 실시예). 백본 비교 실험 결과는 model_output/*/eval_results.json 참고:
#   me5_large 0.762 > roberta_large 0.739 > deberta_large 0.721 > modernbert_large 0.697 (micro F1)
MODEL_NAME = "intfloat/multilingual-e5-large"
MAX_LEN = 128
BATCH_SIZE = 64
EPOCHS = 5
LR = 2e-5
WARMUP_RATIO = 0.1
# 학습/검증 리포팅용 임계값. 반응 분포 산출의 추론 임계값 τ(=0.80)와는 별개다 — τ는
# gap_analysis/step2_build_reaction_vector.py 의 --tau 로 지정한다.
THRESHOLD = 0.5

LABELED_PATH = str(PROJECT_ROOT / "01_corpus" / "corpus_labeled_1.jsonl")
TRAIN_PATH = str(BASE_DIR / "train.jsonl")
VAL_PATH = str(BASE_DIR / "val.jsonl")
SAVE_PATH = str(PROJECT_ROOT / "model_output")
