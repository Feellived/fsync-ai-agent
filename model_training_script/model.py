import torch.nn as nn
from transformers import AutoModel


class MultiLabelClassifier(nn.Module):
    """다국어 임베딩 인코더 + 라벨 수 차원 분류 헤드 (명세서 5.(3)).

    최종 백본은 intfloat/multilingual-e5-large 이며 RoBERTa 계열이 아니다.
    로짓만 반환하고 sigmoid/BCEWithLogitsLoss 는 호출부에서 적용한다.
    """

    def __init__(self, model_name, num_labels):
        super().__init__()
        try:
            self.encoder = AutoModel.from_pretrained(model_name, attn_implementation="sdpa")
        except (ValueError, NotImplementedError):
            self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, num_labels)

    # 기존 체크포인트(state_dict 키가 roberta.* 로 저장됨)와의 호환용 별칭
    @property
    def roberta(self):
        return self.encoder

    def load_state_dict(self, state_dict, strict=True, **kw):
        if any(k.startswith("roberta.") for k in state_dict):
            state_dict = {k.replace("roberta.", "encoder.", 1) if k.startswith("roberta.") else k: v
                          for k, v in state_dict.items()}
        return super().load_state_dict(state_dict, strict=strict, **kw)

    def forward(self, input_ids, attention_mask):
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = output.last_hidden_state[:, 0, :].to(self.classifier.weight.dtype)
        logits = self.classifier(self.dropout(cls))
        return logits  # shape: [batch, num_labels], raw logits


# 기존 import 경로 호환
RoBERTaMultiLabel = MultiLabelClassifier
