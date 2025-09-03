# text_explainer.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
텍스트 단일 입력(예: '요약 문장')에 대해:
  - 분류 posterior / pred label
  - Attention 기반 saliency
  - Integrated Gradients(가능하면 Captum 이용)
  - LIME(필수) 기반 토큰/단어 기여도
를 산출하고, 그 결과(JSON payload)를 GPT에 전달하여
  - 비전문가용(쉬운 설명)
  - 전문가용(기술적 요약)
두 종류 한국어/영어/프랑스어/일본어/스페인어/러시아어 리포트를 생성합니다.

의존성:
  pip install transformers torch captum lime openai python-dotenv
"""

from __future__ import annotations
import os
import json
from typing import Any, Dict, List, Optional, Tuple

import torch
import numpy as np

# --- Optional: Captum (IG). 없으면 IG 비활성 처리 ---
try:
    from captum.attr import IntegratedGradients
    _CAPTUM_OK = True
except Exception:
    _CAPTUM_OK = False

# --- Required: LIME (필수) ---
try:
    from lime.lime_text import LimeTextExplainer
    _LIME_OK = True
except Exception:
    _LIME_OK = False

# --- OpenAI (Responses API) ---
from dotenv import load_dotenv

load_dotenv()
try:
    from openai import AsyncOpenAI
except Exception as e:
    raise RuntimeError("openai 패키지가 필요합니다. 설치: pip install openai") from e


DEFAULT_GPT_MODEL = os.environ.get("GPT_EXPLAIN_MODEL", "gpt-4.1-mini")


# =====================================================================================
# Core Engine: 예측 + 어트리뷰션(Attention / IG / LIME)
# =====================================================================================
class TextAttributionEngine:
    """
    hf_model / hf_tokenizer / id2label 을 받아
      - predict(): posterior/label/tokens/attentions
      - attention_saliency(): 마지막 레이어 평균헤드 attention으로 saliency
      - integrated_gradients(): Captum IG (가능 시)
      - lime(): LIME 텍스트 설명 (필수)
      - run_all(): 위 항목을 모두 실행 후 payload(JSON) 반환
    를 제공합니다.
    """

    def __init__(
        self,
        hf_model,
        hf_tokenizer,
        id2label: Dict[int, str],
        device: Optional[str] = None,
    ):
        self.model = hf_model
        self.tok = hf_tokenizer
        self.id2label = id2label
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()

    # -------------------- 기본 예측 --------------------
    @torch.inference_mode()
    def predict(self, text: str, max_length: int = 256) -> Dict[str, Any]:
        """
        return:
          - probs, pred_idx, pred_label
          - tokens (WordPiece 기준)
          - attentions (tuple of last-layer attention if enabled)
        """
        enc = self.tok(text, return_tensors="pt", truncation=True, max_length=max_length)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        out = self.model(**enc, output_attentions=True, return_dict=True)

        logits = out.logits  # (1, C)
        probs = torch.softmax(logits, dim=-1)[0].detach().cpu().numpy().tolist()
        pred_idx = int(np.argmax(probs))
        pred_label = self.id2label.get(pred_idx, str(pred_idx))

        tokens = self.tok.convert_ids_to_tokens(enc["input_ids"][0].detach().cpu().tolist())

        return {
            "probs": probs,
            "pred_idx": pred_idx,
            "pred_label": pred_label,
            "tokens": tokens,
            "attentions": out.attentions,  # tuple(layer) of (B, H, T, T)
            "enc": enc,
        }

    # -------------------- Attention saliency --------------------
    def attention_saliency(self, attns, focus: str = "cls_to_token") -> List[float]:
        """
        마지막 레이어의 헤드 평균 attention으로 토큰 saliency를 계산.
        focus:
          - 'cls_to_token': CLS(0) → 각 토큰 주의도 (일반적)
          - 그 외: 열 평균
        """
        if not attns:
            return []
        last = attns[-1]              # (B, H, T, T)
        att = last.mean(dim=1)        # (B, T, T)
        A = att[0].detach().cpu().numpy()
        if focus == "cls_to_token":
            sal = A[0]                # CLS → 각 토큰
        else:
            sal = A.mean(axis=0)
        sal = np.maximum(sal, 0)
        if sal.sum() > 0:
            sal = sal / (sal.sum() + 1e-8)
        return sal.tolist()

    # -------------------- Integrated Gradients --------------------
    def integrated_gradients(
        self,
        text: str,
        target_idx: int,
        steps: int = 50,
        max_length: int = 256,
    ) -> Dict[str, Any]:
        """
        Captum IG. Captum 미설치 시 available=False로 반환.
        """
        if not _CAPTUM_OK:
            return {"available": False, "token_scores": [], "tokens": []}

        self.model.zero_grad()
        enc = self.tok(text, return_tensors="pt", truncation=True, max_length=max_length)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]

        # 임베딩 층
        emb_layer = self.model.get_input_embeddings()

        def forward_fn(embeddings):
            outputs = self.model(
                inputs_embeds=embeddings, attention_mask=attention_mask, return_dict=True
            )
            logits = outputs.logits  # (1, C)
            return logits[:, target_idx]

        baseline = torch.zeros_like(emb_layer(input_ids))
        inputs = emb_layer(input_ids).requires_grad_(True)

        ig = IntegratedGradients(forward_fn)
        attributions, _ = ig.attribute(
            inputs, baselines=baseline, n_steps=steps, return_convergence_delta=True
        )
        token_scores = attributions.abs().sum(dim=-1)[0].detach().cpu().numpy()
        token_scores = token_scores / (token_scores.sum() + 1e-8)

        tokens = self.tok.convert_ids_to_tokens(input_ids[0].detach().cpu().tolist())
        return {"available": True, "token_scores": token_scores.tolist(), "tokens": tokens}

    # -------------------- LIME (필수) --------------------
    def lime(
        self,
        text: str,
        class_names: List[str],
        target_idx: int,
        num_features: int = 10,
        max_length: int = 256,
    ) -> Dict[str, Any]:
        """
        LIME 텍스트 설명. 설치 필수(_LIME_OK=False면 예외).
        weights: List[(token, weight)]
        """
        if not _LIME_OK:
            raise RuntimeError("LIME이 설치되어 있지 않습니다. `pip install lime` 후 다시 시도하세요.")

        def predict_proba(texts: List[str]) -> np.ndarray:
            outs = []
            with torch.inference_mode():
                for t in texts:
                    enc = self.tok(t, return_tensors="pt", truncation=True, max_length=max_length)
                    enc = {k: v.to(self.device) for k, v in enc.items()}
                    logits = self.model(**enc).logits
                    probs = torch.softmax(logits, dim=-1)[0].detach().cpu().numpy()
                    outs.append(probs)
            return np.vstack(outs)

        explainer = LimeTextExplainer(class_names=class_names)
        exp = explainer.explain_instance(
            text, predict_proba, num_features=num_features, labels=[target_idx]
        )
        weights = exp.as_list(label=target_idx)  # [(token, weight), ...]
        return {"available": True, "weights": weights}

    # -------------------- 전체 실행 --------------------
    def run_all(self, text: str) -> Dict[str, Any]:
        """
        Attention + (가능 시)IG + (필수)LIME 수행 후 payload 반환.
        """
        if not _LIME_OK:
            # LIME 필수 정책
            raise RuntimeError("LIME이 설치되어 있지 않습니다. `pip install lime` 후 다시 시도하세요.")

        pred = self.predict(text)
        tokens = pred["tokens"]

        # Attention saliency
        att_sal = self.attention_saliency(pred["attentions"], focus="cls_to_token")

        # IG (옵션)
        ig_res = self.integrated_gradients(text, target_idx=pred["pred_idx"])
        ig_scores = ig_res["token_scores"] if ig_res.get("available") else [0.0] * len(tokens)

        # LIME (필수)
        class_names = [self.id2label[i] for i in sorted(self.id2label)]
        lime_res = self.lime(text, class_names=class_names, target_idx=pred["pred_idx"], num_features=10)

        token_scores = {
            "tokens": tokens,
            "attention": att_sal if att_sal else [0.0] * len(tokens),
            "integrated_gradients": ig_scores,
        }

        payload = {
            "text": text,
            "pred_idx": pred["pred_idx"],
            "pred_label": pred["pred_label"],
            "posterior": pred["probs"][pred["pred_idx"]],
            "probs": pred["probs"],
            "token_scores": token_scores,
            "lime": lime_res,  # 필수 포함
            "notes": {
                "captum_available": _CAPTUM_OK,
                "lime_available": _LIME_OK,
                "explanations": "attention_saliency, integrated_gradients(if_available), lime(required)"
            },
        }
        return payload


# =====================================================================================
# Reporter: GPT로 리포트(비전문가/전문가) 생성
# =====================================================================================
class GPTReporter:
    """
    payload(JSON dict) → 비전문가용 / 전문가용 보고서 생성
    언어 코드: ko, en, fr, ja, es, ru
    """
    _LANG_MAP = {
        "ko": "Korean",
        "en": "English",
        "fr": "French",
        "ja": "Japanese",
        "es": "Spanish",
        "ru": "Russian",
    }

    def __init__(
        self,
        model: str = DEFAULT_GPT_MODEL,
        temperature: float = 0.4,
        max_tokens: int = 700,
    ):
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY 환경변수가 설정되어 있지 않습니다.")
        self.client = AsyncOpenAI()
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def _ask(self, payload: Dict[str, Any], audience: str, lang_code: str) -> str:
        """
        audience: 'layperson' | 'expert'
        lang_code: 'ko'|'en'|'fr'|'ja'|'es'|'ru'
        """
        lang_name = self._LANG_MAP.get(lang_code, "Korean")
        sys_msg = "You convert model interpretation payloads into clear, accurate reports."

        if audience == "layperson":
            style = (f"Write in {lang_name} for non-experts. Use simple, friendly language. "
        "Clearly state whether the model judged the summarized text as likable or unlikable. "
        "Mention how confident the model is (posterior probability). "
        "Then highlight a few key words or phrases that most influenced this judgment "
        "(from attention, Integrated Gradients, and LIME), but avoid technical jargon. "
        "Explain them in terms of why they might feel positive or negative. "
        "Use short sentences or a bullet list. "
        "Conclude with a gentle metaphorical reminder, such as: "
        "‘Think of this as a sketch by AI — a rough mirror image, not the final portrait.’")
        else:
            style = (
                f"Write in {lang_name} for technical experts. Start with predicted label and posterior. "
                "Summarize token-level contributions from attention and Integrated Gradients; include LIME top features with signs. "
                "Briefly discuss assumptions/limitations: tokenization granularity, baseline choice for IG, perturbation instability in LIME, "
                "and that saliency is correlational, not causal. Keep it compact but precise."
            )

        user = (
            f"PAYLOAD(JSON):\n{json.dumps(payload, ensure_ascii=False)}\n\n"
            f"Task: Produce a {audience} report. {style}"
        )

        resp = await self.client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": [{"type": "input_text", "text": sys_msg}]},
                {"role": "user", "content": [{"type": "input_text", "text": user}]},
            ],
            max_output_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return (getattr(resp, "output_text", "") or "").strip()

    async def make_reports(self, payload: Dict[str, Any], lang_code: str = "ko") -> Tuple[str, str]:
        """
        return: (layperson_report, expert_report)
        """
        lay = await self._ask(payload, audience="layperson", lang_code=lang_code)
        exp = await self._ask(payload, audience="expert", lang_code=lang_code)
        return lay, exp