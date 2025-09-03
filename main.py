# streamlit_app.py
# 실행: streamlit run streamlit_app.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import io, os, time
from pathlib import Path
from typing import List, Dict, Any
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="캡션→요약→호감도 분류+해석 리포트", layout="wide")
st.title("📸 캡션 6개 → 📝 요약 1개 → 🧠 호감도 분류 + 🔍 해석 리포트")
st.caption("이미지/동영상 1개 업로드 → 6캡션 생성 → 1요약 생성 → 요약만 DeBERTa 분류 → Attention/IG/LIME 해석 → GPT 리포트(비전문가/전문가, 다국어)")

# ─────────────────────────────────────────────────────────────
# 사이드바 옵션
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.subheader("① 캡션 옵션")
    pos_text    = st.text_input("프레임 비율(쉼표, 0~1)", "0.10,0.20,0.45,0.55,0.80,0.90")
    caption_model  = st.text_input("캡션 모델", os.environ.get("GPT_CAPTION_MODEL", "gpt-4.1-mini"))
    base_temp   = st.slider("temperature", 0.0, 1.5, 0.7, 0.05)
    max_tokens  = st.number_input("max tokens", min_value=32, max_value=512, value=120, step=8)
    concurrency = st.slider("동시 처리", 1, 16, 6, 1)

    st.subheader("② 분류(DeBERTa) 모델")
    ckpt_dir = st.text_input("로컬 체크포인트 경로", r"C:\Users\DC\2025_kds\MODULES\deberta_prompt_model_V4_test3")
    cls_bs   = st.slider("분류 배치 크기", 1, 32, 8, 1)

    st.subheader("③ 리포트 언어")
    lang = st.selectbox(
        "리포트 언어 선택",
        ["ko","en","fr","ja","es","ru"],
        index=0,
        format_func=lambda x: {
            "ko":"한국어","en":"English","fr":"Français",
            "ja":"日本語","es":"Español","ru":"Русский"
        }[x]
    )

# ─────────────────────────────────────────────────────────────
# 업로더 (단일 파일)
# ─────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "이미지(.jpg/.png/...) 또는 동영상(.mp4 등) 1개 업로드",
    type=["jpg","jpeg","png","webp","bmp","mp4","avi","mov","mkv","wmv","m4v"],
    accept_multiple_files=False
)

def parse_positions(text: str) -> List[float]:
    try:
        vals = [float(x.strip()) for x in text.split(",") if x.strip()]
        vals = [min(max(v, 0.0), 1.0) for v in vals]
        return vals or [0.10,0.20,0.45,0.55,0.80,0.90]
    except:
        return [0.10,0.20,0.45,0.55,0.80,0.90]

# ─────────────────────────────────────────────────────────────
# 캐시된 캡셔너 (버튼 이후 임포트 → 초기 로딩 가볍게)
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_captioner(positions, model, base_temp, max_tokens, concurrency):
    from MODULES.image import MediaCaptioner  # lazy import
    return MediaCaptioner(
        positions=positions,
        model=model,
        base_temperature=float(base_temp),
        max_tokens=int(max_tokens),
        concurrency=int(concurrency),
        retries=5,
    )

# ─────────────────────────────────────────────────────────────
# 캐시된 간단 분류기 (DeBERTa) - 요약 1개만 분류
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_deberta_classifier(ckpt_dir: str):
    import torch
    from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification

    class _Cls:
        def __init__(self, model_dir: str):
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.fp16 = (self.device == "cuda")
            self.config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
            self.tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True, local_files_only=True)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True).to(self.device).eval()
            # 라벨: labels.txt 우선
            labels = None
            labels_path = os.path.join(model_dir, "labels.txt")
            if os.path.exists(labels_path):
                with open(labels_path, "r", encoding="utf-8") as f:
                    labels = [ln.strip() for ln in f if ln.strip()]
            if labels:
                self.id2label = {i: lab for i, lab in enumerate(labels)}
            elif getattr(self.config, "id2label", None):
                self.id2label = {int(k): v for k, v in self.config.id2label.items()}
            else:
                # 기본(예시)
                self.id2label = {0: "Unlikable", 1: "Likable"}

        @torch.inference_mode()
        def predict_one(self, text: str, max_len: int = 256) -> Dict[str, Any]:
            enc = self.tok(text, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
            enc = {k: v.to(self.device) for k, v in enc.items()}
            if self.fp16:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = self.model(**enc).logits
            else:
                logits = self.model(**enc).logits
            probs = logits.softmax(dim=-1)[0].detach().float().cpu().tolist()
            pred_idx = int(torch.tensor(probs).argmax().item())
            return {
                "pred_idx": pred_idx,
                "pred_label": self.id2label.get(pred_idx, str(pred_idx)),
                "probs": probs
            }

    return _Cls(ckpt_dir)

def is_video(name: str) -> bool:
    from MODULES.image import VIDEO_EXTS  # lazy
    return Path(name).suffix.lower() in VIDEO_EXTS

# ─────────────────────────────────────────────────────────────
# 메인 버튼: 캡션 6 → 요약 1 → 분류(요약만) → 해석(Attention/IG/LIME) → GPT 리포트(탭)
# ─────────────────────────────────────────────────────────────
if uploaded is not None:
    file_name  = uploaded.name
    file_bytes = uploaded.read()
    st.write(f"**파일명:** {file_name} | **크기:** {len(file_bytes):,} bytes")

    if not os.environ.get("OPENAI_API_KEY"):
        st.error("OPENAI_API_KEY 환경변수가 설정되어 있지 않습니다. .env 또는 환경변수로 설정하세요.")
        st.stop()

    if st.button("🚀 캡션 6개 생성 → 요약 → (요약만) 분류 + 해석 리포트"):
        t0 = time.perf_counter()
        positions = parse_positions(pos_text)

        # 준비
        mc   = get_captioner(positions, caption_model, base_temp, max_tokens, concurrency)
        clf  = get_deberta_classifier(ckpt_dir)

        # 1) 캡션 6개 + 요약
        with st.spinner("캡션 생성 + 요약 중..."):
            import asyncio
            captions, summary, previews = asyncio.run(mc.process_with_summary(file_bytes, file_name))
        st.success(f"캡션 {len(captions)}개 + 요약 생성 완료 (+{time.perf_counter()-t0:.2f}s)")

        st.subheader("캡션 (6개)")
        for i, c in enumerate(captions, 1):
            st.markdown(f"**#{i}** {c}")

        st.subheader("요약 (이 문장만 분류)")
        st.write(summary)

        # 2) 요약만 분류
        with st.spinner("DeBERTa 분류(요약) 추론 중..."):
            pred = clf.predict_one(summary)
        st.markdown(f"**예측 라벨:** {pred['pred_label']}")
        st.markdown(f"**Posterior:** {max(pred['probs']):.4f}")

        # 3) 해석(Attention/IG/LIME) + GPT 리포트(다국어)
        with st.spinner("해석 산출(Attention/IG/LIME) + GPT 리포트 생성 중..."):
            # 같은 체크포인트로 explainer 구성
            from MODULES.text_explain import TextAttributionEngine, GPTReporter
            import torch
            engine = TextAttributionEngine(
                hf_model=clf.model, hf_tokenizer=clf.tok, id2label=clf.id2label,
                device=("cuda" if torch.cuda.is_available() else "cpu")
            )
            try:
                payload = engine.run_all(summary)  # LIME 필수, SHAP 없음
            except RuntimeError as e:
                st.error(str(e))
                st.stop()

            # 리포트 생성 (호감도 전용 프롬프트가 text_explainer.GPTReporter에 반영되어 있어야 함)
            reporter = GPTReporter(
                model=os.environ.get("GPT_EXPLAIN_MODEL", "gpt-4.1-mini"),
                temperature=0.4, max_tokens=700
            )
            import asyncio
            lay_report, exp_report = asyncio.run(reporter.make_reports(payload, lang_code=lang))

        # 4) 리포트 탭(토글)
        st.subheader("해석 리포트")
        tab1, tab2 = st.tabs(["🧑‍🤝‍🧑 비전문가용", "🧪 전문가용"])
        with tab1:
            st.markdown(lay_report)
        with tab2:
            st.markdown(exp_report)

        # 5) 프리뷰(추출 프레임 확인)
        st.subheader("미리보기")
        cols = st.columns(3)
        for i, img in enumerate(previews):
            with cols[i % 3]:
                st.image(img, caption=f"preview {i}", use_column_width=True)

        # (선택) 원본 동영상 미리보기
        if is_video(file_name):
            st.markdown("#### 업로드한 원본 동영상 미리보기")
            st.video(io.BytesIO(file_bytes))

            ## ㅁㄴㅇㅁㅇㄴ