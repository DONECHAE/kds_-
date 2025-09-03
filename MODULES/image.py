#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import asyncio
import tempfile
import random
import base64
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from dotenv import load_dotenv

load_dotenv()  # .env 로드 (OPENAI_API_KEY 등)

try:
    from openai import AsyncOpenAI
except Exception as e:
    raise RuntimeError("openai 패키지가 필요합니다. pip install openai") from e


# ==========================
# 설정
# ==========================
DEFAULT_MODEL = os.environ.get("GPT_CAPTION_MODEL", "gpt-4.1-mini")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v"}
DEFAULT_POSITIONS = [0.10, 0.20, 0.45, 0.55, 0.80, 0.90]

DETAILED_PROMPT_EN = (
    "You are analyzing only the human face in an image for subtle visual inconsistencies. "
    "Ignore any other non-facial elements. For each observation, identify the specific facial area "
    "and describe the unusual aspect in precise detail. Remain strictly factual and observational; "
    "do not speculate or draw conclusions. Write your response in one short paragraph of 2–3 sentences only."
)


# ==========================
# 유틸
# ==========================
def _assert_api_key() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY 환경변수가 설정되어 있지 않습니다.")


def _bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _image_bytes_to_bgr(data: bytes) -> Optional[np.ndarray]:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def _to_data_url_from_bgr(img_bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", img_bgr)
    if not ok:
        raise RuntimeError("이미지를 인코딩하지 못했습니다.")
    b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


# ==========================
# 비디오 프레임 추출기
# ==========================
class MediaSampler:
    def __init__(self, positions: Optional[List[float]] = None):
        self.positions = positions or DEFAULT_POSITIONS

    @staticmethod
    def is_image_name(name: str) -> bool:
        return Path(name).suffix.lower() in IMAGE_EXTS

    @staticmethod
    def is_video_name(name: str) -> bool:
        return Path(name).suffix.lower() in VIDEO_EXTS

    def extract_frames_from_video_bytes(self, data: bytes, orig_name: str = "upload.mp4") -> List[np.ndarray]:
        frames: List[np.ndarray] = []
        _, ext = os.path.splitext(orig_name)
        if not ext:
            ext = ".mp4"

        with tempfile.TemporaryDirectory() as td:
            tmp_path = os.path.join(td, f"tmp{ext}")
            with open(tmp_path, "wb") as f:
                f.write(data)

            cap = cv2.VideoCapture(tmp_path)
            if not cap.isOpened():
                return frames

            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # 기본 시도
            if total and total > 0:
                for pos in self.positions:
                    idx = min(int(total * float(pos)), max(total - 1, 0))
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ok, frame = cap.read()
                    if ok and frame is not None:
                        frames.append(frame)

            # 그래도 부족하면 선형 스캔
            if len(frames) < len(self.positions):
                base_total = total if total and total > 0 else 3000
                target_idxs = [min(int(base_total * p), max(base_total - 1, 0)) for p in self.positions]

                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                got = {ti: None for ti in target_idxs}
                cur = 0
                max_scan = total if total and total > 0 else 3000

                while cur < max_scan:
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        break
                    if cur in got and got[cur] is None:
                        got[cur] = frame
                    cur += 1
                    if all(v is not None for v in got.values()):
                        break

                for want in target_idxs:
                    if got[want] is not None:
                        frames.append(got[want])

            cap.release()
        return frames


# ==========================
# OpenAI 캡셔너
# ==========================
class CaptionerAsync:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_temperature: float = 0.7,
        max_tokens: int = 120,
        concurrency: int = 6,
        retries: int = 5,
    ):
        _assert_api_key()
        self.model = model
        self.base_temperature = base_temperature
        self.max_tokens = max_tokens
        self.concurrency = concurrency
        self.retries = retries
        self.client = AsyncOpenAI()
        self.sem = asyncio.Semaphore(concurrency)

    async def _caption_once(self, img_bgr: np.ndarray, temperature: float) -> str:
        data_url = _to_data_url_from_bgr(img_bgr)
        resp = await self.client.responses.create(
            model=self.model,
            input=[{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": DETAILED_PROMPT_EN},
                    {"type": "input_image", "image_url": data_url},
                ],
            }],
            max_output_tokens=self.max_tokens,
            temperature=temperature,
        )
        return resp.output_text.strip()

    async def _caption_with_retry(self, img_bgr: np.ndarray, temperature: float) -> str:
        delay = 1.0
        for attempt in range(1, self.retries + 1):
            try:
                return await self._caption_once(img_bgr, temperature)
            except Exception:
                if attempt == self.retries:
                    return ""
                await asyncio.sleep(delay + random.random() * 0.3)
                delay = min(delay * 2, 10.0)

    async def caption_many(self, images_bgr: List[np.ndarray]) -> List[str]:
        async def _task(img: np.ndarray, temp: float) -> str:
            async with self.sem:
                return await self._caption_with_retry(img, temp)

        if len(images_bgr) == 1:
            base = self.base_temperature
            temps = [max(0.0, min(1.5, base + random.uniform(-0.2, 0.4))) for _ in range(6)]
            tasks = [_task(images_bgr[0], t) for t in temps]
        else:
            temps = [max(0.0, min(1.5, self.base_temperature + random.uniform(-0.15, 0.15)))
                     for _ in images_bgr]
            tasks = [_task(img, t) for img, t in zip(images_bgr, temps)]

        return await asyncio.gather(*tasks)


# ==========================
# (NEW) 캡션 세트 요약기 (메타 대화 요약, 영어 2~3문장)
# ==========================
import re
from collections import Counter

try:
    from openai import AsyncOpenAI  # 이미 위에서 임포트된 경우 무시
except:
    pass

# ── 형용사 추출(영어) 간단 버전: NLTK 없을 때를 고려해 휴리스틱 우선
_EN_WORD_RE = re.compile(r"[A-Za-z][A-Za-z\-']+")
def _extract_adj_en_text(text: str) -> list[str]:
    toks = _EN_WORD_RE.findall(text)
    return [
        w.lower() for w in toks
        if re.search(r"(ful|less|ous|ive|ary|ant|ent|able|ible|ish|al|ic|y)$", w.lower())
    ]

def _adj_stats_en(captions: list[str], topn: int | None = 10) -> tuple[int, list[tuple[str,int,float]]]:
    adjs: list[str] = []
    for c in captions:
        adjs.extend(_extract_adj_en_text(c))
    total = len(adjs)
    ctr = Counter(adjs)
    ordered = ctr.most_common(None if topn is None else topn)
    stats = [(a, c, (c/total if total else 0.0)) for a, c in ordered]
    return total, stats

def _build_system_prompt_en() -> str:
    # ⚠️ 기존 캡셔닝 프롬프트는 건드리지 않음. (요약용 별도 프롬프트)
    return (
        "You are a 'meta dialogue summarizer'. Given multiple frame-level captions and "
        "an adjective frequency distribution, write a natural, cohesive 2–3 sentence summary "
        "as if two people are talking. Prefer phrasing like “The two people are talking about …”. "
        "Reflect the proportions of adjectives in the tone/mood intensity. "
        "Avoid brands or incidental background details; focus on expressions, interaction, and atmosphere. "
        "Output only the 2–3 sentence summary in English."
    )

def _build_user_prompt_en(captions: list[str], ratios: list[tuple[str,int,float]]) -> str:
    caps_block = "\n".join(f"- {c}" for c in captions)
    if ratios:
        adj_block = "\n".join([f"- {a}: {cnt} ({ratio:.2%})" for a, cnt, ratio in ratios])
    else:
        adj_block = "- (no adjectives found)"
    return (
        f"[CAPTIONS]\n{caps_block}\n\n"
        f"[ADJECTIVE DISTRIBUTION (Top)]\n{adj_block}\n\n"
        "Use these proportions when choosing tone and mood. "
        "Return only the 2–3 sentence summary in English."
    )

class CaptionSetSummarizerAsync:
    """
    6개 캡션(문자열 리스트)을 입력으로 받아 1개의 영어 요약(2~3문장)을 생성.
    - 프롬프트는 위의 요약용 프롬프트 사용(캡셔닝 프롬프트는 변경 없음)
    """
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        max_tokens: int = 160,
        temperature: float = 0.7,
        topn_adj: int | None = 10,
        retries: int = 5,
    ):
        _assert_api_key()
        self.client = AsyncOpenAI()
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.topn_adj = topn_adj
        self.retries = retries

    async def summarize(self, captions: list[str]) -> str:
        # 형용사 통계
        _, ratios = _adj_stats_en(captions, topn=self.topn_adj)
        sys_prompt = _build_system_prompt_en()
        user_prompt = _build_user_prompt_en(captions, ratios)

        async def _once():
            resp = await self.client.responses.create(
                model=self.model,
                input=[
                    {"role": "system", "content": [{"type": "input_text", "text": sys_prompt}]},
                    {"role": "user",   "content": [{"type": "input_text", "text": user_prompt}]},
                ],
                max_output_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            return (resp.output_text or "").strip()

        # 재시도(지수 백오프)
        delay = 1.0
        for attempt in range(1, self.retries + 1):
            try:
                text = await _once()
                if text:
                    # 요구사항: "한 문장에서 한 문단" → 기본 프롬프트가 2~3문장 권고이나,
                    # 모델이 더 길게 쓰면 한 단락으로 오므로 추가 제약은 걸지 않음.
                    return text
                else:
                    raise RuntimeError("Empty response")
            except Exception:
                if attempt == self.retries:
                    return ""
                import asyncio, random as _rnd
                await asyncio.sleep(delay + _rnd.random() * 0.3)
                delay = min(delay * 2, 10.0)

# ==========================
# 통합 클래스
# ==========================
class MediaCaptioner:
    def __init__(
        self,
        positions: Optional[List[float]] = None,
        model: str = DEFAULT_MODEL,
        base_temperature: float = 0.7,
        max_tokens: int = 120,
        concurrency: int = 6,
        retries: int = 5,
    ):
        self.sampler = MediaSampler(positions=positions)
        self.captioner = CaptionerAsync(
            model=model,
            base_temperature=base_temperature,
            max_tokens=max_tokens,
            concurrency=concurrency,
            retries=retries,
        )

    def _is_image(self, name: str) -> bool:
        return self.sampler.is_image_name(name)

    def _is_video(self, name: str) -> bool:
        return self.sampler.is_video_name(name)

    async def process(self, file_bytes: bytes, file_name: str) -> Tuple[List[str], List[np.ndarray]]:
        if self._is_image(file_name):
            img_bgr = _image_bytes_to_bgr(file_bytes)
            if img_bgr is None:
                raise RuntimeError("이미지를 디코딩하지 못했습니다.")
            captions = await self.captioner.caption_many([img_bgr])
            return captions, [_bgr_to_rgb(img_bgr)]

        elif self._is_video(file_name):
            frames_bgr = self.sampler.extract_frames_from_video_bytes(file_bytes, orig_name=file_name)
            if not frames_bgr:
                raise RuntimeError("동영상에서 프레임을 추출하지 못했습니다.")
            captions = await self.captioner.caption_many(frames_bgr)
            previews = [_bgr_to_rgb(f) for f in frames_bgr]
            return captions, previews

        else:
            raise RuntimeError("지원하지 않는 파일 형식입니다.")
        
    async def process_with_summary(self, file_bytes: bytes, file_name: str) -> tuple[list[str], str, list[np.ndarray]]:
        """
        (새 기능) 단일 파일 처리 → (캡션 6개, 요약 1개, 미리보기 이미지들) 반환
        - 이미지: 동일 이미지로 6캡션 생성
        - 동영상: 6프레임 캡션 생성
        - 요약: 위 6개 캡션을 기반으로 영어 2~3문장 요약 1개
        """
        # 1) 캡션 생성
        captions, previews = await self.process(file_bytes, file_name)

        # 2) 요약 생성
        summarizer = CaptionSetSummarizerAsync(
            model=self.captioner.model,          # 캡션과 같은 모델 사용(원하면 별도 환경변수로 분리 가능)
            max_tokens=160,
            temperature=self.captioner.base_temperature,  # 기본 temp 재사용
            topn_adj=10,
            retries=5,
        )
        summary = await summarizer.summarize(captions)
        return captions, summary, previews
