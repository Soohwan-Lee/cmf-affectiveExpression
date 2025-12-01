from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


def get_base_and_dirs() -> Tuple[Path, Path, Path]:
    base_dir = Path(__file__).resolve().parent
    raw_dir = base_dir / "affectiveExpression_raw"
    out_dir = base_dir / "affectiveExpression_gpt"
    out_dir.mkdir(parents=True, exist_ok=True)
    return base_dir, raw_dir, out_dir


def read_csv_smart(csv_path: Path) -> pd.DataFrame:
    encodings_to_try = ["utf-8-sig", "utf-8", "cp949"]
    last_err = None
    for enc in encodings_to_try:
        try:
            return pd.read_csv(csv_path, encoding=enc)
        except Exception as e:  # noqa: BLE001
            last_err = e
    raise RuntimeError(f"CSV 읽기 실패: {csv_path} ({last_err})")


_HEADER_PREFIX_RE = re.compile(r"^\s*\d+\.\s*")


def clean_header(header: str) -> str:
    if header is None:
        return ""
    header = str(header)
    header = _HEADER_PREFIX_RE.sub("", header)
    return header.strip()


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    cleaned_columns: List[str] = [clean_header(c) for c in df.columns]
    df = df.copy()
    df.columns = cleaned_columns
    return df


def collect_sentences(series: pd.Series) -> List[str]:
    sentences: List[str] = []
    for value in series.dropna().astype(str):
        text = value.strip()
        if text:
            sentences.append(text)
    return sentences


def call_openai_merge(sentences: List[str], model: str = "gpt-5.1") -> str:
    """
    sentences를 하나의 문단으로 합치도록 OpenAI에 요청합니다.
    - 모델: gpt-5.1
    - 프롬프트: 한국어, ~음. 체 요구
    """
    if not sentences:
        return ""

    prompt_header = (
        "아래 문장들은 소재에 대한 감성언어들을 표현한거야. "
        "전부 합쳐서 문단으로 만들어 주고, 문장 끝맺음을 ~음. 체로 해줘. "
        "출력은 하나의 문단만 제공해줘."
    )
    bullet_list = "\n".join(f"- {s}" for s in sentences)
    user_content = f"{prompt_header}\n\n문장 목록:\n{bullet_list}"

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("환경 변수 OPENAI_API_KEY가 설정되어 있지 않습니다.")

    # OpenAI 최신 SDK 호환: chat.completions 우선, 실패 시 responses API 시도
    try:
        from openai import OpenAI  # type: ignore

        client = OpenAI(api_key=api_key)
        # 1) chat.completions 먼저 시도
        try:
            chat = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "당신은 간결하고 정확한 한국어 글쓰기 보조자입니다."},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.3,
            )
            content = chat.choices[0].message.content if chat.choices else ""
            return (content or "").strip()
        except Exception:
            # 2) responses API로 재시도
            try:
                resp = client.responses.create(
                    model=model,
                    input=user_content,
                )
                # openai>=1.0의 helper
                if hasattr(resp, "output_text"):
                    return (resp.output_text or "").strip()
                # 안전 파싱
                try:
                    outputs = getattr(resp, "output", []) or getattr(resp, "outputs", [])
                    if outputs:
                        first = outputs[0]
                        content_items = getattr(first, "content", [])
                        if content_items:
                            text = getattr(content_items[0], "text", "") or ""
                            return text.strip()
                except Exception:
                    pass
                return ""
            except Exception as e:  # noqa: BLE001
                raise RuntimeError(f"OpenAI 호출 실패(responses): {e}") from e
    except ImportError as e:
        raise ImportError(
            "openai 패키지가 필요합니다. 설치: pip install --upgrade openai"
        ) from e


def backoff_call(sentences: List[str], model: str = "gpt-5.1", retries: int = 5) -> str:
    for attempt in range(retries):
        try:
            return call_openai_merge(sentences, model=model)
        except Exception as e:  # noqa: BLE001
            if attempt == retries - 1:
                raise
            # 지수 백오프
            sleep_s = 2 ** attempt
            print(f"OpenAI 호출 재시도 예정({attempt + 1}/{retries}) 대기 {sleep_s}s: {e}")
            time.sleep(sleep_s)
    return ""


def process_file(input_path: Path, output_path: Path, model: str = "gpt-5.1") -> None:
    print(f"[처리 시작] {input_path.name}")
    df = read_csv_smart(input_path)
    df = clean_column_names(df)

    sample_codes: List[str] = list(df.columns)
    results: List[Tuple[str, str]] = []

    for code in sample_codes:
        col_series = df[code]
        sentences = collect_sentences(col_series)
        merged_paragraph = backoff_call(sentences, model=model) if sentences else ""
        results.append((code, merged_paragraph))
        print(f" - {code}: {len(sentences)}개 문장 -> 합침 완료")

    out_df = pd.DataFrame(results, columns=["sample_code", "gpt_paragraph"])
    out_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"[저장 완료] {output_path} (행 수: {len(out_df)})")


def main() -> None:
    _, raw_dir, out_dir = get_base_and_dirs()

    io_map: Dict[str, str] = {
        "beauty.csv": "beauty_gpt.csv",
        "homeAppliance.csv": "homeAppliance_gpt.csv",
        "life.csv": "life_gpt.csv",
    }

    for in_name, out_name in io_map.items():
        in_path = raw_dir / in_name
        out_path = out_dir / out_name
        process_file(in_path, out_path, model="gpt-5.1")

    print("모든 파일 처리 완료.")


if __name__ == "__main__":
    main()

