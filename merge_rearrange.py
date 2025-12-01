from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


def get_dirs() -> Tuple[Path, Path, Path]:
    base_dir = Path(__file__).resolve().parent
    in_dir = base_dir / "affectiveExpression_gpt"
    out_dir = base_dir / "affectiveExpression_gpt_rearrange"
    out_dir.mkdir(parents=True, exist_ok=True)
    return base_dir, in_dir, out_dir


def read_csv_smart(csv_path: Path) -> pd.DataFrame:
    encodings_to_try = ["utf-8-sig", "utf-8", "cp949"]
    last_err: Exception | None = None
    for enc in encodings_to_try:
        try:
            # 'N/A' 같은 값을 NaN으로 오인하지 않도록 문자열로 강제 로드
            return pd.read_csv(csv_path, encoding=enc, dtype=str, keep_default_na=False)
        except Exception as e:  # noqa: BLE001
            last_err = e
    raise RuntimeError(f"CSV 읽기 실패: {csv_path} ({last_err})")


_MULTISPACE_RE = re.compile(r"\s+")


def normalize_code(code: str) -> str:
    """
    매칭 안전성을 위해 샘플코드를 정규화합니다.
    - 좌우 공백 제거
    - 중복 공백을 단일 공백으로
    - 끝의 '/' 제거
    - 대소문자 무시 매칭을 위해 소문자화
    """
    s = str(code or "").strip()
    s = _MULTISPACE_RE.sub(" ", s)
    s = s.rstrip("/")
    return s.lower()


def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    입력 CSV는 보통 'sample_code', 'gpt_paragraph' 두 열입니다.
    만약 컬럼명이 다르면 첫 번째/두 번째 열을 사용합니다.
    """
    if "sample_code" in df.columns and "gpt_paragraph" in df.columns:
        return df[["sample_code", "gpt_paragraph"]].copy()

    # 폴백: 첫 두 열을 사용
    cols = list(df.columns)
    if len(cols) < 2:
        raise ValueError("입력 CSV에 최소 2개 열이 필요합니다.")
    tmp = df[[cols[0], cols[1]]].copy()
    tmp.columns = ["sample_code", "gpt_paragraph"]
    return tmp


def rearrange_file(in_csv: Path, desired_order: List[str], out_csv: Path) -> None:
    print(f"[로드] {in_csv.name}")
    df = read_csv_smart(in_csv)
    df = ensure_columns(df)

    # 원본 코드 -> 행 매핑 (정규화 키 사용)
    mapping: Dict[str, Tuple[str, str]] = {}
    for _, row in df.iterrows():
        orig_code = str(row["sample_code"])
        paragraph = str(row["gpt_paragraph"])
        key = normalize_code(orig_code)
        # 첫 등장 우선
        if key not in mapping:
            mapping[key] = (orig_code, paragraph)

    output_rows: List[Tuple[str, str]] = []
    missing: List[str] = []

    for code in desired_order:
        key = normalize_code(code)
        if key in mapping:
            output_rows.append(mapping[key])
        else:
            missing.append(code)

    out_df = pd.DataFrame(output_rows, columns=["sample_code", "gpt_paragraph"])
    out_df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print(f"[저장] {out_csv} (행 수: {len(out_df)})")
    if missing:
        print(f"  경고: {len(missing)}개 매칭 실패 → {missing}")


def main() -> None:
    _, in_dir, out_dir = get_dirs()

    # beauty 주문 순서
    beauty_order: List[str] = [
        "Gold-박도금",
        "NSF-299QT",
        "Popping Terrazzo A",
        "C-07-P4",
        "NCF-449HY",
        "코드 - 073",
        "NHS-2001SV",
        "B-07-P3",
        "Sunset yellow",
        "NTC2060-WS (Relative-Chrome)",
        "VE-0856 G/18547",
        "바이오 플라스틱 사출 (Bio Plastic Injection Molding)",
        "HJZ17-0350A",
        "HJZ17-0303N",
        "pink",
        "AY1-220804-E71-002-001",
        "코드 - 005",
        "117",
        "A-04-P1",
        "YWB-EP260(유광)",
        "B-03-P1",
        "B-06-P1",
        "Silver1-1",
        "Violet Green Gradation",
        "B-10-P1",
        "C-12-Base",
        "D-11-P3",
        "D-13-Base",
        "D-15-Base",
        "코드 - 038",
        "PVED GR 2",
        "HG-0760GP/",
        "A-02-P4",
        "인탑스CMF샘플012",
        "Champagne-gold-유광",
        "ORANGE 125g",
        "D-01-P3",
        "코드 - 069",
        "Zirconia",
        "M.S_HEX_004",
    ]

    # homeAppliance 주문 순서
    home_order: List[str] = [
        "SVHS020",
        "B-05-P3",
        "M.S_HEX_001",
        "패브릭융착샘플_001",
        "Al-CB300-ANDZ-SV",
        "PL-MD-HL1-WH",
        "01(K Silver 02)",
        "AL-HL01-ANDZ-TI",
        "PL-MD-SAND1-WH",
        "PL-MD-HL1-WH",
        "Gold-무광",
        "249",
        "NCF-462QR",
        "NSE-509QW",
        "GY-HL-03",
        "GY-P22",
        "SG-0790 T/F35021",
        "HJZ17-0043K(MS) (UV G100%)",
        "HJZ17-0100D(AL)",
        "Dark Grey",
        "Light Beige",
        "MRC380L3-M0M352",
        "Silver2-2",
        "C-09-P1",
        "D-12-Base",
        "D-14-Base",
        "HG-0760GP/",
        "AY1-220804-E71-007-001",
        "RTP(4R_B0501-60%)black",
        "N/A",
        "D-06-P2",
        "Al-CB300-ANDZ-DB",
        "NM-5453/G73705 E",
        "237",
        "M.S_HEX_016",
        "E14N2",
        "Holographic Shell A",
        "S-002",
        "EXC-K23-0071",
        "042",
    ]

    # life 주문 순서
    life_order: List[str] = [
        "SU765",
        "YWB-EP260(무광)",
        "INCH-10",
        "ST-101",
        "Expanded Polypropylene(Yellow)",
        "GTE92",
        "인탑스CMF샘플008",
        "NSF-300QC",
        "Zirconia",
        "라이트 아몬드",
        "A104",
        "PVAD B 3",
        "PVED BR 3",
        "90G",
        "HW-346877(Black_gloss)",
        "코르코_01",
        "옥수수가죽",
        "R-157",
        "Popping Terrazzo B",
        "WX-9120 83575 B",
        "GY069",
        "21563-01-180215",
        "코드 - 01",
        "스카이골드",
        "C-01-Base",
        "D-09-P1",
        "YWB-C0502(무광)",
        "C-10-Base",
        "C-08-Base",
        "B-08-P1",
        "C-17-P3",
        "C-21-P1",
        "코드 - 078",
        "19SP-071",
        "코드 - 026",
        "OK019",
        "C-05-P2",
        "124",
        "DW025",
        "C-32-G",
    ]

    rearrange_file(
        in_dir / "beauty_gpt.csv",
        beauty_order,
        out_dir / "beauty_gpt_rearrange.csv",
    )
    rearrange_file(
        in_dir / "homeAppliance_gpt.csv",
        home_order,
        out_dir / "homeAppliance_gpt_rearrange.csv",
    )
    rearrange_file(
        in_dir / "life_gpt.csv",
        life_order,
        out_dir / "life_gpt_rearrange.csv",
    )
    print("모든 파일 정렬 완료.")


if __name__ == "__main__":
    main()

