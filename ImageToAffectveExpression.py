import os
import csv
import base64
from typing import List
from openai import OpenAI


IMAGE_DIR = "image-lifeA"
OUTPUT_CSV = "csv/affective_expressions.csv"
MODEL = "gpt-5.1"


def encode_image_to_data_url(image_path: str) -> str:
    ext = os.path.splitext(image_path)[1].lower()
    mime = "jpeg" if ext in [".jpg", ".jpeg"] else "png"
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/{mime};base64,{b64}"


def list_images(directory: str) -> List[str]:
    files = [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, f))
        and f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    def natural_key(p: str):
        name = os.path.splitext(os.path.basename(p))[0]
        return (0, int(name)) if name.isdigit() else (1, name.lower())
    return sorted(files, key=natural_key)


def describe_image(client: OpenAI, image_path: str) -> str:
    data_url = encode_image_to_data_url(image_path)
    prompt = (
        "다음 재료/표면 이미지를 보고 CMF(색상·소재·마감) 관점의 감성 묘사를 한국어로 2~3문장으로만 작성해줘. "
        "과장 없이 구체적이고, 색/광택/질감/온기/무게감 등 느낌을 중심으로 표현하고, 문장 외의 불필요한 접두사는 쓰지 마.\n"
        "참고 어휘(필수는 아님): 부드러운, 차분한, 선명한, 은은한, 묵직한, 화사한, 고요한, 강렬한, 따뜻한, 차가운, "
        "명료한, 자연스러운, 매트한, 글로시한, 새틴 느낌, 실키한, 금속광의, 미세하게 반짝이는, 은은하게 빛나는, "
        "매끄러운, 거친, 보송한, 포근한, 시원한, 단단한, 무게감 있는, 유연한, 텍스처가 느껴지는, 촉촉한, 건조한, "
        "견고한, 모던한, 미니멀한, 클래식한, 우아한, 고급스러운, 미래적인, 감각적인, 자연 친화적인, 산업적인, "
        "빈티지한, 편안한, 정제된, 세련된, 스포티한, 프리미엄 느낌의, 아날로그적인, 하이테크적인, 입체적인, "
        "균일한, 부드럽게 흐르는, 미세한 패턴의, 조밀한, 촘촘한, 섬세한, 날카로운 느낌의, 유기적인, 메탈릭한, "
        "우디한, 스톤 느낌의, 패브릭 같은, 종이 질감의, 글래스 느낌의, 샌드 블라스트 같은, 브러쉬 텍스처의, "
        "세라믹 느낌의, 코팅된 듯한, 파우더리한, 소프트터치 느낌의, 미러 피니시의, 노을 같은, 모래 같은, 구름 같은, 눈처럼.\n"
        "참고 표현(예시 2~3문장):\n"
        "- 알루미늄 소재의 회색은 차가운 인상을 주고 부드러운 표면과 합쳐져 고급스러운 느낌을 준다.\n"
        "- 광택있는 표면은 촉촉하게 젖은 듯한 투명감을 가지고 있어 마치 빨간색 잉크가 안쪽에서 퍼지는 듯하다.\n"
        "- 공기처럼 가벼운 투명감과 은은한 깊이, 안개 낀 아침의 빛을 머금은 듯 하다."
    )
    resp = client.responses.create(
        model=MODEL,
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text", "text": prompt},
                {"type": "input_image", "image_url": data_url},
            ],
        }],
    )
    return (resp.output_text or "").strip()


def main() -> None:
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    image_paths = list_images(IMAGE_DIR)
    if not image_paths:
        print("이미지 파일을 찾지 못했습니다.")
        return

    client = OpenAI()  # OPENAI_API_KEY 환경변수가 필요합니다.

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "affective_korean"])
        for path in image_paths:
            desc = describe_image(client, path)
            writer.writerow([os.path.basename(path), desc])
            print(f"완료: {os.path.basename(path)}")

    print(f"저장됨: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()

