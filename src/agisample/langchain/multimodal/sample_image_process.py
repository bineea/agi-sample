from pathlib import Path

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from PIL import Image
from pytesseract import pytesseract


DEFAULT_IMAGE_PATH = Path.home() / "Pictures" / "Screenshots" / "屏幕截图 2024-09-23 184625.png"


def extract_image_boxes(image_path: Path = DEFAULT_IMAGE_PATH) -> str:
    image = Image.open(image_path)
    return pytesseract.image_to_boxes(image)


def parse_boxes_to_json(text: str) -> str:
    template = """解析图片内容，并整理为 JSON 格式输出：
{text}"""
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"text": text})


def main() -> None:
    boxes = extract_image_boxes()
    result = parse_boxes_to_json(boxes)
    print(result)


if __name__ == "__main__":
    main()

