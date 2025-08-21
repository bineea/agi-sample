# 必须包含excel的文本内容，否则llm容易将繁体字直接转换为简体字
import base64
import re
from io import BytesIO
from pathlib import Path

from PIL import Image
from dotenv import load_dotenv, find_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.chat import _TextTemplateParam, HumanMessagePromptTemplate, _ImageTemplateParam, \
    MessagesPlaceholder
from langchain_openai import ChatOpenAI

system_no_example = """
你是联想（Lenovo）的客户付款文档数据解析和理解的专业财务助手。
请根据以下指引，解析指定文件路径{file_path}的excel文件，提取并整理客户付款文档中符合要求的相关属性和值。

# 思考
{thought}

# 处理步骤
1. 仔细解析并理解客户付款文档的全部内容，理解整体上下文，然后逐页阅读以确保理解每一页的内容。
2. 仔细解析文档内容，理解每一个数据值，合并被拆分的内容，确保数据值的完整性，特别注意以下情况：
    2.1. 检查当前页表格数据是否完整，如果下一页表格数据的第一行数据是当前页表格数据被拆分的一部分数据内容，请将当前页表格数据与下一页表格数据的第一行数据合并，确保当前页表格数据完整且无丢失
3. 仔细解析文档当前页和下一页的表格数据，根据上下文，理解每一行的数据，确保数据的完整性和一致性，特别注意以下情况：
    3.1. 检查当前页表格数据是否完整，如果下一页表格数据的第一行数据是当前页表格数据延续的一部分数据内容，请将当前页表格数据与下一页表格数据的第一行数据合并，确保当前页表格数据完整且无丢失
4. 理解客户付款文档内容，客户付款文档包含两部分信息：付款信息和关联订单信息
    4.1. 付款信息: 付款的主要信息，聚焦整体付款情况，即谁付款、多少钱、何时付款等
    4.2. 关联订单信息: 与付款相关的明细信息，列出付款的具体明细，即该笔款项涉及哪些订单或发票，每个订单或发票对应的金额、税额、编号等
5. 逐行解析文档当前页数据，提取文档当前页的付款信息数据:
    5.1. payment customer name: 付款客户名称，排除与联想（Lenovo）或摩托罗拉（Motorola）相关的内容。
    5.2. payment date: 付款日期。
    5.3. payment amount: 付款总金额，客户付款文档中明确存在的付款总金额，严格禁止任何数据推测、编造或数学运算。
    5.4. payment reference code: 付款参考编码。
    5.5. **检查每个数据**是否符合要求，数据是否合理，是否有充足的依据，如果没有找到对应的数据，需要重点检查是否遗漏文档数据；存在不通过的检查项，则重新执行5.1，5.2，5.3，5.4步骤提取数据。
6. 逐行解析文档当前页数据，提取并整理文档当前页的关联订单信息数据，并将当前页符合要求的关联订单信息数据整理为集合:
    6.1. so code: 销售订单编号，符合**^[Gg4][A-Za-z0-9][0-9]{{8}}$**的数据，例如: G123456789、4123456789。
    6.2. billing code: 开票凭证号，符合**^[Hh6][A-Za-z0-9][0-9]{{8}}$**的数据，例如: H123456789、6123456789。
    6.3. billing reference: 开票参考号，必须与so code和billing code为不同的数据，针对印度地区为符合**^RV[0-9]{{12}}$**的数据，针对台湾地区为符合**^[A-Z]{{2}}[0-9]{{12}}$**的数据，针对阿根廷地区为符合的**^[0-9]{{4}}$**数据（Nro值），其他地区请自行分析提取，即类似RV123456789012、RV123456789 012、EY12345678、2378、20043272、4281431的数据。
    6.4. tds amount: 源头扣税金额，必须与vat amount和wht amount为不同的金额数据，若为负数，转换为正数；请特别注意tds amount和wht amount的区分，如果存在tds类似术语时，优先尝试将金额设置为tds amount的值。
    6.5. vat amount: 增值税金额，必须与tds amount和wht amount为不同的金额数据，若为负数，转换为正数。
    6.6. wht amount: 代扣税金额，必须与tds amount和vat amount为不同的金额数据，若为负数，转换为正数；请特别注意tds amount和wht amount的区分，如果存在withholding或wht类似术语时，优先尝试将金额设置为wht amount的值。
    6.7. paid amount: 付款金额，如果存在多个付款金额，则为扣除税款后的净付款金额。
    6.8. **检查每个数据**是否符合要求，数据是否合理，是否有充足的依据，数据是否重复，如果没有找到对应的数据，需要重点检查是否遗漏文档数据；如果存在不通过的检查项，则重新执行6.1，6.2，6.3，6.4，6.5，6.6，6.7步骤提取数据。
    6.9. 重复执行6步骤，直到提取到文档当前页的所有关联订单信息数据。
7. **评估上一次提取数据**{observation}，检查上一次提取数据是否符合要求，数据是否合理，是否有充足的依据，数据是否属于文档当前页。
8. **结合思考内容**{thought}，以及上一次提取数据的评估结果，仔细评估所有提取到的文档当前页数据，检查是否完整执行了所有步骤，是否遗漏数据，数据是否符合要求，数据是否合理，是否有充足的依据，数据是否属于文档当前页，存在不通过的检查项，则重新执行1，2，3，4，5，6，7，8步骤。
9. 确定文档当前页的最终结果数据，确保最终结果数据的真实性和完整性和准确性和一致性。

# 注意事项
- 所有属性的value必须符合当前属性的格式要求，否则**设置当前属性的value为空字符串**。
- 所有value必须保留文档中数据原来的格式，如果金额相关数据存在千位分隔符，金额相关数据就需要保留千位分隔符的格式，但是金额相关数据需要去除币种符号，如：$、¥、€、£、₹、AUR、USD、INR等。
- 严禁编造或猜测source field和value。
- 付款信息数据和关联订单信息数据中每个属性必须来源于不同的数据。
- 所有value必须保留文档中数据原来的格式，如果金额相关数据存在千位分隔符，金额相关数据就需要保留千位分隔符的格式，但是金额相关数据需要去除币种符号，如：$、¥、€、£、₹、AUR、USD、INR等。
- 区分tds amount和wht amount，如果存在withholding或wht类似术语时，优先尝试金额设置为wht amount的值，如果存在tds类似术语时，优先尝试将金额设置为tds amount的值。
- 忽略任何与联想（Lenovo）或摩托罗拉（Motorola）相关的名称数据。
- 忽略与付款信息和关联订单信息无关的内容。
- 确保证据准确、完整，不进行任何数据推测、编造或数学运算。
- 仔细区分文档当前页数据和下一页的数据，只提取文档当前页的数据共{total_field}个有效字段，确保不要遗漏当前页的任何一个字段。

请始终遵循以上指引。
"""

_ = load_dotenv(find_dotenv())

def convert_to_markdown(file_path: Path) -> str:
    """
    Convert the content of an Excel file to markdown format.
    This function uses the MarkItDown library to read the Excel file and convert it to markdown.
    """
    from markitdown import MarkItDown

    md = MarkItDown()
    result = md.convert(file_path)
    return result.text_content


def image_to_base64(image_path: Path) -> str:
    image = Image.open(image_path)
    buffered = BytesIO()
    image.save(buffered, format=image.format if image.format is not None else "JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def extract_python_code(markdown_string: str) -> str:
    # Strip whitespace to avoid indentation errors in LLM-generated code
    markdown_string = markdown_string.strip()

    # Regex pattern to match Python code blocks
    pattern = r"```[\w\s]*python\n([\s\S]*?)```|```([\s\S]*?)```"

    # Find all matches in the markdown string
    matches = re.findall(pattern, markdown_string, re.IGNORECASE)

    # Extract the Python code from the matches
    python_code = []
    for match in matches:
        python = match[0] if match[0] else match[1]
        python_code.append(python.strip())

    if len(python_code) == 0:
        return markdown_string

    return python_code[0]


file_path = Path(r"E:\document\CASH相关\Remittance文件\邮件附件Excel\SIT-AP\Lenovo HK payment listing 260325.xlsx")
content = convert_to_markdown(file_path)
image = image_to_base64(Path(r"E:\document\CASH相关\Remittance文件\邮件附件Excel\SIT-AP\Lenovo HK payment listing 260325截图.png"))


PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
        (
            "system",
            system_no_example,
        ),
        HumanMessagePromptTemplate.from_template([
            _TextTemplateParam(text="请根据以下内容，提取并整理客户付款文档中符合要求的相关属性和值。输出10条数据，并判断是否输出了所有数据。"),
            _TextTemplateParam(text="这是文档当前页面的文本内容：{content}"),
            _TextTemplateParam(text="这是显示文档当前页面内容的图像："),
            _ImageTemplateParam(image_url="data:image/JPEG;base64,{image}"),
        ]),
        MessagesPlaceholder("messages"),
    ])

chain = (
        PROMPT_TEMPLATE
        | ChatOpenAI(model="gpt-4o", temperature=0)
)

messages = []

for i in range(10):

    response = chain.invoke({
        "file_path": file_path,
        "content": content,
        "image": image,
        "thought": "数据可能是字符串，也可能是浮点数，谨慎处理数据格式。金额数据保留小数位",
        "observation": "",
        "total_field": 1000,
        "messages": messages
    })

    print(response.content)

    # If no response with text is found, return the first response's content (which may be empty)
    messages.append(response.content)
    human_message = HumanMessagePromptTemplate.from_template([_TextTemplateParam(text="继续输出10条数据，并判断是否输出了所有数据。")])
    messages.append(human_message.format_messages()[0])




