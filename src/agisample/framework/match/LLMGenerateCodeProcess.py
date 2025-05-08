# 必须包含excel的文本内容，否则llm容易将繁体字直接转换为简体字
import base64
import re
from io import BytesIO
from pathlib import Path

from PIL import Image
from dotenv import load_dotenv, find_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.chat import _TextTemplateParam, HumanMessagePromptTemplate, _ImageTemplateParam
from langchain_openai import ChatOpenAI

system_no_example = """
你是联想（Lenovo）的客户付款文档数据解析和理解的专业财务助手。
请根据以下指引，编写python代码，使用openpyxl读取指定文件路径{file_path}的excel文件，提取并整理客户付款文档中符合要求的相关属性和值。

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

请始终遵循以上指引生成python代码。
"""

_ = load_dotenv(find_dotenv())

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


PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
        (
            "system",
            system_no_example,
        ),
        HumanMessagePromptTemplate.from_template([
            _TextTemplateParam(text="请根据以下内容，编写python代码，使用openpyxl读取指定文件路径{file_path}的excel文件，提取并整理客户付款文档中符合要求的相关属性和值。"),
            _TextTemplateParam(text="这是文档当前页面的文本内容：{content}"),
            _TextTemplateParam(text="这是显示文档当前页面内容的图像："),
            _ImageTemplateParam(image_url="data:image/JPEG;base64,{image}"),
        ])
    ])

chain = (
        PROMPT_TEMPLATE
        | ChatOpenAI(model="gpt-4o-mini", temperature=0)
)

response = chain.invoke({
    "file_path": Path("E:\document\CASH相关\Remittance文件\PaymentAdvice20241106170146（客户发出的）.xlsx"),
    "content": """
| 0                    | 1                                 | 2          | 3             | 4        |
|:---------------------|:----------------------------------|:-----------|:--------------|:---------|
| 聯強國際股份有限公司 | nan                               | nan        | nan           | nan      |
| 付款明細             | nan                               | nan        | nan           | nan      |
| nan                  | nan                               | nan        | nan           | nan      |
| 付款對象             | 荷蘭商聯想股份有限公司 台灣分公司 | nan        | nan           | nan      |
| 付款金額             | NTD 19,139,525.00                 | nan        | nan           | nan      |
| 付款日               | 2024/11/06                        | nan        | nan           | nan      |
| nan                  | nan                               | nan        | nan           | nan      |
| 序號                 | 憑證日期                          | 憑證號碼   | 付款金額      | 沖帳資料 |
| 1                    | 2024/09/02                        | EY17444515 | 1475040       | nan      |
| 2                    | 2024/09/02                        | EY17444557 | 9094          | nan      |
| 3                    | 2024/09/03                        | EY17444613 | 288750        | nan      |
| 4                    | 2024/09/03                        | EY17444614 | 30450         | nan      |
| 5                    | 2024/09/03                        | EY17444617 | 5565          | nan      |
| 6                    | 2024/09/03                        | EY17444649 | 71820         | nan      |
| 7                    | 2024/09/03                        | EY17444652 | 220931        | nan      |
| 8                    | 2024/09/04                        | EY17444694 | 26250         | nan      |
| 9                    | 2024/09/04                        | EY17444695 | 59640         | nan      |
| 10                   | 2024/09/04                        | EY17444696 | 714000        | nan      |
| 11                   | 2024/09/04                        | EY17444729 | 433125        | nan      |
| 12                   | 2024/09/04                        | EY17444733 | 625644        | nan      |
| 13                   | 2024/09/04                        | EY17444738 | 735000        | nan      |
| 14                   | 2024/09/04                        | EY17444739 | 567000        | nan      |
| 15                   | 2024/09/04                        | EY17444741 | 14537         | nan      |
| 16                   | 2024/09/04                        | EY17444758 | 819000        | nan      |
| 17                   | 2024/09/04                        | EY17444759 | 590625        | nan      |
| 18                   | 2024/09/05                        | EY17444769 | 926100        | nan      |
| 19                   | 2024/09/05                        | EY17444816 | 7876          | nan      |
| 20                   | 2024/09/06                        | EY17444822 | 345443        | nan      |
| 21                   | 2024/09/06                        | EY17444857 | 7875          | nan      |
| 22                   | 2024/09/06                        | EY17444858 | 21525         | nan      |
| 23                   | 2024/09/06                        | EY17444880 | 511875        | nan      |
| 24                   | 2024/09/06                        | EY17444881 | 1967858       | nan      |
| 25                   | 2024/09/06                        | EY17444887 | 1445903       | nan      |
| 26                   | 2024/09/06                        | EY17444901 | 713664        | nan      |
| 27                   | 2024/09/06                        | EY17444904 | 29925         | nan      |
| 28                   | 2024/09/06                        | EY17444905 | 488615        | nan      |
| 29                   | 2024/09/06                        | EY17444907 | 2801400       | nan      |
| 30                   | 2024/09/06                        | EY17444924 | 636300        | nan      |
| 31                   | 2024/09/06                        | EY17444925 | 33023         | nan      |
| 32                   | 2024/09/06                        | EY17444926 | 297203        | nan      |
| 33                   | 2024/09/07                        | EY17444975 | 4300          | nan      |
| 34                   | 2024/09/07                        | EY17444979 | 33020         | nan      |
| 35                   | 2024/09/08                        | EY17445016 | 1732500       | nan      |
| 36                   | 2024/09/09                        | EY17445039 | 61753         | nan      |
| 37                   | 2024/09/09                        | EY17445051 | 1638          | nan      |
| 38                   | 2024/09/09                        | EY17445052 | 2457          | nan      |
| 39                   | 2024/09/09                        | EY17445108 | 74550         | nan      |
| 40                   | 2024/09/09                        | EY17445112 | 955537        | nan      |
| 41                   | 2024/09/09                        | EY17445124 | 14364         | nan      |
| 42                   | 2024/09/10                        | EY17445165 | 3591          | nan      |
| 43                   | 2024/09/10                        | EY17445176 | 1638          | nan      |
| 44                   | 2024/10/22                        | AZ17450016 | -108360       | nan      |
| 45                   | 2024/10/22                        | AZ17450016 | -95760        | nan      |
| 46                   | 2024/10/22                        | AZ17450016 | -14700        | nan      |
| 47                   | 2024/10/23                        | ZA17445261 | -228987       | nan      |
| 48                   | 2024/10/23                        | ZA17445261 | -105987       | nan      |
| 49                   | 2024/10/23                        | ZA17445261 | -91350        | nan      |
| 50                   | 2024/10/23                        | ZA17445261 | -21735        | nan      |
| nan                  | 合計                              | nan        | 19,139,525.00 | nan      |
    """,
    "image": image_to_base64(Path("E:\document\CASH相关\Remittance文件\PaymentAdvice20241106170146（客户发出的）.png")),
    "thought": "所有数据均转换为字符串再进行逻辑处理，同时金额数据保留小数位",
    "observation": "",
    "total_field": 1000,
})

# If no response with text is found, return the first response's content (which may be empty)
result = response.content

plotly_code = extract_python_code(result)

print("Extracted Python code: \n" + plotly_code)
ldict = {
    "file_path": Path("E:\document\CASH相关\Remittance文件\PaymentAdvice20241106170146（客户发出的）.xlsx")
}
try:
    import PyPDF2
    import fitz
    from PyPDF2 import PdfReader
    import pandas as pd
    import openpyxl
    exec(plotly_code, globals(), ldict)
    print(ldict)
    # print(json.dumps(ldict, indent=2))
except Exception as e:
    raise RuntimeError(f"Error executing code: {e}")
