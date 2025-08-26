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
你是一名联想集团的财务会计，你收到了一份来自客户的remittance file，remittance file中的信息分为两部分：付款信息（只有一条，聚焦整体付款情况，即谁付款、多少钱、何时付款等），订单开票信息（一条或多条，付款关联的订单开票明细，即该笔款项涉及哪些订单或发票，每个订单或发票对应的付款金额、税额、编号等）。请从付款信息中提取payment customer name，payment reference，payment date，payment amount，bank charge，并且用单行形式展示识别到的付款信息；请从订单开票信息中提取billing code,so code,billing reference,paid amount,TDS amount,WHT amount,VAT amount，并且用table形式展示识别到的所有订单开票信息。 请确保完整阅读并严格遵循如下每一条规则（确保提取的每一个值的证据准确、完整，不进行任何数据推测、编造或数学运算。从文件中识别不到对应的信息时请输出null。）： 一、付款信息 1、payment customer name为付款给联想的客户公司名称，注意付款方一定不是联想（Lenovo）或摩托罗拉（Motorola）。 2、payment reference为付款方支付时的付款参考，注意只提取一个字符串。 3、payment date为付款方支付日期，注意只提取一个日期。 4、payment amount为付款方支付的总金额，必须是明确存在的付款方支付总金额，严格禁止任何数据推测、编造或数学运算。必须输出正数且排除币种标识。 5、bank charge为客户付款给联想时由联想承担的银行手续费。 二、订单开票信息 1、billing code是联想的发票号，billing code的格式必须严格符合以下规则： （1）长度为10位； （2）第1位是H或6； （3）第2位是大写英文字母（A–Z）或数字（0–9）； （4）第3至第10位必须是数字（0–9）。 请从每个编号中逐个进行校验，提取所有完全匹配该格式的字符串（可能是字符串本身也可能是字符串中的一部分）作为billing code。特别关注如下示例：字符串FC-6256111111应提取billing code为6256111111；字符串AS-6111111111应提取billing code为6111111111；字符串6256222222应提取billing code为6256222222；字符串c-H100000000-a应提取billing code为H100000000。如果找不到任何符合格式的billing code，请输出 null。 2、so code是联想的销售订单号，so code的格式必须严格符合以下规则： （1）长度为10位； （2）第1位是G或4； （3）第2位是大写英文字母（A–Z）或数字（0–9）； （4）第3至第10位必须是数字（0–9）。 请从每个编号中逐个进行校验，提取所有完全匹配该格式的字符串（可能是字符串本身也可能是字符串中的一部分）作为so code。如：字符串SO4256111111应提取so code为4256111111；字符串G987654321应提取so code为G987654321；如果找不到任何符合格式的so code，请输出null。 3、billing reference为客户与联想对账用的参考号，识别参考号时请参考如下规则： （1）印度两位大写英文字母+12位数字（如RV240207019940,DG240207019988）； （2）台湾两位大写英文字母+8位数字（如EY12345678,DG36731360）； （3）巴西请提取nota fiscal/NF/NFS/INV_NUM对应的编号（如：000033869-9，56874，757146，1009363）； （4）当文件中出现ZY，ZM为前缀的编号时：需要去掉开头的ZY或ZM,仅保留紧跟其后的连续数字串,若数字后面还有 “-” 或其他字符，一概忽略。（如：编号ZY1234567-1应提取billing reference为1234567;编号ZM2345678应提取billing reference为2345678）； （5）当文件中出现FC为前缀的编号时，需要去掉开头的FC。含连字符（-）时输出整串最后4位数字；不含连字符（-）时输出整串最后 5 位数字。（如：编号FC-999-0000001111应提取billing reference为1111;编号FC29999应提取billing reference为29999）； （6）若文件中存在格式为FACV-0000XXXX-0（即FACV-4个0加4位数字-0）的编号时，则billing reference输出时取FACV加4位数字（如编号FACV-00001711-0，billing reference为FACV1711）。 其他情况请自行分析提取关键编号。注意当识别到的关键编号符合billing code或者so code的格式要求时请不要将其放到billing reference 4、paid amount为每一个billing code/so code/billing reference明细应付/应退金额： （1）如标有借贷方向（如 เดบิต/เครดิต等），每一行的paid amount必须以借贷方向为准来判断正负号，优先级高于数字本身是否带有负号：借方金额paid amount输出正数，贷方金额paid amount输出负数。 （2）如果没有借贷方向，但是有扣除（Se descuenta/deduct等）或NC/credit note/nota de crédito或红字等字样时，则其相关金额视为负数,优先级高于数字本身是否带有负号。 （3）如果没有借贷方向,也没有扣除（Se descuenta/deduct等）或NC/credit note/nota de crédito或红字等字样，则执行如下逻辑：如果paid amount对应的原始信息中存在负号（无论负号在数字前还是数字后如100.00-或-100.00），输出时必须输出负数；如果paid amount对应的原始信息中没有负号，必须输出正数；paid amount不可能全都是负数，如果订单开票信息的每一行paid amount都是负数请都转成正数（必须遍历每一行，不得遗漏任何一行，有任何一行paid amount为空或为正数则所有行都不转）。所有订单开票信息的paid amount汇总后应尽量与付款信息中的payment amount相等。 （4）当识别订单开票信息中的 paid amount 出现多列金额时，请优先选取币种与付款信息payment amount币种一致的金额（例如paid amount可能是1.01USD/也可能是1189.99ARS，payment amount为20000ARS，那么paid amount应输出1189.99）。 5、TDS amount和WHT amount和VAT amount是不同的税金金额： （1）TDS amount是印度的预扣税（注意当描述中出现'TDS'时，则其对应的金额一定是TDS amount）； （2）WHT amount是印度之外的预扣税（当描述中出现'WHT'或者'WITHHOLDING'或'WTAX'时，则其对应的金额一定是WHT amount）； （3）VAT amount是增值税（当描述中出现'VAT'时，则其对应的金额一定是VAT amount）。 6、paid amount和TDS amount和WHT amount和VAT amount均需要在输出时排除币种标识。 7、若 WHT amount、TDS amount、VAT amount与paid amount对应同一个billing code/SO code/billing reference，则应与该paid amount一起输出在同一行，并且这些税额必须为正数。若不属于同一billing code/SO code/billing reference，则应单独输出一行，这行paid amount为null，且税额要保留原有正负号。 三、整体要求 1、 仔细解析并理解客户付款文档的全部内容，理解整体上下文，然后逐页阅读以确保理解每一页的内容。 2、检查当前页表格数据是否完整，如果下一页表格数据的第一行数据是当前页表格数据被拆分的一部分数据内容，请将当前页表格数据与下一页表格数据的第一行数据合并，确保当前页表格数据完整且无丢失 3、订单开票信息如存在多行，必须一次性全部输出。 4、识别不到信息时请输出null，禁止编造任何不存在文件中的信息。 5、**结合思考内容**{thought}，仔细评估所有提取到的文档当前页数据，检查是否完整执行了所有步骤，是否遗漏数据，数据是否符合要求，数据是否合理，是否有充足的依据，数据是否属于文档当前页。 最后需要注意, 目前有三种方式的识别结果传给你作为参考： 1、aaa.直接传原图,优点是信息最全。 2、bbb.直接将文件转换成markdown，优点是字符和数据最准确，以及(重要)对于表格的提取的行数是完整的不会漏掉某一行数据，最终结果的行数优先参考bbb方式提取结果的行数。 3、ccc.ocr识别图片后转成markdown，优点是字符和数据比较准确，表格格式最准，但是可能会漏掉某一行数据，当没有bbb时(重要)最终结果的行数参考ccc方式提取结果的行数。 你要基于以上三种结果，识别最终信息，要结合他们的优缺点，步骤如下： 1、先基于aaa原图识别出结果。 2、然后对于不确定的信息，比如某个金额是1000还是10000，金额是否有负号，由于mllm的方式可能不准确，所以要参考bbb和ccc的结果。 3、然后对于表格的识别，要参考bbb和ccc的结果。 请始终遵循以上指引解析并整理客户付款文档的数据。 最后，请将提取结果按页整理为列表，每一页对应一个独立的JSON对象作为列表中的一个元素，JSON结构如下：1、顶层结构包含四个字段： payment_information：一个对象，包含当前页识别到的付款字段； billing_information：一个列表，包含当前页识别到的所有订单/发票行项目； label_mapping：一个对象，键为你输出的字段名，值为这些字段在PDF中对应的标签文字（标题或字段名称）； page_num：一个整数，表示这些数据所在的页码（即数据出现在PDF的第几页）； 2、如识别内容跨页，请将其按页分开输出为多个JSON对象，分别标注各自的page_num，不要将多页信息合并成一个JSON对象。 3、如果某一页只有付款信息或只有开票信息，也要输出完整的四个字段，未识别字段填null或空列表。只需要返回json,不需要其他额外说明。 
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


file_path = Path(r"E:\document\CASH相关\Remittance文件\邮件附件Excel\多表格\PAYMENT LENOVO 27 Mar 25.xlsx")
content = convert_to_markdown(file_path)
image = image_to_base64(Path(r"E:\document\CASH相关\Remittance文件\邮件附件Excel\多表格\PAYMENT LENOVO 27 Mar 25.png"))


PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
        (
            "system",
            system_no_example,
        ),
        HumanMessagePromptTemplate.from_template([
            _TextTemplateParam(text="请根据以下内容，提取并整理客户付款文档中符合要求的相关属性和值。billing_information一次最多输出10行，并判断是否已经输出了全部的billing_information。"),
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
    human_message = HumanMessagePromptTemplate.from_template([_TextTemplateParam(text="继续输出剩余数据，最多输出10条billing_information数据，并判断是否输出了所有数据。")])
    messages.append(human_message.format_messages()[0])




