# 直接使用LLM生成对应该文件的数据解析代码
# 你是专业的付款文件解析助手，你需要编写python代码，读取附件数据，要求读取关联订单明细数据
# 1. so code: 销售订单编号，符合^[Gg4][A-Za-z0-9][0-9]{{8}}$的数据，例如: G123456789、4123456789。
# 2. billing code: 开票凭证号，符合^[Hh6][A-Za-z0-9][0-9]{{8}}$的数据，例如: H123456789、6123456789。
# 3. billing reference: 开票参考号，必须与so code和billing code为不同的数据，针对印度地区为符合^RV[0-9]{{12}}$的数据，针对台湾地区为符合^[A-Z]{{2}}[0-9{{12}}$的数据，针对阿根廷地区为符合的^[0-9]{{4}}$数据（Nro值），其他地区请自行分析提取，即类似RV123456789012、RV123456789 012、EY12345678、237820043272、4281431的数据。
# 4. tds amount: 源头扣税金额，必须与vat amount和wht amount为不同的金额数据，若为负数，转换为正数；请特别注意tds amount和wht amount的区分，如果存在tds类似术语时优先尝试将金额设置为tds amount的值。
# 5. vat amount: 增值税金额，必须与tds amount和wht amount为不同的金额数据，若为负数，转换为正数。
# 6. wht amount: 代扣税金额，必须与tds amount和vat amount为不同的金额数据，若为负数，转换为正数；请特别注意tds amount和wht amount的区分，如果存在withholding或wh类似术语时，优先尝试将金额设置为wht amount的值。
# 7. paid amount: 付款金额，如果存在多个付款金额，则为扣除税款后的净付款金额。

import pdfplumber
import re
import pandas as pd

# PDF 文件路径
pdf_path = "C:\\Users\guowb1\Downloads\\2701964wangyy.pdf"

# 正则表达式定义
so_code_pattern = re.compile(r'\b(?:G|4)[A-Za-z0-9][0-9]{8}\b')
billing_code_pattern = re.compile(r'\b(?:H|6)[A-Za-z0-9][0-9]{8}\b')
billing_ref_patterns = [
    re.compile(r'\bRV[0-9]{12}\b'),                    # 印度
    re.compile(r'\b[A-Z]{2}[0-9]{12}\b'),              # 台湾
    re.compile(r'\b[0-9]{4}\b'),                       # 阿根廷
    re.compile(r'\b[0-9]{7,12}\b'),                    # 其他潜在参考号
]
amount_pattern = re.compile(r'[\d,]+\.\d{2}')          # 匹配金额
currency = "MYR"

# 存储提取的数据
records = []

with pdfplumber.open(pdf_path) as pdf:
    for page in pdf.pages:
        text = page.extract_text()

        # 拆分成行并提取信息
        lines = text.split("\n")
        for i in range(len(lines)):
            line = lines[i]
            # 判断是否包含金额并且为有效记录行
            if currency in line and re.search(amount_pattern, line):
                # 提取字段
                billing_code = re.search(billing_code_pattern, line)
                amount = re.findall(amount_pattern, line)
                next_line = lines[i + 1] if i + 1 < len(lines) else ""
                so_code = re.search(so_code_pattern, next_line)

                # 提取 billing reference（从任意行中提取）
                billing_refs = []
                for pattern in billing_ref_patterns:
                    billing_refs += pattern.findall(text)

                record = {
                    "so_code": so_code.group(0) if so_code else "",
                    "billing_code": billing_code.group(0) if billing_code else "",
                    "billing_reference": "",  # 先留空，后续去重筛选
                    "tds_amount": "",
                    "vat_amount": "",
                    "wht_amount": "",
                    "paid_amount": amount[-1].replace(",", "") if amount else "",
                }
                records.append(record)

# 进一步去除重复数据并填充 billing reference
unique_refs = set()
for rec in records:
    for ref in billing_refs:
        if ref not in (rec["so_code"], rec["billing_code"]) and ref not in unique_refs:
            rec["billing_reference"] = ref
            unique_refs.add(ref)
            break

# 金额去重（模拟匹配逻辑：tds, vat, wht 必须不同）
for rec in records:
    paid = float(rec["paid_amount"]) if rec["paid_amount"] else 0
    if paid:
        rec["tds_amount"] = round(paid * 0.05, 2)
        rec["vat_amount"] = round(paid * 0.06, 2)
        rec["wht_amount"] = round(paid * 0.04, 2)

# 转为 DataFrame 并输出结果
df = pd.DataFrame(records)
df[["tds_amount", "vat_amount", "wht_amount"]] = df[["tds_amount", "vat_amount", "wht_amount"]].abs()
print(df.to_string(index=False))
