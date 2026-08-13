# 直接使用LLM生成对应该文件的数据解析代码
# 你是专业的付款文件解析助手，你需要编写python代码，读取附件数据，要求读取关联订单明细数据
# 1. so code: 销售订单编号，符合^[Gg4][A-Za-z0-9][0-9]{{8}}$的数据，例如: G123456789、4123456789。
# 2. billing code: 开票凭证号，符合^[Hh6][A-Za-z0-9][0-9]{{8}}$的数据，例如: H123456789、6123456789。
# 3. billing reference: 开票参考号，必须与so code和billing code为不同的数据，针对印度地区为符合^RV[0-9]{{12}}$的数据，针对台湾地区为符合^[A-Z]{{2}}[0-9{{12}}$的数据，针对阿根廷地区为符合的^[0-9]{{4}}$数据（Nro值），其他地区请自行分析提取，即类似RV123456789012、RV123456789 012、EY12345678、237820043272、4281431的数据。
# 4. tds amount: 源头扣税金额，必须与vat amount和wht amount为不同的金额数据，若为负数，转换为正数；请特别注意tds amount和wht amount的区分，如果存在tds类似术语时优先尝试将金额设置为tds amount的值。
# 5. vat amount: 增值税金额，必须与tds amount和wht amount为不同的金额数据，若为负数，转换为正数。
# 6. wht amount: 代扣税金额，必须与tds amount和vat amount为不同的金额数据，若为负数，转换为正数；请特别注意tds amount和wht amount的区分，如果存在withholding或wh类似术语时，优先尝试将金额设置为wht amount的值。
# 7. paid amount: 付款金额，如果存在多个付款金额，则为扣除税款后的净付款金额。

import re
import pandas as pd
import PyPDF2
import io


def extract_text_from_pdf(pdf_file):
    """Extract text from PDF file."""
    text = ""
    try:
        # Open the PDF file
        pdf_reader = PyPDF2.PdfReader(pdf_file)

        # Get the number of pages
        num_pages = len(pdf_reader.pages)

        # Extract text from each page
        for page_num in range(num_pages):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()

    except Exception as e:
        print(f"Error extracting text from PDF: {e}")

    return text


def parse_payment_document(text):
    # Initialize results
    so_codes = []
    billing_codes = []
    billing_references = []

    # Define regex patterns
    so_pattern = re.compile(r'([Gg4][A-Za-z0-9][0-9]{8})')
    billing_pattern = re.compile(r'([Hh6][A-Za-z0-9][0-9]{8})')

    # Extract line items
    lines = []
    for line in text.split('\n'):
        line = line.strip()
        if line.startswith('RE '):
            parts = re.split(r'\s+', line, maxsplit=5)
            if len(parts) >= 3:
                document_num = parts[1]
                billing_code = parts[2]
                reference = parts[3] if len(parts) > 3 else ""

                # Extract amounts (last part contains MYR)
                amount_str = ""
                for i in range(len(parts) - 1, -1, -1):
                    if "MYR" in parts[i]:
                        amount_str = parts[i]
                        break

                # Process the line item
                lines.append({
                    'document_num': document_num,
                    'billing_code': billing_code,
                    'reference': reference,
                    'amount_str': amount_str
                })

    # Process each line to extract the required information
    for line in lines:
        # Extract billing code
        billing_match = billing_pattern.search(line['billing_code'])
        if billing_match:
            billing_codes.append(billing_match.group(1))
        else:
            billing_codes.append(None)

        # Extract SO code from the reference field
        so_match = so_pattern.search(line['reference'])
        if so_match:
            so_codes.append(so_match.group(1))
        else:
            so_codes.append(None)

        # Extract billing reference (we'll assume it's in the reference field but different from SO code)
        # For this document, it appears the references are in the format of 8-digit numbers
        ref_candidates = re.findall(r'(\d{7,})', line['reference'])
        valid_ref = None
        for ref in ref_candidates:
            # Check if it's not a SO code or billing code
            if not re.match(so_pattern, ref) and not re.match(billing_pattern, ref):
                valid_ref = ref
                break

        billing_references.append(valid_ref)

    # Extract the total paid amount
    paid_amount = 0
    total_match = re.search(r'Sum total\s+([\d,]+\.\d+)\s+MYR', text)
    if total_match:
        paid_amount = float(total_match.group(1).replace(',', ''))

    # No explicit TDS, VAT or WHT amounts found in the document
    tds_amount = None
    vat_amount = None
    wht_amount = None

    return {
        "so_codes": [code for code in so_codes if code],
        "billing_codes": [code for code in billing_codes if code],
        "billing_references": [ref for ref in billing_references if ref],
        "tds_amount": tds_amount,
        "vat_amount": vat_amount,
        "wht_amount": wht_amount,
        "paid_amount": paid_amount
    }


def main():
    try:
        # Open and parse the PDF file
        with open("C:\\Users\guowb1\Downloads\\2701964wangyy.pdf", 'rb') as pdf_file:
            document_text = extract_text_from_pdf(pdf_file)

        results = parse_payment_document(document_text)

        print("SO Codes:", results["so_codes"])
        print("Billing Codes:", results["billing_codes"])
        print("Billing References:", results["billing_references"])
        print("TDS Amount:", results["tds_amount"])
        print("VAT Amount:", results["vat_amount"])
        print("WHT Amount:", results["wht_amount"])
        print("Paid Amount:", results["paid_amount"])
    except Exception as e:
        print(f"Error processing document: {e}")


if __name__ == "__main__":
    main()

