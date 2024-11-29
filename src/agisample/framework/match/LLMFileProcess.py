import base64
from io import BytesIO
from pathlib import Path
from pprint import pprint
from typing import Type

import camelot
import cv2
import fitz
import os
import time
from mimetypes import guess_type

import layoutparser
import pdfplumber
import pymupdf4llm
from PIL import Image
from dotenv import load_dotenv, find_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate, \
    SystemMessagePromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from layoutparser.elements import Layout
from layoutparser.models import PaddleDetectionLayoutModel
from layoutparser.visualization import draw_box
from openai import OpenAI, AzureOpenAI
from paddlenlp import Taskflow
from paddleocr import PaddleOCR, PPStructure
from paddleocr.ppstructure.predict_system import save_structure_res
from paddleocr.ppstructure.recovery.recovery_to_doc import sorted_layout_boxes
from unstructured.partition.auto import partition
from unstructured.partition.pdf import partition_pdf

from agisample.framework.match.recovery_to_markdown import convert_info_markdown

_ = load_dotenv(find_dotenv())

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)


# client = AzureOpenAI(
#     api_key=os.getenv("OPENAI_API_KEY"),
#     api_version="2024-05-01-preview",
#     azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
# )


class HandleImgProcess:

    def pdf_to_image(self) -> str:
        # 打开 PDF 文件
        doc = fitz.open(os.path.join(Path(__file__).resolve().parents[4], "docs", "MY01-2701964.pdf"))
        # 获取第一页
        page = doc.load_page(0)
        # 提取图像
        pix = page.get_pixmap(dpi=200)

        # 保存图像
        pix.save("E:\document\CASH相关\Remittance文件\page_image.png")

        # 将图像转换为 PIL 图像对象
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        # 将 PIL 图像对象转换为字节流
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        # 将字节流转换为 base64 编码的字符串
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        # print(img_str)
        return img_str

    # Function to encode a local image into data URL
    def local_image_to_data_url(image_path):
        # Guess the MIME type of the image based on the file extension
        mime_type, _ = guess_type(image_path)
        if mime_type is None:
            mime_type = 'application/octet-stream'  # Default MIME type if none is found

        # Read and encode the image file
        with open(image_path, "rb") as image_file:
            base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')

        # Construct the data URL
        return f"data:{mime_type};base64,{base64_encoded_data}"

    def read_image(self, image_path: str):
        if image_path is None or len(image_path.strip()) == 0:
            image_path = os.path.join(Path(__file__).resolve().parents[4], "docs", "page_image.jpg")
        image = cv2.imread(image_path)
        image = image[..., ::-1]
        print(layoutparser.is_paddle_available())
        # model = PaddleDetectionLayoutModel('lp://PubLayNet/ppyolov2_r50vd_dcn_365e/config',
        #                                    extra_config={"threshold": 0.3},
        #                                    device='cpu',
        #                                    label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"})
        model = PaddleDetectionLayoutModel('lp://TableBank/ppyolov2_r50vd_dcn_365e/config',
                                           extra_config={"threshold": 0.2},
                                           device='cpu',
                                           label_map={0: "Table"})
        layout = model.detect(image)
        # text_blocks = Layout([b for b in layout if b.type == 'Text'])
        # for txt in text_blocks.get_texts():
        #     print(txt, end='\n***\n')
        table_blocks = Layout([b for b in layout if b.type == 'Table'])
        for txt in table_blocks.get_texts():
            print(txt, end='\n---\n')
        draw_box(image, layout, box_width=3).show()

        paddleocr = PaddleOCR(lang='en', show_log=True)
        img = cv2.imread(image_path)  # 打开需要识别的图片
        result = paddleocr.ocr(img)
        for i in range(len(result[0])):
            print(result[0][i][1][0])  # 输出识别结果

        table_engine = PPStructure(recovery=True, lang='en')

        save_folder = os.path.join(Path(__file__).resolve().parents[4], "docs", "")
        img = cv2.imread(image_path)
        result = table_engine(img)
        save_structure_res(result, save_folder, os.path.basename(image_path).split('.')[0])

        for line in result:
            line.pop('img')
            print(line)

        h, w, _ = img.shape
        res = sorted_layout_boxes(result, w)
        convert_info_markdown(res, save_folder, os.path.basename(image_path).split('.'))

        schema = ["document","your document","date","gross amount","currency","reference number"]
        task_flow = Taskflow("information_extraction", schema=schema, model="uie-x-base")
        # task_flow = Taskflow("information_extraction", schema=schema, model="uie-x-base", layout_analysis=True)
        pprint(task_flow({"doc": image_path}))

    def handle(self, image_data):
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": [
                    {
                        "type": "text",
                        "text": "Search for data similar to 2001293255 in this picture:"
                        # "text": "Describe this picture:"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/jpeg;base64,"+image_data
                        }
                    }
                ]}
            ],
            max_tokens=16384
        )
        print(response)


class HandleFileProcess:

    def handle_file(self):
        # https://platform.openai.com/docs/assistants/tools/file-search
        # client = OpenAI()
        # Create a vector store caled "Financial Statements"
        # vector_store = client.beta.vector_stores.create(name="Financial Statements")

        # https://learn.microsoft.com/zh-cn/azure/ai-services/openai/how-to/file-search?tabs=rest
        # Create a vector store called "Remittance Data"
        vector_store = client.beta.vector_stores.create(name="Remittance Data")

        # Ready the files for upload to OpenAI
        file_paths = ["E:\document\CASH相关\Remittance文件/AU02-Telstra Limited Remit. Adv 2001293255.pdf",
                      "E:\document\CASH相关\Remittance文件.MY01-2701964.pdf"]
        file_streams = [open(path, "rb") for path in file_paths]

        # Use the upload and poll SDK helper to upload the files, add them to the vector store,
        # and poll the status of the file batch for completion.
        file_batch = client.beta.vector_stores.file_batches.upload_and_poll(
            vector_store_id=vector_store.id, files=file_streams
        )

        # You can print the status and the file counts of the batch to see the result of this operation.
        print(file_batch.status)
        print(file_batch.file_counts)

        assistant = client.beta.assistants.create(
            name="Financial Analyst Assistant",
            instructions="You are an expert financial analyst. Use your knowledge base to answer questions about audited financial remittance.",
            model="gpt-4o-mini",
            tools=[{"type": "file_search"}],
            tool_resources={"file_search": {"vector_store_ids": [vector_store.id]}},

        )

        print(assistant.model_dump_json(indent=2))

        # 线程实质上是助手和用户之间对话会话的记录
        thread = client.beta.threads.create()
        print(thread)

        message = client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content="Search for data similar to 2001293255 in the image"
        )

        # 运行线程
        run = client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=assistant.id,
        )

        # Retrieve the status of the run
        run = client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id
        )

        status = run.status
        print(status)

        start_time = time.time()
        while status not in ["completed", "cancelled", "expired", "failed"]:
            time.sleep(5)
            run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
            print("Elapsed time: {} minutes {} seconds".format(int((time.time() - start_time) // 60),
                                                               int((time.time() - start_time) % 60)))
            status = run.status
            print(f'Status: {status}')

        messages = client.beta.threads.messages.list(
            thread_id=thread.id
        )

        print(messages.model_dump_json(indent=2))


class RemittanceDataVectorManager:
    BASE_DIR = Path(__file__).resolve().parents[3]
    INDEX_NAME = "remittance_index"
    DB_LOCAL_FILE_NAME = "remittance_db"

    def __init__(self):
        self.__embedding = OpenAIEmbeddings()

    def save(self, documents=Type[list[Document]]):
        faiss_client = FAISS.from_documents(documents, self.__embedding)
        faiss_client.save_local(
            os.path.join(RemittanceDataVectorManager.BASE_DIR, "data", RemittanceDataVectorManager.DB_LOCAL_FILE_NAME),
            RemittanceDataVectorManager.INDEX_NAME)

    def retriever(self):
        db_local_file_path = os.path.join(RemittanceDataVectorManager.BASE_DIR, "data",
                                          RemittanceDataVectorManager.DB_LOCAL_FILE_NAME)
        faiss_client = FAISS.load_local(db_local_file_path, self.__embedding, RemittanceDataVectorManager.INDEX_NAME,
                                        allow_dangerous_deserialization=True)
        return faiss_client.as_retriever()

    def search(self, query):
        docs = self.retriever().get_relevant_documents(query)
        return docs[0].page_content


class HandleFileVectorStoreProcess:
    def init_data_by_pymupdf(self):
        pdf_loader = PyMuPDFLoader(os.path.join(Path(__file__).resolve().parents[4], "docs", "MY01-2701964.pdf"))
        pages = pdf_loader.load_and_split()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len,
            add_start_index=True,
        )
        split_docs = text_splitter.create_documents(
            [page.page_content for page in pages[:10]]
        )

        print([page.page_content for page in pages[:10]])
        return pages[0].page_content
        # print(split_docs)
        # RemittanceDataVectorManager().save(split_docs)

    # 使用pymupdf4llm的to_markdown方法，因为没有保留制表符，可能会导致LLM理解错误
    def init_data_by_pymupdf4llm(self):
        pages = pymupdf4llm.to_markdown(os.path.join(Path(__file__).resolve().parents[4], "docs", "Payment Summary.PDF"), margins=(0, 0, 0, 0), page_chunks=True, pages=[0])
        all_page_content = []
        for page in pages:
            page_metadata = page["metadata"]
            page_content = page["text"]
            doc = Document(page_content=page_content, metadata={"file_name": page_metadata["file_path"], "page_count": page_metadata["page_count"], "page_number": page_metadata["page"]})
            print(doc)
            all_page_content.append(doc)
            # RemittanceDataVectorManager().save(doc)
        return all_page_content

    def init_data_by_pdfplumber(self):
        with pdfplumber.open(os.path.join(Path(__file__).resolve().parents[4], "docs", "MY01-2701964.pdf")) as pdf:
            page = pdf.pages[0]

            # 设置表格提取参数
            table_settings = {
                "vertical_strategy": "text",  # 基于文本对齐检测垂直线
                "horizontal_strategy": "text",  # 基于文本对齐检测水平线
                "intersection_y_tolerance": 10,  # 调整行间距容差
                "join_tolerance": 10,  # 调整列合并容差
                "edge_min_length": 3,  # 最小边缘长度
                "min_words_vertical": 10,  # 最小垂直文字数
                "text_y_tolerance": 2,  # 降低到2
                "text_x_tolerance": 2,  # 添加水平文本容差
                "snap_tolerance": 2,  # 添加对齐容差
            }
            # 提取表格
            # table = page.extract_table(table_settings)
            print(page.extract_text())
            table = page.extract_table()
            if not table:
                return None
            result_table = []
            for row in table:
                all_cell_is_blank = True
                for cell in row:
                    if len(cell.strip()) > 0:
                        all_cell_is_blank = False
                        break

                if all_cell_is_blank == False:
                    result_table.append(row)
            print(result_table)
            # if table:
            #     df = pd.DataFrame(table[1:], columns=table[0])
            #     # 合并跨行的Your document列
            #     df['Your document'] = df['Your document'].apply(
            #         lambda x: ' '.join(x.split()) if pd.notnull(x) else ''
            #     )
            #     return df
            return result_table

    def init_data_by_camelot(self):
        tables = camelot.read_pdf(
            os.path.join(Path(__file__).resolve().parents[4], "docs", "MY01-2701964.pdf"),
            pages=str(1),
            flavor='stream',  # stream模式更适合没有边框的表格
            split_text=True,  # 处理跨行文本
            row_tol=10  # 调整行容差
        )
        for table in tables:
            print(table.df)

    def init_data_by_unstructured(self):
        file_path = os.path.join(Path(__file__).resolve().parents[4], "docs", "MY01-2701964.pdf")
        elements = partition(file_path)
        # elements = partition_pdf(file_path)
        print("\n\n".join([str(el) for el in elements]))

    # def init_data_by_marker

class HandleFileAssistant:
    HANDLE_FILE_ASSISTANT_PROMPT = ChatPromptTemplate.from_messages(
        [
            (
                "system"
                """
                你是客户付款数据解析和理解的专业助手。
                解析并理解客户付款数据: {payment_content}，整理为json格式。
                
                注意事项:
                - 如果某一列数据因长度超长导致换行，请将换行部分合并到同一行数据。
                - 列数据的完整性由前后逻辑判断。
                - 避免编造或猜测任何信息。
                - 避免任何数据计算。
                - 只返回最终结果，不需要返回中间过程。
                
                示例：
                示例1：
                客户付款数据:  ['DocType', 'Docu', 'ment', 'Yourdocument', 'Date', 'Grossa', 'mount'], ['RE', '62047', '80', 'H121006016', '14.08.2024', '2,095.10', 'MYR'], ['', '', '', '8400109550', '', '', '']
                助手:
                {{
                    "total_payment": ,
                    "payment_reference_number": ,
                    "related_order_reference_number": "H1210060168400109550"
                }}
                """
            )
        ]
    )

    # HANDLE_FILE_ASSISTANT_PROMPT = ChatPromptTemplate.from_messages(
    #     [
    #         # (
    #         #     "system",
    #         #     """
    #         #     你是客户付款数据解析和理解的专业助手。
    #         #     解析并理解客户付款数据: {content}，将付款参考编号(payment_reference_number)和付款总金额(total_payment)和付款关联的订单参考编号(related_order_reference_number)整理为json格式。
    #         #     每个单子的付款总金额
    #         #
    #         #     注意事项:
    #         #     - 避免编造或猜测任何信息。
    #         #     - 避免任何数据计算。
    #         #     - 只返回最终结果，不需要返回中间过程。
    #         #     """,
    #         # ),
    #         # ("placeholder", "{messages}"),
    #
    #         (
    #             "system",
    #             """
    #             你是客户付款数据解析和理解的专业助手，根据客户付款数据提取指定的字段（例如“付款数据总金额(total_payment_amount)”、“付款参考编号(payment_reference_number)”等）以及对应的数值。如果没有明确对应的字段，不能根据其他数据推测或计算。
    #             处理步骤:
    #             1. 解析并理解客户付款数据: {content}；
    #             2. 部分内容因长度超出而换行，请将因为换行被拆分的内容合并为完整的值；
    #             3. 列数据的完整性由前后逻辑判断；
    #             2. 解析与付款数据总金额(total_payment_amount)匹配程度超过60%的数据，否则银行付款数据总金额(total_payment_amount)对应数据设置为空；
    #             3. 解析与付款参考编号(payment_reference_number)匹配程度超过60%的数据，否则银行付款数据总金额(payment_reference_number)对应数据设置为空；
    #             4. 解析与关联订单参考编号(related_order_reference_number)匹配程度超过60%的数据，否则关联订单参考编号(related_order_reference_number)对应数据设置为空；
    #             4. 解析与关联订单其他编号(related_order_other_number)匹配程度超过60%的数据，否则关联订单参考编号(related_order_other_number)对应数据设置为空；
    #             5. 解析与关联订单订单金额(related_order_amount)，匹配程度超过60%的数据，否则关联订单订单金额(related_order_amount)对应数据设置为空；
    #             6. 解析与关联订单支付金额(related_order_payment_amount)匹配程度超过60%的数据，否则关联订单支付金额(related_order_payment_amount)对应数据设置为空；
    #             7. 解析与关联订单扣税金额(related_order_tax_amount)匹配程度超过60%的数据，否则关联订单扣税金额(related_order_tax_amount)对应数据设置为空；
    #             7. 解析所有可能的参考编号(possible_reference_number)；
    #             8. 将付款数据总金额(total_payment_amount)、关联订单参考编号(related_order_reference_number)、关联订单订单金额(related_order_amount)、关联订单支付金额(related_order_payment_amount)、关联订单扣税金额(related_order_tax_amount)、参考编号(possible_reference_number)的数据整理并格式化为json格式。
    #
    #             注意事项:
    #             - 部分内容因长度超出而换行，请将因为换行被拆分的内容合并为完整的值，
    #             - 如果内容属于同一列或行，请保持它们的上下文一致性，
    #             - 严格禁止编造或猜测任何数据和信息，
    #             - 严格禁止任何数学运算，
    #             - 严格禁止任何数据推测，
    #             请始终遵循以上指引。
    #
    #             示例：
    #             示例1：
    #             客户付款数据:
    #             助手:
    #             {{
    #                 "total_payment_amount": 100,
    #                 "payment_reference_number": "payment_reference_number",
    #                 "related_orders": [
    #                     {{
    #                         "related_order_reference_number": "related_order_reference_number",
    #                         "related_order_other_number": "related_order_other_number",
    #                         "related_order_amount": 50,
    #                         "related_order_payment_amount": 40,
    #                         "related_order_tax_amount": 10,
    #                     }}
    #                 ],
    #                 "possible_reference_number": ["123","abc"]
    #             }}
    #             """,
    #         ),
    #     ]
    # )

    HANDLE_FILE_WITH_IMAGE_ASSISTANT_PROMPT = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                """
                你是一个专业表格图片解析助手
                1. 如果某一列数据因长度超长导致换行，请将换行部分合并到同一行数据，
                2. 列数据的完整性由前后逻辑判断，比如在“Your document”列中，H121006016 和 8400109550 应合并为同一行，
                3. 表格应包含以下列：Doc Type、Document、Your document、Date、Gross Amount、Currency。
                """
            ),
            HumanMessagePromptTemplate.from_template(
                template = [
                    {
                        "type": "text",
                        "text": "根据图片信息，将单元格中因为长度超出导致换行被拆分的内容合并为完整的值，重新整理为表格形式输出，图片: "
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/jpeg;base64,{payment_image}"
                        }
                    }
                ]
            )
        ]
    )

    # HANDLE_FILE_WITH_IMAGE_ASSISTANT_PROMPT = ChatPromptTemplate.from_messages(
    #     [
    #         SystemMessage(
    #             content=
    #             """
    #             你是客户付款数据解析和理解的专业助手，根据客户付款数据提取指定的字段（例如“付款数据总金额(total_payment_amount)”、“付款参考编号(payment_reference_number)”等）以及对应的数值。如果没有明确对应的字段，不能根据其他数据推测或计算。
    #
    #             处理步骤:
    #             1. 解析并理解客户付款数据；
    #             2. 部分内容因长度超出而换行，请将因为换行被拆分的内容合并为完整的值；
    #             2. 解析与付款数据总金额(total_payment_amount)匹配程度超过60%的数据，否则银行付款数据总金额(total_payment_amount)对应数据设置为空；
    #             3. 解析与付款参考编号(payment_reference_number)匹配程度超过60%的数据，否则银行付款数据总金额(payment_reference_number)对应数据设置为空；
    #             4. 解析与关联订单参考编号(related_order_reference_number)匹配程度超过60%的数据，否则关联订单参考编号(related_order_reference_number)对应数据设置为空；
    #             4. 解析与关联订单其他编号(related_order_other_number)匹配程度超过60%的数据，否则关联订单参考编号(related_order_other_number)对应数据设置为空；
    #             5. 解析与关联订单订单金额(related_order_amount)，匹配程度超过60%的数据，否则关联订单订单金额(related_order_amount)对应数据设置为空；
    #             6. 解析与关联订单支付金额(related_order_payment_amount)匹配程度超过60%的数据，否则关联订单支付金额(related_order_payment_amount)对应数据设置为空；
    #             7. 解析与关联订单扣税金额(related_order_tax_amount)匹配程度超过60%的数据，否则关联订单扣税金额(related_order_tax_amount)对应数据设置为空；
    #             7. 解析所有可能的参考编号(possible_reference_number)；
    #             8. 将付款数据总金额(total_payment_amount)、关联订单参考编号(related_order_reference_number)、关联订单订单金额(related_order_amount)、关联订单支付金额(related_order_payment_amount)、关联订单扣税金额(related_order_tax_amount)、参考编号(possible_reference_number)的数据整理并格式化为json格式。
    #
    #             注意事项:
    #             - 部分内容因长度超出而换行，请将因为换行被拆分的内容合并为完整的值，
    #             - 如果内容属于同一列或行，请保持它们的上下文一致性，
    #             - 严格禁止编造或猜测任何数据和信息，
    #             - 严格禁止任何数学运算，
    #             - 严格禁止任何数据推测，
    #             请始终遵循以上指引。
    #
    #             示例：
    #             示例1：
    #             助手:
    #             {{
    #                 "total_payment_amount": 100,
    #                 "payment_reference_number": "payment_reference_number",
    #                 "related_orders": [
    #                     {{
    #                         "related_order_reference_number": "related_order_reference_number",
    #                         "related_order_other_number": "related_order_other_number",
    #                         "related_order_amount": 50,
    #                         "related_order_payment_amount": 40,
    #                         "related_order_tax_amount": 10,
    #                     }}
    #                 ],
    #                 "possible_reference_number": ["123","abc"]
    #             }}
    #             """
    #         ),
    #         HumanMessagePromptTemplate.from_template(
    #             template = [
    #                 {
    #                     "type": "text",
    #                     "text": "客户付款数据: {payment_content}；根据图片信息，将因为长度超出导致换行被拆分的内容合并为完整的值，图片: "
    #                 },
    #                 {
    #                     "type": "image_url",
    #                     "image_url": {
    #                         "url": "data:image/jpeg;base64,{payment_image}"
    #                     }
    #                 }
    #             ]
    #         )
    #     ]
    # )

    def invoke(self, content: str):
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)
        runnable = HandleFileAssistant.HANDLE_FILE_ASSISTANT_PROMPT | llm
        result = runnable.invoke(
            {
                "payment_content": content
            }
        )
        print(result)

    def invoke_with_image(self, content: str, image: str):
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)
        runnable = HandleFileAssistant.HANDLE_FILE_WITH_IMAGE_ASSISTANT_PROMPT | llm
        result = runnable.invoke(
            {
                "payment_content": content,
                "payment_image": image
            }
        )
        print(result)


if __name__ == '__main__':
    # 图像处理
    # image_data = HandleImgProcess().pdf_to_image()
    HandleImgProcess().read_image(None)
    # print(HandleImgProcess().handle(image_data))

    # all_page_content = HandleFileVectorStoreProcess().init_data_by_pymupdf()
    # print(HandleFileAssistant().invoke_with_image(all_page_content, image_data))

    # all_page_content = HandleFileVectorStoreProcess().init_data_by_pymupdf4llm()
    # print(HandleFileAssistant().invoke(all_page_content[0].page_content))

    # all_page_content = HandleFileVectorStoreProcess().init_data_by_pdfplumber()
    # print(HandleFileAssistant().invoke(all_page_content))

    # HandleFileVectorStoreProcess().init_data_by_camelot()

    # HandleFileVectorStoreProcess().init_data_by_unstructured()



