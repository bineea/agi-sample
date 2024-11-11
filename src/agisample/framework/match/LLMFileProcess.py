import base64
from io import BytesIO

import fitz
import os
import time
from mimetypes import guess_type

from PIL import Image
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI, AzureOpenAI

_ = load_dotenv(find_dotenv())

# client = OpenAI(
#     api_key=os.getenv("OPENAI_API_KEY"),
#     base_url=os.getenv("OPENAI_BASE_URL")
# )

client = AzureOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    api_version="2024-05-01-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)


class HandleImgProcess:

    def pdf_to_image(self):
        # 打开 PDF 文件
        doc = fitz.open("E:\document\CASH相关\Remittance文件/AU02-Telstra Limited Remit. Adv 2001293255.pdf")
        # 获取第一页
        page = doc.load_page(0)
        # 提取图像
        pix = page.get_pixmap(dpi=200)

        # 保存图像
        pix.save("page_image.png")

        # 将图像转换为 PIL 图像对象
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        # 将 PIL 图像对象转换为字节流
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        # 将字节流转换为 base64 编码的字符串
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        print(img_str)

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

    def handle(self):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": [
                    {
                        "type": "text",
                        "text": "Describe this picture:"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/jpeg;base64,<your_image_data>"
                        }
                    }
                ]}
            ],
            max_tokens=2000
        )
        print(response)

    pass


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
            content="Search for data similar to 2001293255"
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


if __name__ == '__main__':
    HandleFileProcess().handle_file()