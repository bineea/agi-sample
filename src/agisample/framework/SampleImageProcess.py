from PIL import Image

# 加载图像
image_path = "C:\\Users\\guowb1\\Pictures\\Screenshots\\屏幕截图 2024-09-23 184625.png"
image = Image.open(image_path)

# 使用 pytesseract 获取每个字符的坐标
boxes = pytesseract.image_to_boxes(image)


from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

# 定义处理过程的模板
template = """解析图片内容，并整理为json格式输出:
{text}"""

# 创建一个 PromptTemplate
prompt = PromptTemplate(input_variables=["text"], template=template)

# 创建一个 OpenAI LLMChain
llm = OpenAI(model="gpt-4o")
chain = LLMChain(llm=llm, prompt=prompt)

# 使用链执行解析操作
result = chain.run(text=boxes)
print(result)
