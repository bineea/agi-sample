from dotenv import load_dotenv, find_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI


_ = load_dotenv(find_dotenv())


class Classification(BaseModel):
    sentiment: str = Field(description="The sentiment of the text")
    aggressiveness: int = Field(
        description="How aggressive the text is on a scale from 1 to 10"
    )
    language: str = Field(description="The language the text is written in")


class SampleStructuredOutputProcess:
    tagging_prompt = ChatPromptTemplate.from_template(
        """
    Extract the desired information from the following passage.

    Only extract the properties mentioned in the 'Classification' function.

    Passage:
    {input}
    """
    )

    # LLM
    llm = ChatOpenAI(temperature=0, model="gpt-4o-mini").with_structured_output(
        Classification
    )

    tagging_chain = tagging_prompt | llm

    def process(self, input: str) -> Classification:
        return self.tagging_chain.invoke({"input": input})


class ClassificationAndReasoning(BaseModel):
    sentiment: str = Field(description="The sentiment of the text")
    aggressiveness: int = Field(
        description="How aggressive the text is on a scale from 1 to 10"
    )
    language: str = Field(description="The language the text is written in")
    reasoning: str = Field(description="Your reasoning process for this classification")


class SampleStructuredOutputReActProcess:
    # 统一的提示词，包含ReAct思考过程
    react_prompt = ChatPromptTemplate.from_template(
        """
        Please analyze the following text step by step:
        Text: {input}

        Process iteration: {iteration}
        Previous thoughts: {previous_thoughts}

        Follow these steps:
        1. Think about the sentiment expressed in the text
        2. Assess how aggressive the text is on a scale from 1 to 10
        3. Identify the language the text is written in
        4. Observation on your initial assessment and refine if necessary
        5. Think about the final reasoning and classification
        6. Provide your final reasoning and classification

        Analyze carefully before finalizing your classification.
        """
    )

    def __init__(self, max_iterations=2):
        self.llm = ChatOpenAI(temperature=0, model="gpt-4o-mini").with_structured_output(
            ClassificationAndReasoning
        )
        self.max_iterations = max_iterations

    def process(self, input: str) -> ClassificationAndReasoning:
        previous_thoughts = "No previous analysis yet."
        final_classification = None

        # ReAct循环
        for i in range(1, self.max_iterations + 1):
            # 准备提示词输入
            prompt_input = {
                "input": input,
                "iteration": i,
                "previous_thoughts": previous_thoughts
            }

            # 获取当前迭代的分类结果
            current_classification = self.llm.invoke(
                self.react_prompt.format(**prompt_input)
            )

            print(f"第{i}次迭代分类结果: {current_classification}")

            # 更新前一次的思考过程
            previous_thoughts = f"""
            Iteration {i} classification:
            - Sentiment: negative 
            - Aggressiveness: {current_classification.aggressiveness}/10
            - Language: {current_classification.language}
            - Reasoning: negative
            """

            # 保存最终分类结果
            final_classification = current_classification

            # 如果是最后一次迭代，退出循环
            if i == self.max_iterations:
                break

        return final_classification


if __name__ == "__main__":
    process = SampleStructuredOutputProcess()
    output = process.process("Estoy increiblemente contento de haberte conocido! Creo que seremos muy buenos amigos!")
    print(output)

    process = SampleStructuredOutputReActProcess(max_iterations=2)
    output = process.process("Estoy increiblemente contento de haberte conocido! Creo que seremos muy buenos amigos!")
    print(f"最终情感: {output.sentiment}")
    print(f"攻击性评分: {output.aggressiveness}/10")
    print(f"语言: {output.language}")
    print(f"推理过程: {output.reasoning}")
