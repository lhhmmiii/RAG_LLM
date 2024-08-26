from langchain_core.tools import tool
from typing import Annotated, List
from langchain.pydantic_v1 import BaseModel, Field
'''
Langchain hỗ trợ việc tạo tool từ function, runnable, và class con của BaseTool
'''

## ----------------------- Creating tools from functions -------------------- ##
dash_line = '_'.join(' ' for i in range(100))
'''
Function name sẽ được sử dụng như là tool name và chuỗi doc của hàm như là mô tả của tool(Vì vậy phải có docstring - phần trong dấu '''''')
'''
@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

print(multiply.name)
print(multiply.description)
print(multiply.args)
print(dash_line)

@tool
async def amultiply(a: int, b: int) -> int: # 
    """Multiply two numbers."""
    return a * b


print(amultiply.name)
print(amultiply.description)
print(amultiply.args)
print(dash_line)

@tool # Thêm mô tả cho biến
def multiply_by_max(
    a: Annotated[str, "scale factor"],
    b: Annotated[List[int], "list of ints over which to take maximum"],
) -> int:
    """Multiply a by the maximum of b."""
    return a * max(b)

print(multiply_by_max.name)
print(multiply_by_max.description)
print(multiply_by_max.args)
print(dash_line)

# Thay đổi schema cho 1 hàm bằng phương pháp sau
class CalculatorInput(BaseModel):
    a: int = Field(description="first number")
    b: int = Field(description="second number")


@tool("multiplication-tool", args_schema=CalculatorInput, return_direct=True)
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

print(multiply.name)
print(multiply.description)
print(multiply.args)
print(multiply.return_direct)
print(dash_line)

## ------------------------- Runnables --------------------------- ##
from langchain_core.language_models import GenericFakeChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages(
    [("human", "Hello. Please respond in the style of {answer_style}.")]
)

# Placeholder LLM
llm = GenericFakeChatModel(messages=iter(["hello matey"]))

chain = prompt | llm | StrOutputParser()

as_tool = chain.as_tool( # Sử dụng as_tool để chuyển runnable sang tool(nhưng nó vẫn đang trong giai đoạn thử nghiệm)
    name="Style responder", description="Description of when to use tool."
)
print(as_tool.args)
print(dash_line)
