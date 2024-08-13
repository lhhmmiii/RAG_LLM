# How to use few shot examples
'''
- Đầu tiên là tạo example prompt cho 1 example trước
- Tiếp theo là tạo bộ examples tuân theo các biến có trong prompt
- Tiếp theo sử dụng hàm FewShotPromptTemplate, trong đó phải suffix là phần sẽ được ghi ra và input_variables thì để trong []
'''

# import thư viện
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate

examples = [
    {
        "question": "Who lived longer, Muhammad Ali or Alan Turing?",
        "answer": """
Are follow up questions needed here: Yes.
Follow up: How old was Muhammad Ali when he died?
Intermediate answer: Muhammad Ali was 74 years old when he died.
Follow up: How old was Alan Turing when he died?
Intermediate answer: Alan Turing was 41 years old when he died.
So the final answer is: Muhammad Ali
""",
    },
    {
        "question": "When was the founder of craigslist born?",
        "answer": """
Are follow up questions needed here: Yes.
Follow up: Who was the founder of craigslist?
Intermediate answer: Craigslist was founded by Craig Newmark.
Follow up: When was Craig Newmark born?
Intermediate answer: Craig Newmark was born on December 6, 1952.
So the final answer is: December 6, 1952
""",
    },
    {
        "question": "Who was the maternal grandfather of George Washington?",
        "answer": """
Are follow up questions needed here: Yes.
Follow up: Who was the mother of George Washington?
Intermediate answer: The mother of George Washington was Mary Ball Washington.
Follow up: Who was the father of Mary Ball Washington?
Intermediate answer: The father of Mary Ball Washington was Joseph Ball.
So the final answer is: Joseph Ball
""",
    },
    {
        "question": "Are both the directors of Jaws and Casino Royale from the same country?",
        "answer": """
Are follow up questions needed here: Yes.
Follow up: Who is the director of Jaws?
Intermediate Answer: The director of Jaws is Steven Spielberg.
Follow up: Where is Steven Spielberg from?
Intermediate Answer: The United States.
Follow up: Who is the director of Casino Royale?
Intermediate Answer: The director of Casino Royale is Martin Campbell.
Follow up: Where is Martin Campbell from?
Intermediate Answer: New Zealand.
So the final answer is: No
""",
    },
]

dot = '_'.join('' for i in range(100))

example_prompt = PromptTemplate.from_template("Question: {question}\n{answer}") # 1 Q-A
print(example_prompt.invoke(examples[0]).to_string())

## ---------------- Few shot prompt ------------------ ##
prompt2 = FewShotPromptTemplate(
    example_prompt = example_prompt,
    examples = examples,
    suffix="Question: {input}",
    input_variables = ["input"]
)

print(
    prompt2.invoke({"input": "Who was the father of Mary Ball Washington?"}).to_string()
)