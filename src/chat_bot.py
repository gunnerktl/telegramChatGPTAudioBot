from langchain import OpenAI, PromptTemplate, ConversationChain
from langchain.memory import ConversationBufferMemory

from src.config import config

template = """The following is a friendly conversation between a human and an AI. The AI is talkative and provides details in couple of sentences at most. If the AI does not know the answer to a question, it truthfully says it does not know.

Current conversation:
{history}
Friend: {input}
AI:"""

llm = OpenAI(temperature=0, openai_api_key=config.openai_api_key)

PROMPT = PromptTemplate(input_variables=["history", "input"], template=template)
conversation = ConversationChain(
    prompt=PROMPT,
    llm=llm,
    verbose=True,
    memory=ConversationBufferMemory(human_prefix="Friend"),
)


def generate_text_response(input_text: str) -> str:
    return conversation.predict(input=input_text)
