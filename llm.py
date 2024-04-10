from langchain_google_genai import ChatGoogleGenerativeAI

class LLm_Init:
    def __init__(self) -> None:
        self.llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0, convert_system_message_to_human=True)