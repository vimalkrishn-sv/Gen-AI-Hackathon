import environ
env = environ.Env()
environ.Env.read_env()
from langchain_google_genai import ChatGoogleGenerativeAI

def get_chat_googlegenai(model_name):
    """
    Returns an instance of the Google Generative AI class initialized with the specified model name.

    Args:
        model_name (str): The name of the model to use.

    Returns:
        ChatGoogleGenerativeAI: An instance of the ChatGoogleGenerativeAI class.

    """
    llm = ChatGoogleGenerativeAI(model=model_name, temperature=0, convert_system_message_to_human=True)
    return llm