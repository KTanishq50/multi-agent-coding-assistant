# from __future__ import annotations
# import requests
# from langchain_ollama import ChatOllama
# from langchain_core.messages import SystemMessage, HumanMessage
# import logging
# from langsmith import traceable

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)


# MODEL = "tinyllama"
# MAX_CONTEXT_CHARS = 1200

# llm = ChatOllama(
#     model=MODEL,
#     temperature=0.12,
#     num_ctx=3072,
#     num_predict=400,
# )

#@traceable(name="llm_call")
# def call_llm(messages: list, temperature: float = None) -> str:
#     """Main function for agents. Takes list of SystemMessage/HumanMessage."""
#     try:
#         if temperature is not None:
#             response = llm.invoke(messages, config={"temperature": temperature})
#         else:
#             response = llm.invoke(messages)
#         return response.content.strip()
#     except Exception as e:
#         logger.error(f"[LLM ERROR] {e}")
#         return f"LLM Error: {str(e)[:80]}"


# def call_llm_raw(prompt: str, max_tokens: int = 50, timeout: int = 20) -> str:
#     """
#     Lightweight raw LLM call for classifiers and short decisions.
#     Uses direct Ollama API instead of LangChain to avoid overhead.
#     Used by: intent classifier, supervisor, query rewriter.
#     """
#     try:
#         payload = {
#             "model": MODEL,
#             "prompt": prompt,
#             "stream": False,
#             "options": {
#                 "num_ctx": 512,
#                 "temperature": 0.1,
#                 "num_predict": max_tokens,
#             },
#         }
#         res = requests.post(
#             "http://localhost:11434/api/generate",
#             json=payload,
#             timeout=timeout
#         )
#         res.raise_for_status()
#         return res.json().get("response", "").strip()
#     except Exception as e:
#         logger.warning(f"[LLM RAW] {e}")
#         return ""


# def generate_answer(context: str, query: str) -> str:
#     """Fallback for when graph fails."""
#     if not context or not context.strip():
#         context = "No relevant code found in the codebase."
#     context = context[:MAX_CONTEXT_CHARS]
#     prompt = (
#         f"You are a helpful code assistant.\n\n"
#         f"Relevant code context:\n{context}\n\n"
#         f"Question: {query}\n\n"
#         f"Answer concisely:"
#     )
#     try:
#         payload = {
#             "model": MODEL,
#             "prompt": prompt,
#             "stream": False,
#             "options": {"num_ctx": 2048, "temperature": 0.1, "num_predict": 300},
#         }
#         res = requests.post("http://localhost:11434/api/generate", json=payload, timeout=60)
#         res.raise_for_status()
#         return res.json().get("response", "").strip()
#     except Exception as e:
#         logger.error(f"[RAW LLM ERROR] {e}")
#         return "Sorry, I could not generate a response."


from __future__ import annotations
import os
import requests
from dotenv import load_dotenv
import logging

from langchain_groq import ChatGroq

from langsmith import traceable

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#  GROQ SETUP
MODEL = "llama-3.1-8b-instant"

llm = ChatGroq(
    model=MODEL,
    temperature=0.1,
    max_tokens=1200,
    api_key=os.getenv("GROQ_API_KEY"),
)

MAX_CONTEXT_CHARS = 2000

@traceable(name="llm_call", run_type="llm")
def call_llm(messages: list, temperature: float = None) -> str:
    """Main function for agents."""
    try:
        if temperature is not None:
            response = llm.invoke(messages, config={"temperature": temperature})
        else:
            response = llm.invoke(messages)
        return response.content.strip()
    except Exception as e:
        logger.error(f"[LLM ERROR] {e}")
        return f"LLM Error: {str(e)[:100]}"


# def call_llm_raw(prompt: str, max_tokens: int = 50, timeout: int = 20) -> str:
#     """For classifiers, supervisor, etc. - Using Groq"""
#     try:
#         # Using the same Groq LLM for simplicity
#         response = llm.invoke(prompt)
#         return response.content.strip()
#     except Exception as e:
#         logger.warning(f"[LLM RAW] {e}")
#         return ""
    
def call_llm_raw(prompt: str, max_tokens: int = 50, timeout: int = 20) -> str:
    """For classifiers, supervisor, short decisions."""
    from langchain_core.messages import HumanMessage
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content.strip()
    except Exception as e:
        logger.warning(f"[LLM RAW] {e}")
        return ""


def generate_answer(context: str, query: str) -> str:
    """Fallback"""
    if not context or not context.strip():
        context = "No relevant code found in the codebase."
    context = context[:MAX_CONTEXT_CHARS]
    
    prompt_text = (
        f"You are a helpful code assistant.\n\n"
        f"Relevant code context:\n{context}\n\n"
        f"Question: {query}\n\n"
        f"Answer concisely:"
    )
    try:
        response = llm.invoke(prompt_text)
        return response.content.strip()
    except Exception as e:
        logger.error(f"[GENERATE ANSWER ERROR] {e}")
        return "Sorry, I could not generate a response."