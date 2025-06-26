import asyncio
from langchain_google_genai import ChatGoogleGenerativeAI
from uqlm import LLMPanel
from dotenv import load_dotenv
import os

load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

async def main():
    judge_llm1 = ChatGoogleGenerativeAI(model='gemini-2.0-flash')
    judge_llm2 = ChatGoogleGenerativeAI(model='gemini-2.0-flash')

    llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash')

    llmpanel = LLMPanel(llm = llm , judges = [judge_llm1, judge_llm2])

    results = await llmpanel.generate_and_score(
        prompts=["What is the capital of France?"],
    )
    
    print(results.to_df())

# Run the async main function
if __name__ == "__main__":
    asyncio.run(main())