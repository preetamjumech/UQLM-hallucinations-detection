import asyncio
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import AzureChatOpenAI
from uqlm import WhiteBoxUQ
from dotenv import load_dotenv
import os

load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")
os.environ["AZURE_OPENAI_KEY"] = os.getenv("AZURE_OPENAI_KEY")

async def main():
    llm = AzureChatOpenAI(
        azure_deployment='gpt-4o',
        api_version = "2024-12-01-preview",
        temperature=0.0,
        max_tokens=1000,
    )
    wbuq = WhiteBoxUQ(llm=llm, scorers=["min_probability"])

    results = await wbuq.generate_and_score(
        prompts=["What is the capital of France?"],
    )
    
    print(results.to_df())

# Run the async main function
if __name__ == "__main__":
    asyncio.run(main())