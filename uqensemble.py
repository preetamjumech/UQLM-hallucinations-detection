import asyncio
from langchain_google_genai import ChatGoogleGenerativeAI
from uqlm import UQEnsemble
from dotenv import load_dotenv
import os

load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

async def main():
    llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash')
    uqensemble = UQEnsemble(llm=llm, scorers=["exact_match", "cosine_sim"], use_best=True)

    results = await uqensemble.generate_and_score(
        prompts=["What is the capital of France?"],
    )
    
    print(results.to_df())

# Run the async main function
if __name__ == "__main__":
    asyncio.run(main())