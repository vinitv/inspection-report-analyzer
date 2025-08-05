
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

def setup_environment():
    """Setup environment variables for LangChain and LangSmith"""
    # Load environment variables from .env file
    load_dotenv()
    
    # Set up LangSmith tracing
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com/"
    os.environ["LANGSMITH_PROJECT"] = "Agentic-Inspection-RAG"
    
    # Check if API keys are set
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  OPENAI_API_KEY not found in .env file")
    if not os.getenv("LANGSMITH_API_KEY"):
        print("⚠️  LANGSMITH_API_KEY not found in .env file")

def main():
    # Setup environment
    setup_environment()
    
    # Create LLM instance
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # Test the LLM with tracing
    print("🚀 Testing LangChain with LangSmith tracing...")
    result = llm.invoke("Hello, world! This is a test of LangSmith tracing.")
    print(f"📄 Response: {result.content}")
    print("🔍 Check LangSmith for traces at: https://smith.langchain.com/")

if __name__ == "__main__":
    main()
