import os
import sys

# --- MAC OS ENVIRONMENT FIX ---
# Prevents OMP: Error #15 crash when using faiss-cpu on Mac
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from dotenv import load_dotenv

# LangChain Core Modules
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# LLM Modules (OpenAI for Part 1, Ollama for Part 2)
from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama

# Import Member A's vectorstore loader helper
from vectorstore_builder import load_vectorstore

# Load API Key from .env file for Part 1
load_dotenv(override=True)

def get_conversation_chain(vectorstore, llm):
    """
    Creates a Retrieval-Augmented Generation (RAG) conversation chain with memory.
    """
    # Initialize memory to store chat history
    memory = ConversationBufferMemory(
        memory_key="chat_history", 
        return_messages=True
    )

    # Create the conversational chain, retrieving the top 4 most relevant chunks
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        ),
        memory=memory
    )
    return conversation_chain

def main():
    """
    Driver function to test the conversation chain and interact with the user.
    """
    print("=== DSCI-560 Lab 9 Chatbot (Member B) ===")
    print("Select Mode:")
    print("[1] Part 1: OpenAI Pipeline (Requires API Key)")
    print("[2] Part 2: Open-Source Pipeline (Requires Ollama & llama3)")
    
    mode = input("\nChoice (1 or 2): ").strip()

    try:
        if mode == "1":
            print("\nInitializing Part 1 (OpenAI)...")
            # Load the FAISS index built with OpenAI embeddings
            vs = load_vectorstore("faiss_index_openai", use_openai=True)
            # Initialize the OpenAI GPT-3.5 model
            llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
            
        elif mode == "2":
            print("\nInitializing Part 2 (Open-Source)...")
            # Load the local FAISS index built with BGE embeddings (No API Key needed)
            vs = load_vectorstore("faiss_index_opensource", use_openai=False)
            # Initialize the local Llama 3 model via Ollama (No API Key needed, works offline)
            print("Connecting to local Llama 3 model via Ollama...")
            llm = Ollama(model="llama3")
            
        else:
            print("Invalid choice. Exiting.")
            sys.exit(1)

        # Build the complete conversation chain
        chat_chain = get_conversation_chain(vs, llm)
        print("\n--- System Ready! (Type 'exit' to quit) ---")

        # Interaction Loop
        while True:
            query = input("\nUser: ")
            if query.lower() == 'exit':
                print("Goodbye!")
                break
            
            print("Bot is thinking...")
            response = chat_chain.invoke({"question": query})
            print(f"Bot: {response['answer']}")

    except Exception as e:
        print(f"\n[SYSTEM ERROR]: {e}")
        if mode == "2":
            print("💡 Tip for Part 2: Make sure you have installed Ollama and run 'ollama run llama3' in your terminal.")
        elif mode == "1":
            print("💡 Tip for Part 1: Check your OPENAI_API_KEY in the .env file.")

if __name__ == "__main__":
    main()