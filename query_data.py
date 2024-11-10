# import argparse
# # from dataclasses import dataclass
# from langchain_community.vectorstores import Chroma
# from langchain_openai import OpenAIEmbeddings
# from langchain_openai import ChatOpenAI
# from langchain.prompts import ChatPromptTemplate

# CHROMA_PATH = "chroma"

# PROMPT_TEMPLATE = """
# Answer the question based only on the following context:

# {context}

# ---

# Answer the question based on the above context: {question}
# """


# def main():
#     # Create CLI.
#     parser = argparse.ArgumentParser()
#     parser.add_argument("query_text", type=str, help="The query text.")
#     args = parser.parse_args()
#     query_text = args.query_text

#     # Prepare the DB.
#     embedding_function = OpenAIEmbeddings()
#     db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

#     # Search the DB.
#     results = db.similarity_search_with_relevance_scores(query_text, k=3)
#     if len(results) == 0 or results[0][1] < 0.7:
#         print(f"Unable to find matching results.")
#         return

#     context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
#     prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
#     prompt = prompt_template.format(context=context_text, question=query_text)
#     print(prompt)

#     model = ChatOpenAI()
#     response_text = model.predict(prompt)
    
#     #extract the sources of the retreived documents, formates the final response with both the AI's answer and the sources
#     sources = [doc.metadata.get("source", None) for doc, _score in results]
#     formatted_response = f"Response: {response_text}\nSources: {sources}"
#     print(formatted_response)


# if __name__ == "__main__":
#     main()


import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import re

CHROMA_PATH = "chroma"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "google/flan-t5-base"

PROMPT_TEMPLATE = """
You are an AI assistant for the DHR IT Division. 
Based on the following context about the IT Division of the Department of Human Resources (DHR),answer questions naturally, as a human expert would.

Context:
Department: Department of Human Resources (DHR)
Division: Information Technology Division

{context}

Question: {question}

Important Instructions:
- Understand the intent of the question and provide relevant information from the context
- Look for both direct matches and semantically similar content
- When asked about "it", refer to the subject of the previous response or question
- When asked about quantities or numbers, carefully check the context for specific numerical information
- Respond 'You are welcome' when receives 'thank you'
- Respond approprately when users say 'hi'
- If no relevant information is found, respond with "I don't have enough information to answer that question."

Answer:
"""

# Load the embedding model
try:
    embedding_function = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={'normalize_embeddings': True}
    )
except Exception as e:
    raise Exception(f"Failed to initialize embedding model: {str(e)}")

# Load the Chroma database
try:
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
except Exception as e:
    raise Exception(f"Failed to load Chroma database: {str(e)}")

# Load the LLM model
try:
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL)
except Exception as e:
    raise Exception(f"Failed to initialize LLM model: {str(e)}")

def query_data(query_text):
    # Handle greetings
    if re.search(r'\b(hello|hi|hey)\b', query_text.lower()):
        return "Hello! How can I help you with questions about the DHR IT Division?"
    
    # Handle thank you messages    
    if re.search(r'\b(thank|thanks|thx)\b', query_text.lower()):
        return "You're welcome!"

    try:
        # Normalize query by removing common filler words
        normalized_query = re.sub(r'\b(some|of|the|are|what|do|does)\b', '', query_text.lower())
        normalized_query = re.sub(r'\s+', ' ', normalized_query).strip()
        
        # Use both original and normalized queries for better matching
        results1 = db.similarity_search_with_relevance_scores(query_text, k=3)
        results2 = db.similarity_search_with_relevance_scores(normalized_query, k=1)
        
        # Combine and deduplicate results
        all_results = list({(r[0].page_content, r[1]) for r in results1 + results2})
        all_results.sort(key=lambda x: x[1], reverse=True)
        
        if not all_results:
            return "I don't have enough information to answer that question."

        context_text = "\n".join([result[0] for result in all_results[:3]])
        
        # Add reference resolution for "it"
        if "it" in query_text.lower():
            context_text = "Note: When referring to 'it', consider the Department of Human Resources (DHR) IT Division.\n" + context_text

        prompt = PROMPT_TEMPLATE.format(context=context_text, question=query_text)

        inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        outputs = model.generate(**inputs, max_length=50, num_return_sequences=1, temperature=0.2, do_sample=True)
        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        return response_text
    except Exception as e:
        return f"Error processing query: {str(e)}"

def get_db_info():
    return db._collection.count()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    
    try:
        response = query_data(args.query_text)
        print(response)
    except Exception as e:
        print(f"Error: {str(e)}")