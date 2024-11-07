import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnablePassthrough
from langchain.tools import Tool
from langchain.agents import AgentExecutor, create_openai_functions_agent
from dotenv import load_dotenv

load_dotenv()

def load_knowledge_base(directory):
    loader = DirectoryLoader(directory, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    return FAISS.from_documents(texts, embeddings)

# Setup knowledge bases
mb_ageas_retirement = load_knowledge_base("knowledge_base/retirement_plans")
mb_ageas_saving = load_knowledge_base("knowledge_base/savings_plans")
mb_ageas_illness = load_knowledge_base("knowledge_base/illness_plans")
mb_ageas_accident = load_knowledge_base("knowledge_base/accident_plans")
mb_ageas_child = load_knowledge_base("knowledge_base/child_plans")

llm = ChatOpenAI(temperature=0)

def run_qa(query, knowledge_base) -> str:
    query_text = query if isinstance(query, str) else query.get('query', '')
    docs = knowledge_base.similarity_search(query_text, k=10)
    context = "\n".join([doc.page_content for doc in docs])
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an AI assistant specializing in insurance plans.  Provide a comprehensive summary of ALL different insurance plans mentioned in the context. Focus on key features and differences between plans. If only one plan is mentioned, state that clearly."),
        ("human", "Context: {context}\n\nQuery: {query}\n\nProvide a summary of the insurance plans mentioned:"),
    ])
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(context=context, query=query_text)

def run_mb_ageas_retirement_plan_qa(query) -> str:
    return run_qa(query, mb_ageas_retirement)

def run_mb_ageas_saving_plan_qa(query) -> str:
    return run_qa(query, mb_ageas_saving)

def run_mb_ageas_illness_plan_qa(query) -> str:
    return run_qa(query, mb_ageas_illness)

def run_mb_ageas_accident_plan_qa(query) -> str:
    return run_qa(query, mb_ageas_accident)

def run_mb_ageas_child_plan_qa(query) -> str:
    return run_qa(query, mb_ageas_child)

tools = [
    Tool(
        name="MBagaesRetirementPlanQA",
        func=run_mb_ageas_retirement_plan_qa,
        description="Useful for answering questions about MB Ageas retirement insurance plans."
    ),
    Tool(
        name="MBagaesSavingsPlanQA",
        func=run_mb_ageas_saving_plan_qa,
        description="Useful for answering questions about MB Ageas savings insurance plans."
    ),
    Tool(
        name="MBagaesIllnessPlanQA",
        func=run_mb_ageas_illness_plan_qa,
        description="Useful for answering questions about MB Ageas illness insurance plans.."
    ),
    Tool(
        name="MBagaesAccidentPlanQA",
        func=run_mb_ageas_accident_plan_qa,
        description="Useful for answering questions about MB Ageas accident plans."
    ),
    Tool(
        name="MBagaesChildPlanQA",
        func=run_mb_ageas_child_plan_qa,
        description="Useful for answering questions about MB Ageas child plans."
    ),
]

# Update the agent prompt
agent_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an AI assistant specializing in insurance products. "
               "Use the provided tools to answer questions about child and retirement insurance plans. "
               "Always provide a final, concise summary of the plans discussed, even if you've used multiple tools."
               "Do not provide the same response again & again. Always answer in English Language."),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
    ("human", "Summarize the key points about the insurance plans discussed in your response. If you've used multiple tools, consolidate the information into a single, coherent answer.")
])

# Create the agent
agent = create_openai_functions_agent(llm=llm, tools=tools, prompt=agent_prompt)

# Create the agent executor with a max_iterations limit
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=3)

# Combine runnables
product_agent = RunnablePassthrough() | agent_executor

# def test_product_agent():
#     while True:
#         query = input("Enter your question (or 'quit' to exit): ")
#         if query.lower() == 'quit':
#             break
        
#         try:
#             response = product_agent.invoke({"input": query})
#             print("\nProduct Agent Response:")
#             if 'output' in response:
#                 print(response['output'])
#             else:
#                 print("The agent didn't provide a final answer. Here's a summary of the information gathered:")
#                 print(run_qa(query, child_plan_kb))  # Fallback to direct QA if agent fails
#             print("\n" + "-"*50 + "\n")
#         except Exception as e:
#             print(f"An error occurred: {str(e)}")
#             import traceback
#             traceback.print_exc()

# if __name__ == "__main__":
#     test_product_agent()