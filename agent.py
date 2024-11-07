from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

def router(state):
    llm = ChatOpenAI(model="gpt-4o-mini")
    prompt = PromptTemplate.from_template("""
    You are an intelligent router for a life insurance conversation. Analyze the user's input to determine their primary intent. Consider the following categories:

    1. General Interest (sales_agent): The user is in the early stages of inquiry, seeking general information, or engaging in friendly conversation about life insurance.
    2. Product Information (product_agent): The user is asking about specific insurance products, policy types, or requesting detailed information about coverage options.
    3. Purchase Intent (needs_agent): The user is expressing a clear intent to buy insurance, is ready for a needs assessment, or is asking about the process of buying an insurance product.
    4. Specific Recommendations (recommendation_agent): The user wants personalized product recommendations or is following up on previous recommendations.

    Current User Input: {input}

    Analyze the input carefully, considering both explicit and implicit indications of the user's intent. If unsure, default to the sales_agent for further exploration of the user's needs.

    Respond with only one of these words: "sales_agent", "product_agent", "needs_agent", or "recommendation_agent".

    Your answer:
    """)
    chain = prompt | llm
    response = chain.invoke({"input": state["input"]})
    decision = response.content.strip().lower()
    print(f"Decision made: {decision}")
    return {"decision": decision, "input": state["input"]}

def sales_agent(state):
    print("Using sales agent")
    llm = ChatOpenAI(model="gpt-4o-mini")
    
    prompt = PromptTemplate.from_template(
     """You are a friendly and empathetic life insurance sales agent. Your primary goal is to engage in a warm conversation with the customer and determine if they want to buy life insurance or learn more about it. In 50 words your approach should be:

        1. Build rapport and make the customer feel comfortable. Keep your conversation short.
        2. Listen actively and respond to the customer's comments or questions.
        3. Provide brief, easy-to-understand information about life insurance when appropriate.
        4. Directly, but gently, ask if the customer is interested in purchasing life insurance or if they want to learn more about it.

        Be natural, patient, and avoid being pushy. Your responses should be conversational and always end by asking if the customer wants to buy life insurance or learn more about it.
        Maintain a friendly tone and end by asking if they want to buy life insurance or learn more about it.
        
        User: {input}

        Your response (ask questions warmly and explain their importance):
        """
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.invoke({"input": state["input"]})
    
    return {"output": response["text"]}

def needs_agent(state):
    print("Using needs agent")
    return {
         "output": "Please fill out this quick form to help us understand your needs better:",
        #"output": "Vui lòng điền vào biểu mẫu nhanh này để giúp chúng tôi hiểu rõ hơn nhu cầu của bạn:",
        "show_form": True
    }

# Export the agents
__all__ = ['router', 'sales_agent', 'needs_agent']