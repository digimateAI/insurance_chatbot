def needs_agent(state):
    import pandas as pd
    import json
    from langchain_openai import ChatOpenAI
    from langchain.chains import LLMChain
    from langchain.prompts import ChatPromptTemplate
    
    def process_multiselect_response(response):
        """Convert string representation of list back to list if needed"""
        if isinstance(response, str) and response.startswith('['):
            try:
                return json.loads(response)
            except:
                return response
        return response

    questions = [
        {
            "question": "How old are you?",
            "type": "number",
            "key": "Age",
            "min_value": 18,
            "max_value": 100,
            "step": 1
        },
        {
            "question": "What is your marital status?",
            "type": "radio",
            "options": ["Single", "Married"],
            "key": "MaritalStatus"
        },
        {
            "question": "Do you have children?",
            "type": "radio",
            "options": ["Yes", "No"],
            "key": "HasChildren"
        },
        {
            "question": "What is your monthly income range?",
            "type": "select",
            "options": [
                "Less than 10 million VND",
                "10-20 million VND",
                "20-50 million VND",
                "Above 50 million VND"
            ],
            "key": "Income"
        },
        {
            "question": "What is your preferred premium payment method?",
            "type": "radio",
            "options": ["One-time payment", "Regular payments"],
            "key": "PaymentPreference"
        },
        {
            "question": "What are your primary insurance needs?",
            "type": "multiselect",
            "options": [
                "Basic life protection",
                "Savings and investment",
                "Children's education fund",
                "Health protection",
                "Accident protection",
                "Critical illness coverage",
                "Family income protection"
            ],
            "key": "InsuranceNeeds"
        },
        {
            "question": "Do you have any specific health concerns?",
            "type": "multiselect",
            "options": [
                "Cancer risks",
                "Critical illnesses",
                "Hospital and surgery expenses",
                "No specific concerns"
            ],
            "key": "HealthConcerns"
        }
    ]

    # Initialize or get current state
    needs_step = state.get("needs_step", 0)
    needs_responses = state.get("needs_responses", {})
    welcome_message = "As you want to buy insurance, please answer a few questions so that I can suggest plans suited for you." if needs_step == 0 else ""
    
    # If recommendations already generated, return them
    if state.get("recommendations_generated"):
        return {
            "output": state.get("recommendations", "Assessment complete."),
            "decision": "needs_agent",
            "needs_complete": True
        }

    # Process the previous answer if exists
    if state.get("input") and state.get("input") != "NEEDS_AGENT_START":
        try:
            # Process input based on question type
            if questions[needs_step]["type"] == "multiselect":
                needs_responses[questions[needs_step]["key"]] = process_multiselect_response(state["input"])
            else:
                needs_responses[questions[needs_step]["key"]] = state["input"]
            
            print(f"Processed answer for {questions[needs_step]['key']}: {needs_responses[questions[needs_step]['key']]}")  # Debug print
            needs_step += 1
            
            # If all questions answered, generate recommendations
            if needs_step >= len(questions):
                try:
                    # Save responses as JSON
                    with open('insurance_responses.json', 'a') as f:
                        f.write(json.dumps(needs_responses, ensure_ascii=False) + '\n')
                    
                    print("Saved responses:", needs_responses)  # Debug print
                    
                    sample_product_info = """
                    Life Insurance Products:
                    1. "An Tâm Tài Chính" (Financial Peace of Mind)
                    - Bảo hiểm tử kỳ với quyền lợi bảo vệ toàn diện
                    - Sum assured up to 30 times annual income
                    - Premium options: Monthly, Quarterly, Semi-annual, Annual
                    - Entry age: 18-65 years
                    
                    2. "Phúc Bảo An" (Secure Prosperity)
                    - Bảo hiểm trọn đời với tích lũy
                    - Death benefit: 100% sum assured plus accumulated bonuses
                    - Premium payment term: 10, 15, 20 years
                    - Entry age: 0-65 years
                    
                    Health Insurance Products:
                    1. "Sống Khỏe" (Healthy Living)
                    - Bảo hiểm bệnh hiểm nghèo toàn diện
                    - Covers 45 critical illnesses
                    - Lump sum payment up to 2 billion VND
                    - Premium payment term: 10-20 years
                    
                    Education Plans:
                    1. "Học Vấn Tương Lai" (Future Education)
                    - Kế hoạch giáo dục với quyền lợi bảo vệ
                    - Guaranteed education fund
                    - Flexible premium payment terms
                    - Entry age: 0-15 years
                    """
                    
                    # Generate recommendations using LLM
                    prompt = ChatPromptTemplate.from_messages([
                        ("system", """You are an MB Ageas Life insurance specialist. Your task is to analyze the customer profile and recommend suitable insurance products. Please:
                        1. Focus on the customer's specific needs and circumstances
                        2. Recommend relevant products from the provided product information
                        3. Explain why each product is suitable for their situation
                        4. Present all information in Vietnamese language
                        5. Keep explanations clear and concise"""),
                        ("human", """Customer Profile:
                        Tuổi: {Age}
                        Tình trạng hôn nhân: {MaritalStatus}
                        Có con: {HasChildren}
                        Thu nhập: {Income}
                        Phương thức đóng phí: {PaymentPreference}
                        Nhu cầu bảo hiểm: {InsuranceNeeds}
                        Các vấn đề sức khỏe quan tâm: {HealthConcerns}
                        
                        Product Information:
                        {context}
                        
                        Vui lòng đề xuất các sản phẩm bảo hiểm phù hợp và giải thích lý do lựa chọn:""")
                    ])

                    chain = LLMChain(
                        llm=ChatOpenAI(
                            temperature=0,
                            model="gpt-4",
                            max_tokens=3000
                        ),
                        prompt=prompt
                    )

                    print("Generating recommendations with:", needs_responses)  # Debug print
                    recommendations = chain.run(
                        Age=needs_responses["Age"],
                        MaritalStatus=needs_responses["MaritalStatus"],
                        HasChildren=needs_responses["HasChildren"],
                        Income=needs_responses["Income"],
                        PaymentPreference=needs_responses["PaymentPreference"],
                        InsuranceNeeds=needs_responses["InsuranceNeeds"],
                        HealthConcerns=needs_responses["HealthConcerns"],
                        context=sample_product_info
                    )
                    
                    return {
                        "output": recommendations,
                        "decision": "needs_agent",
                        "needs_complete": True,
                        "recommendations_generated": True,
                        "recommendations": recommendations
                    }
                    
                except Exception as e:
                    print(f"Error in recommendation generation: {str(e)}")  # Debug print
                    return {
                        "output": f"Xin lỗi, đã có lỗi xảy ra trong quá trình tạo đề xuất: {str(e)}. Vui lòng thử lại sau.",
                        "decision": "needs_agent",
                        "needs_complete": True
                    }
        except Exception as e:
            print(f"Error processing answer: {str(e)}")  # Debug print
            return {
                "output": f"Error processing answer: {str(e)}",
                "decision": "needs_agent",
                "needs_step": needs_step
            }

    # Get current question
    current_question = questions[needs_step]
    
    if not state.get("recommendations_generated"):
        current_question = questions[needs_step]
        return {
            "output": welcome_message,
            "decision": "needs_agent",
            "needs_step": needs_step,
            "needs_responses": needs_responses,
            "current_question": current_question,
            "progress": {
                "current": needs_step + 1,
                "total": len(questions)
            }
        }