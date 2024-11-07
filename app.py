import os
import streamlit as st
import streamlit.components.v1 as components
from typing import TypedDict
from datetime import datetime, timedelta
from graph import create_graph
from dotenv import load_dotenv
# from needs_agent import needs_agent
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

# Ensure the OPENAI_API_KEY is set
if not os.getenv('OPENAI_API_KEY'):
    st.error("Please set the OPENAI_API_KEY environment variable.")
    st.stop()

class ChatState(TypedDict):
    input: str
    output: str
    decision: str
    show_contact_form: bool

def inject_custom_css():
    st.markdown("""
        <style>
        /* Main container */
        .main {
            padding: 1rem;
        }
        
        /* Chat interface styling */
        .stChatMessage {
            padding: 1rem;
            margin-bottom: 0.5rem;
            border-radius: 0.5rem;
        }
        
        /* Form styling */
        .stForm {
            background-color: white;
            padding: 1.5rem;
            border-radius: 0.5rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 1rem;
        }
        
        /* Input fields */
        .stTextInput > div > div > input {
            border: 1px solid #ddd;
            border-radius: 0.25rem;
            padding: 0.5rem;
        }
        
        /* Radio button styling */
        div.row-widget.stRadio > div {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
        }
        
        div.row-widget.stRadio > div[role="radiogroup"] > label {
            background-color: white;
            padding: 0.5rem 1rem;
            border: 1px solid #ddd;
            border-radius: 0.25rem;
            cursor: pointer;
            min-width: 80px;
            text-align: center;
        }
        
        div.row-widget.stRadio > div[role="radiogroup"] > label:hover {
            border-color: #1f77b4;
        }
        
        div.row-widget.stRadio > div[role="radiogroup"] > label[data-checked="true"] {
            background-color: #1f77b4;
            color: white;
            border-color: #1f77b4;
        }
        
        /* Button styling */
        .stButton > button {
            background-color: #1f77b4;
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 0.25rem;
            cursor: pointer;
            width: 100%;
            transition: background-color 0.2s ease;
        }
        
        .stButton > button:hover {
            background-color: #1a6698;
        }
        
        /* Sidebar styling */
        .sidebar .sidebar-content {
            background-color: #f8f9fa;
            padding: 1rem;
        }
        
        /* Success message styling */
        .success-message {
            background-color: #d4edda;
            color: #155724;
            padding: 1rem;
            border-radius: 0.25rem;
            margin: 1rem 0;
        }
        </style>
    """, unsafe_allow_html=True)

def initialize_session_state():
    """Initialize all session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I'm your Life Insurance AI Agent. Please tell me how do I address you?"}
        ]
    if "setup_complete" not in st.session_state:
        st.session_state.setup_complete = False
    if "agents" not in st.session_state:
        st.session_state.agents = []
    if "form_submitted" not in st.session_state:
        st.session_state.form_submitted = False
    if "show_contact_form" not in st.session_state:
        st.session_state.show_contact_form = False
    if "appointments" not in st.session_state:
        st.session_state.appointments = []
    if "user_data" not in st.session_state:
        st.session_state.user_data = {}

def recommendation_agent(form_data):
    """
    Generate personalized insurance recommendations based on user form data.
    
    Args:
        form_data (dict): Dictionary containing user form responses
    
    Returns:
        dict: Contains recommendation output and UI control flags
    """
    
    # Sample product information - In production, this should come from a database
    sample_product_info = """
    Life Insurance Products:
    1. "An Tâm Tài Chính" (Financial Peace of Mind)
    - Comprehensive term life insurance with full protection benefits
    - Sum assured up to 30 times annual income
    - Premium options: Monthly, Quarterly, Semi-annual, Annual
    - Entry age: 18-65 years
    
    2. "Phúc Bảo An" (Secure Prosperity)
    - Whole life insurance with savings component
    - Death benefit: 100% sum assured plus accumulated bonuses
    - Premium payment term: 10, 15, 20 years
    - Entry age: 0-65 years
    
    Health Insurance Products:
    1. "Sống Khỏe" (Healthy Living)
    - Comprehensive critical illness coverage 
    - Covers 45 critical illnesses
    - Lump sum payment up to 2 billion VND
    - Premium payment term: 10-20 years
    
    Education Plans:
    1. "Học Vấn Tương Lai" (Future Education)
    - Education plan with protection benefits
    - Guaranteed education fund
    - Flexible premium payment terms
    - Entry age: 0-15 years for children
    """
    
    # Create the recommendation prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an MB Ageas Life insurance specialist. Analyze the customer profile and recommend suitable insurance products. Follow these guidelines:

        1. Focus on the customer's specific life situation and needs
        2. Recommend 2-3 most relevant products from the provided options
        3. Explain why each recommended product suits their situation
        4. Present all information in Vietnamese
        5. Keep explanations clear and concise
        6. End with a call to action to schedule a consultation
        
        Format your response with clear sections:
        - Brief profile summary
        - Product recommendations with explanations
        - Next steps"""),
        
        ("human", """Customer Profile:
        Age: {age}
        Marital status: {marital_status}
        Have children: {has_children}
        Number of children: {num_children}
        
        Available Products:
        {products}
        
        Please provide personalized recommendations in Vietnamese:""")
    ])

    # Create the LLM chain
    chain = LLMChain(
        llm=ChatOpenAI(
            temperature=0.7,
            model="gpt-4",
            max_tokens=2000
        ),
        prompt=prompt
    )

    # Generate recommendations
    recommendations = chain.run(
        age=form_data["age"],
        marital_status="Married" if form_data["is_married"] else "Single",
        has_children="Yes" if form_data["has_children"] else "No",
        num_children=form_data["num_children"],
        products=sample_product_info
    )
    
    # Return recommendations with UI control flags
    return {
        "output": recommendations,
        "show_contact_form": True  # Enable scheduling after recommendations
    }

def get_user_details():
    """Collect initial user details"""
    with st.form("user_details", clear_on_submit=True):
        st.markdown("### Welcome to MB Ageas Insurance")
        
        col1, col2, col3 = st.columns([1, 2, 0.5])
        
        with col1:
            title = st.radio("Title", ["Mr.", "Mrs.", "Miss"], horizontal=True)
        
        with col2:
            name = st.text_input("Name", placeholder="Enter your full name")
        
        with col3:
            submit = st.form_submit_button("→")
        
        if submit and name.strip():
            st.session_state.user_title = title
            st.session_state.user_name = name.strip()
            st.session_state.setup_complete = True
            welcome_message = f"Nice to meet you, {title} {name}! I am an AI insurance agent. I can help you to know more about life insurance & suggest suitable products based on your need. Let me know, how can I assist you today?"
            st.session_state.messages.extend([
                {"role": "user", "content": f"Selected: {title} {name}"},
                {"role": "assistant", "content": welcome_message}
            ])
            st.rerun()

def process_needs_form():
    """Handle the needs assessment form and generate recommendations"""
    with st.form("needs_assessment"):
        st.write("### Assess needs quickly")
        
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.slider("Your age", 18, 100, 30)
            is_married = st.radio("Are you married?", options=["Yes", "No"], horizontal=True) == "Yes"
        
        with col2:
            has_children = st.radio("Do you have children?", options=["Yes", "No"], horizontal=True) == "Yes"
            if has_children:
                num_children = st.number_input("Number of children", 0, 10, 0)
            else:
                num_children = 0

        st.write("### Contact information")
        phone = st.text_input("Phone number", placeholder="Enter your 10-digit phone number")
        email = st.text_input("Email address", placeholder="Enter your email address")

        submitted = st.form_submit_button("Get personalized recommendations")
        
        if submitted:
            if not phone or not phone.isdigit() or len(phone) != 10:
                st.error("Please enter a valid 10-digit phone number.")
                return
            
            if not email or '@' not in email:
                st.error("Please enter a valid email address.")
                return
            
            # Store form data
            form_data = {
                "age": age,
                "is_married": is_married,
                "has_children": has_children,
                "num_children": num_children if has_children else 0,
                "phone": phone,
                "email": email
            }
            
            st.session_state.form_submitted = True
            st.session_state.form_data = form_data
            
            # Generate recommendations
            recommendations = recommendation_agent(form_data)
            
            # Add recommendations to chat history
            st.session_state.messages.extend([
                {"role": "assistant", "content": "Based on your profile, here are my personalized recommendations:"},
                {"role": "assistant", "content": recommendations["output"]}
            ])
            
            # Show contact form if recommended
            if recommendations.get("show_contact_form", False):
                st.session_state.show_contact_form = True
            
            st.session_state.agents.append("recommendation_agent")
            st.rerun()

def render_contact_calendar_form():
    """Render the contact and calendar scheduling form"""
    with st.form("contact_calendar"):
        st.write("### Schedule a consultation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Date selection
            min_date = datetime.now().date()
            max_date = min_date + timedelta(days=30)
            selected_date = st.date_input(
                "Preferred Date",
                min_value=min_date,
                max_value=max_date,
                value=min_date
            )
        
        with col2:
            # Time selection
            time_slots = []
            for hour in range(9, 18):  # 9 AM to 5 PM
                time_slots.extend([f"{hour:02d}:00", f"{hour:02d}:30"])
            
            selected_time = st.selectbox("Preferred Time", time_slots)
        
        st.markdown("### Additional Notes")
        notes = st.text_area("Any specific questions or concerns?", 
                           placeholder="Optional: Enter any specific topics you'd like to discuss...")
        
        col3, col4 = st.columns([3, 1])
        with col4:
            submitted = st.form_submit_button("Schedule Call")
        
        if submitted:
            if selected_date and selected_time:
                appointment = {
                    "date": selected_date.strftime("%Y-%m-%d"),
                    "time": selected_time,
                    "notes": notes
                }
                st.session_state.appointments.append(appointment)
                
                success_msg = f"""
                Thank you for scheduling a consultation!
                \nDate: {selected_date.strftime('%B %d, %Y')}
                \nTime: {selected_time}
                \nOur representative will contact you at the scheduled time.
                """
                st.success(success_msg)
                
                st.session_state.show_contact_form = False
                return True
            else:
                st.error("Please select both date and time for the consultation.")
    return False

def process_user_input(user_input: str):
    """Process user input through the conversation graph"""
    if 'graph' not in st.session_state:
        st.session_state.graph = create_graph()
    
    result = st.session_state.graph.invoke({"input": user_input})
    
    if "show_contact_form" in result and result["show_contact_form"]:
        st.session_state.show_contact_form = True
    
    return result

def display_chat_history():
    """Display the chat history and handle dynamic content"""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            if (message["role"] == "assistant" and 
                st.session_state.show_contact_form and 
                message == st.session_state.messages[-1]):
                render_contact_calendar_form()

def display_agent_sidebar():
    """Display the agent information in the sidebar"""
    if st.session_state.setup_complete:
        st.sidebar.title("Conversation Flow")
        
        # Display current user info
        st.sidebar.markdown("### User Information")
        st.sidebar.text(f"Name: {st.session_state.user_title} {st.session_state.user_name}")
        
        # Display agent history
        st.sidebar.markdown("### Agent History")
        for i, agent in enumerate(st.session_state.agents, 1):
            st.sidebar.text(f"Step {i}: {agent}")

        # Display current agent
        if st.session_state.agents:
            st.sidebar.markdown("### Current Agent")
            st.sidebar.markdown(f"**{st.session_state.agents[-1]}**")

def main():
    st.set_page_config(
        page_title="MB Ageas AI Chatbot",
        page_icon="🤖",
        layout="wide"
    )
    
    # Initialize session state and inject custom CSS
    initialize_session_state()
    inject_custom_css()
    
    st.title("MB Ageas AI Chatbot")
    
    # Create layout
    chat_col, agent_col = st.columns([3, 1])

    with chat_col:
        chat_container = st.container()
        input_container = st.container()

        with chat_container:
            display_chat_history()
            
            if not st.session_state.setup_complete:
                get_user_details()
            
            if (st.session_state.agents and 
                st.session_state.agents[-1] == "needs_agent" and 
                not st.session_state.form_submitted):
                process_needs_form()

        with input_container:
            if st.session_state.setup_complete:
                prompt = st.chat_input(f"Type your message here, {st.session_state.user_name}...")
                
                if prompt:
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    response = process_user_input(prompt)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": response["output"]}
                    )
                    st.session_state.agents.append(response["decision"])
                    st.rerun()

    with agent_col:
        display_agent_sidebar()

if __name__ == "__main__":
    main()