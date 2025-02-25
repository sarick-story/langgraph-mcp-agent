import streamlit as st
import asyncio
import uuid
from langchain_core.messages import HumanMessage
from langgraph.types import Command
import os
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper
from typing import Annotated, TypedDict
from langchain_core.messages import (
    BaseMessage,
    AIMessage,
    ToolMessage,
    SystemMessage,
)
from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt
from loguru import logger
from langsmith import Client, traceable
from langchain_core.runnables import RunnableLambda
import time
import random

# Import the necessary functions from agent.py
from agent import create_graph, run_agent

# Load environment variables from .env file
load_dotenv()

# Set page config
st.set_page_config(
    page_title="Story IP Creator",
    page_icon="🎨",
    layout="wide",
)

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "processing" not in st.session_state:
    st.session_state.processing = False

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "graph" not in st.session_state:
    st.session_state.graph = None

if "interrupt_data" not in st.session_state:
    st.session_state.interrupt_data = None

if "waiting_for_input" not in st.session_state:
    st.session_state.waiting_for_input = False

# Display header
st.title("🎨 Story IP Creator")
st.markdown("""
This tool helps you create and mint an image as an IP asset in the Story ecosystem.
Generate an image, upload it to IPFS, and mint it as an NFT with customizable license terms.
""")

# Sidebar for settings
with st.sidebar:
    st.header("Settings")
    
    if st.button("Reset Conversation"):
        st.session_state.messages = []
        st.session_state.processing = False
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.graph = None
        st.session_state.interrupt_data = None
        st.session_state.waiting_for_input = False
        st.rerun()

# Display chat messages
for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message("user"):
            st.markdown(message["content"])
    elif message["role"] == "assistant":
        with st.chat_message("assistant"):
            if "image_url" in message:
                st.image(message["image_url"], caption="Generated Image")
            st.markdown(message["content"])
    elif message["role"] == "system":
        with st.chat_message("system", avatar="ℹ️"):
            st.markdown(message["content"])

# Handle interrupts based on type
def handle_interrupt(interrupt_data):
    st.session_state.interrupt_data = interrupt_data
    st.session_state.waiting_for_input = True
    
    # Image review interrupt
    if "image_url" in interrupt_data:
        with st.chat_message("assistant"):
            st.image(interrupt_data["image_url"], caption="Generated Image")
            st.markdown("Do you like this image? (yes/no + feedback)")
        
        # Use a text input instead of buttons to match what agent.py expects
        user_response = st.text_input("Your response:", key="image_feedback")
        if st.button("Submit"):
            st.session_state.waiting_for_input = False
            # Add the response to messages for display
            st.session_state.messages.append({"role": "user", "content": user_response})
            return user_response  # Return the raw text response
    
    # Question interrupt (for yes/no questions)
    elif "question" in interrupt_data and "message" in interrupt_data:
        with st.chat_message("assistant"):
            st.markdown(interrupt_data.get("message", ""))
            st.markdown(interrupt_data.get("question", ""))
        
        # Use a text input for yes/no responses
        user_response = st.text_input("Your response (yes/no):", key="question_response")
        if st.button("Submit"):
            st.session_state.waiting_for_input = False
            # Add the response to messages for display
            st.session_state.messages.append({"role": "user", "content": user_response})
            return user_response  # Return the raw text response
    
    # Terms negotiation interrupt
    elif "fields" in interrupt_data and "commercial_rev_share" in [field["name"] for field in interrupt_data.get("fields", [])]:
        with st.chat_message("assistant"):
            st.markdown(interrupt_data.get("explanation", ""))
        
        # Use text inputs instead of sliders/checkboxes to match what agent.py expects
        rev_share = st.text_input(
            "Commercial Revenue Share (0-100%):", 
            value="15",
            key="rev_share_input"
        )
        
        derivatives_allowed = st.text_input(
            "Allow Derivative Works? (yes/no):", 
            value="yes",
            key="derivatives_input"
        )
        
        if st.button("Submit"):
            st.session_state.waiting_for_input = False
            # Format the response as expected by agent.py
            response_text = f"Commercial Revenue Share: {rev_share}%, Derivatives Allowed: {derivatives_allowed}"
            # Add the response to messages for display
            st.session_state.messages.append({"role": "user", "content": response_text})
            return response_text  # Return the formatted text response
    
    # Feedback on terms interrupt
    elif "fields" in interrupt_data and "adjust_terms" in [field["name"] for field in interrupt_data.get("fields", [])]:
        with st.chat_message("assistant"):
            st.markdown(interrupt_data.get("explanation", ""))
        
        # Use text input for yes/no response
        adjust_terms = st.text_input(
            "Would you like to adjust your terms based on this feedback? (yes/no):", 
            value="yes",
            key="adjust_terms_input"
        )
        
        if st.button("Submit"):
            st.session_state.waiting_for_input = False
            # Add the response to messages for display
            st.session_state.messages.append({"role": "user", "content": adjust_terms})
            return adjust_terms  # Return the raw text response
    
    # Generic fields handler
    elif "fields" in interrupt_data:
        with st.chat_message("assistant"):
            st.markdown(interrupt_data.get("explanation", ""))
            if "question" in interrupt_data:
                st.markdown(interrupt_data.get("question", ""))
        
        # Use a text area for free-form responses
        user_response = st.text_area(
            "Your response:", 
            height=100,
            key="generic_field_response"
        )
        
        if st.button("Submit"):
            st.session_state.waiting_for_input = False
            # Add the response to messages for display
            st.session_state.messages.append({"role": "user", "content": user_response})
            return user_response  # Return the raw text response
    
    # Generic message interrupt
    else:
        with st.chat_message("assistant"):
            if "message" in interrupt_data:
                st.markdown(interrupt_data.get("message", "Input needed:"))
            else:
                st.markdown("Input needed:")
        
        user_input = st.text_area("Your response:", height=100, key="generic_input")
        if st.button("Submit"):
            st.session_state.waiting_for_input = False
            # Add the response to messages for display
            st.session_state.messages.append({"role": "user", "content": user_input})
            return user_input  # Return the raw text response
    
    return None

# Initialize MCP client and graph
async def initialize_agent():
    if st.session_state.graph is None:
        with st.spinner("Initializing agent..."):
            async with MultiServerMCPClient() as client:
                # Connect to the server
                await client.connect_to_server(
                    "story_server",
                    command="python",
                    args=["/Users/sarickshah/Documents/story/story-sdk-mcp/server.py"],
                )
                # Get all required tools
                ipfs_tools = [
                    tool
                    for tool in client.get_tools()
                    if tool.name
                    in [
                        "upload_image_to_ipfs",
                        "create_ip_metadata",
                        "mint_and_register_ip_with_terms",
                        "mint_license_tokens",
                    ]
                ]
                
                # Create the graph with the tools
                st.session_state.graph = create_graph(ipfs_tools)
                return st.session_state.graph
    return st.session_state.graph

# Process agent events
async def process_events(input_data):
    graph = await initialize_agent()
    
    # Note: Removed the config parameter that was causing errors
    async for event in graph.astream(input_data, stream_mode="updates"):
        # Handle interrupts
        if "__interrupt__" in event:
            interrupt_data = event["__interrupt__"][0].value
            st.session_state.interrupt_data = interrupt_data
            st.session_state.waiting_for_input = True
            st.rerun()
            break
        
        # Handle other events if needed
        # This would be for displaying intermediate results

# Main input form
if not st.session_state.waiting_for_input:
    with st.form(key="prompt_form"):
        user_prompt = st.text_area(
            "What image would you like to create?",
            placeholder="e.g., an anime style image of a person snowboarding",
            height=100,
        )
        submit_button = st.form_submit_button("Generate Image")
        
        if submit_button and user_prompt and not st.session_state.processing:
            st.session_state.processing = True
            st.session_state.messages.append({"role": "user", "content": f"Generate {user_prompt}"})
            
            # Run the agent
            initial_input = {
                "messages": [{"role": "user", "content": f"Generate {user_prompt}"}]
            }
            
            # Use st.rerun to refresh the page and show the message
            st.rerun()

# Handle waiting for input state
if st.session_state.waiting_for_input and st.session_state.interrupt_data:
    response = handle_interrupt(st.session_state.interrupt_data)
    if response is not None:
        # Clear the interrupt data
        interrupt_data = st.session_state.interrupt_data
        st.session_state.interrupt_data = None
        st.session_state.waiting_for_input = False
        
        # Process the response - pass the raw text response
        try:
            # For text responses, we need to format them as expected by the agent
            # The agent expects {"data": response} for simple text inputs
            asyncio.run(process_events(Command(resume={"data": response})))
        except Exception as e:
            st.error(f"Error processing response: {str(e)}")
        st.rerun()

# Process agent if we're in processing state but not waiting for input
if st.session_state.processing and not st.session_state.waiting_for_input and not st.session_state.interrupt_data:
    # Get the last user message
    last_user_message = None
    for message in reversed(st.session_state.messages):
        if message["role"] == "user":
            last_user_message = message
            break
    
    if last_user_message:
        initial_input = {
            "messages": [{"role": "user", "content": last_user_message["content"]}]
        }
        
        # Run the agent
        try:
            asyncio.run(process_events(initial_input))
            
            # Add a placeholder for the response
            with st.chat_message("assistant"):
                st.markdown("Processing your request...")
        except Exception as e:
            st.error(f"Error running agent: {str(e)}")
        
        # Use st.rerun to refresh the page
        st.rerun()
