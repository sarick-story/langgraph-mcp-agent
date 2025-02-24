from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper
from typing import Annotated, TypedDict, Literal
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command, interrupt
from loguru import logger
from langsmith import Client, traceable
import uuid
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables import RunnableWithMessageHistory, RunnableWithFallbacks

# Initialize LangSmith client
langsmith_client = Client()

# Define our state
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], "The messages in the conversation"]
    sender: str
    thread_id: str

# Update system prompt to be simpler
SYSTEM_PROMPT = """Generate an image based on the user's request."""

# Create DALL-E tool
@tool
@traceable(name="DALL-E Image Generation")
def generate_image(prompt: str) -> str:
    """Generate an image using DALL-E based on the prompt."""
    logger.info(f"Generating image with prompt: {prompt}")
    dalle = DallEAPIWrapper()
    image_url = dalle.run(prompt)
    logger.info(f"Generated image URL: {image_url}")
    return f"Generated image URL: {image_url}"

# Create human feedback tool
@tool
@traceable(name="Human Feedback")
def get_human_feedback(image_url: str) -> str:
    """Get human feedback on whether to upload the image to IPFS."""
    response = interrupt({
        "message": f"Image has been generated: {image_url}\nWould you like to upload this to IPFS? (yes/no)"
    })
    logger.info(f"Human feedback received: {response['data']}")
    return response["data"]

async def setup_mcp_client():
    """Setup MCP client and get IPFS tools"""
    async with MultiServerMCPClient() as client:
        await client.connect_to_server(
            "story_server",
            command="python",
            args=["/Users/sarickshah/Documents/story/story-sdk-mcp/server.py"],
        )
        ipfs_tools = [tool for tool in client.get_tools() if tool.name == "upload_image_to_ipfs"]
        return ipfs_tools

class State(MessagesState):
    """Simple state."""

def create_graph(ipfs_tools):
    # Initialize model with BOTH tools available
    model = ChatOpenAI(model="gpt-4o").bind_tools([generate_image, ipfs_tools[0]])

    class CallLLM:
        async def ainvoke(self, state, config=None):
            messages = state["messages"]
            response = await model.ainvoke(messages)
            print(f"LLM Response: {response}")
            return {"messages": [response]}

    class RunTool:
        async def ainvoke(self, state, config=None):
            print("Running tool...")
            new_messages = []
            tools = {"generate_image": generate_image, "upload_image_to_ipfs": ipfs_tools[0]}
            last_message = state["messages"][-1]
            
            for tool_call in last_message.tool_calls:
                try:
                    print(f"Executing tool: {tool_call['name']}")
                    tool = tools[tool_call['name']]
                    
                    # Extract just the string value for image_data if that's the parameter
                    if tool_call['name'] == 'upload_image_to_ipfs' and 'image_data' in tool_call['args']:
                        # Make sure we're passing just the URL string, not a complex object
                        image_url = tool_call['args']['image_data']
                        result = await tool.ainvoke({"image_data": image_url})
                        print(f"IPFS upload result: {result}")
                    else:
                        result = await tool.ainvoke(tool_call['args'])
                    
                    # Make sure result is a string
                    if not isinstance(result, str):
                        result = str(result)
                    
                    new_messages.append(ToolMessage(
                        content=result,
                        name=tool_call['name'],
                        tool_call_id=tool_call['id'],
                    ))
                    
                except Exception as e:
                    print(f"Error executing tool {tool_call['name']}: {str(e)}")
                    new_messages.append(ToolMessage(
                        content=f"Error executing tool: {str(e)}",
                        name=tool_call['name'],
                        tool_call_id=tool_call['id'],
                    ))
                    
            return {"messages": new_messages}

    def human_review_node(state):
        last_message = state["messages"][-1]
        print(f"Message type: {type(last_message)}")
        print(f"Message content: {last_message.content}")

        # Get image URL from the tool message
        image_url = last_message.content.split("Generated image URL: ")[1]
        
        human_review = interrupt({
            "question": "Is this image what you wanted?",
            "image_url": image_url
        })

        if human_review.get("action") == "continue":
            # If yes, create a tool call to upload to IPFS
            return {
                "messages": [AIMessage(
                    content="Uploading approved image to IPFS",
                    tool_calls=[{
                        "id": str(uuid.uuid4()),
                        "name": "upload_image_to_ipfs",
                        "args": {"image_data": image_url}
                    }]
                )],
                "next": "run_ipfs_tool"
            }
        else:
            # If no, send feedback to LLM to regenerate
            return {
                "messages": [HumanMessage(
                    content=f"I don't like that image. {human_review.get('data', 'Please generate a different image.')}"
                )],
                "next": "call_llm"
            }

    workflow = StateGraph(State)
    
    workflow.add_node("call_llm", RunnableLambda(CallLLM().ainvoke))
    workflow.add_node("run_tool", RunnableLambda(RunTool().ainvoke))
    workflow.add_node("run_ipfs_tool", RunnableLambda(RunTool().ainvoke))
    workflow.add_node("human_review_node", RunnableLambda(human_review_node))
    
    # Start -> call LLM to generate image
    workflow.add_edge(START, "call_llm")
    
    # LLM -> run tool (for image generation)
    workflow.add_conditional_edges(
        "call_llm",
        lambda x: "run_tool" if (x["messages"] and len(x["messages"][-1].tool_calls) > 0) else END
    )
    
    # Run tool -> human review (after image generation)
    workflow.add_conditional_edges(
        "run_tool",
        lambda x: "human_review_node" if (
            x["messages"] and 
            isinstance(x["messages"][-1], ToolMessage) and 
            x["messages"][-1].name == "generate_image"
        ) else "call_llm"
    )
    
    # Human review -> either run IPFS tool or call LLM again based on response
    workflow.add_conditional_edges(
        "human_review_node",
        lambda x: x.get("next", "call_llm")
    )
    
    # IPFS tool -> END (always end after IPFS upload)
    workflow.add_edge("run_ipfs_tool", END)

    # Set up memory
    memory = MemorySaver()
    
    # Add memory to graph compilation
    graph = workflow.compile(checkpointer=memory)
    
    # Save visualization
    mermaid_png = graph.get_graph().draw_mermaid_png()
    with open("workflow_graph.png", "wb") as f:
        f.write(mermaid_png)
    
    return graph

async def run_agent():
    # Create the MCP client and keep it open for the entire session
    async with MultiServerMCPClient() as client:
        # Connect to the server
        await client.connect_to_server(
            "story_server",
            command="python",
            args=["/Users/sarickshah/Documents/story/story-sdk-mcp/server.py"],
        )
        # Get IPFS tools
        ipfs_tools = [tool for tool in client.get_tools() if tool.name == "upload_image_to_ipfs"]
        
        # Create the graph with the tools
        graph = create_graph(ipfs_tools)
        
        thread_id = str(uuid.uuid4())
        
        initial_input = {
            "messages": [
                {"role": "user", "content": "Generate an anime style image of a person snowboarding"}
            ]
        }
        
        # Add thread_id to the config
        config = {"configurable": {"thread_id": thread_id}}
        
        # Use astream for async operation
        async for event in graph.astream(initial_input, config, stream_mode="updates"):
            print(event)
            print("\n")
            
            # Check if we hit an interrupt
            if "__interrupt__" in event:
                # Get user input
                user_input = input("Do you like this image? (yes/no + feedback): ")
                
                if user_input.lower().startswith('yes'):
                    # Continue to IPFS upload
                    async for event in graph.astream(
                        Command(resume={"action": "continue"}),
                        config,
                        stream_mode="updates"
                    ):
                        print(event)
                        print("\n")
                        
                        # Check if we need to handle another interrupt
                        if "__interrupt__" in event:
                            break
                else:
                    # Get feedback after "no"
                    feedback = user_input[4:] if len(user_input) > 4 else "Please generate a different image"
                    async for next_event in graph.astream(
                        Command(
                            resume={
                                "action": "feedback",
                                "data": feedback
                            }
                        ),
                        config,
                        stream_mode="updates"
                    ):
                        print(next_event)
                        print("\n")
                        
                        # Handle interrupts recursively for subsequent image reviews
                        if "__interrupt__" in next_event:
                            user_input = input("Do you like this image? (yes/no + feedback): ")
                            
                            if user_input.lower().startswith('yes'):
                                # Continue to IPFS upload
                                async for final_event in graph.astream(
                                    Command(resume={"action": "continue"}),
                                    config,
                                    stream_mode="updates"
                                ):
                                    print(final_event)
                                    print("\n")
                            else:
                                # Get feedback after "no"
                                feedback = user_input[4:] if len(user_input) > 4 else "Please generate a different image"
                                async for _ in graph.astream(
                                    Command(
                                        resume={
                                            "action": "feedback",
                                            "data": feedback
                                        }
                                    ),
                                    config,
                                    stream_mode="updates"
                                ):
                                    # Continue the loop for more iterations if needed
                                    pass

if __name__ == "__main__":
    import asyncio
    asyncio.run(run_agent())