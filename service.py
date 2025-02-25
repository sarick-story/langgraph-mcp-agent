import asyncio
from typing import AsyncGenerator, Dict, List, Optional, Union
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field
import json
import uuid
import traceback

# Import your agent
from agent import create_graph, setup_mcp_client

# Create FastAPI app
app = FastAPI(title="Story IP Agent Service")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request and response models
class AgentRequest(BaseModel):
    message: str
    thread_id: Optional[str] = None
    stream: bool = False

class MessageContent(BaseModel):
    type: str = "text"
    text: str

class Message(BaseModel):
    role: str
    content: Union[str, List[MessageContent]]
    name: Optional[str] = None
    tool_calls: Optional[List[Dict]] = None
    tool_call_id: Optional[str] = None

class AgentResponse(BaseModel):
    messages: List[Message] = Field(default_factory=list)
    thread_id: str

# Graph and tools setup
_graph = None
_ipfs_tools = None

async def get_graph():
    global _graph, _ipfs_tools
    if _graph is None:
        # Initialize MCP client and get IPFS tools
        _ipfs_tools = await setup_mcp_client()
        # Create the graph with the tools
        _graph = create_graph(_ipfs_tools)
    return _graph

# Routes
@app.get("/info")
async def get_info():
    """Return information about the service"""
    return {
        "name": "Story IP Agent Service",
        "description": "An agent service for creating and registering IP assets in the Story ecosystem",
        "version": "0.1.0",
        "agents": ["story_ip_agent"],
        "models": ["gpt-4o", "gpt-4o-mini"]
    }

@app.post("/story_ip_agent/invoke")
async def invoke_agent(request: AgentRequest):
    """Invoke the agent with a non-streaming response"""
    thread_id = request.thread_id or str(uuid.uuid4())
    
    # Get the graph
    graph = await get_graph()
    
    # Prepare input
    input_data = {
        "messages": [{"role": "user", "content": request.message}]
    }
    
    # Configure with thread_id
    config = {"configurable": {"thread_id": thread_id}}
    
    # Run the agent
    try:
        result = await graph.ainvoke(input_data, config)
        
        # Convert response format
        messages = []
        for msg in result.get("messages", []):
            if hasattr(msg, "to_dict"):
                msg_dict = msg.to_dict()
                messages.append(Message(
                    role="assistant" if msg_dict.get("type") == "ai" else msg_dict.get("type", "system"),
                    content=msg_dict.get("content", ""),
                    name=msg_dict.get("name"),
                    tool_calls=msg_dict.get("tool_calls"),
                    tool_call_id=msg_dict.get("tool_call_id")
                ))
        
        return AgentResponse(messages=messages, thread_id=thread_id)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error invoking agent: {str(e)}")

@app.post("/story_ip_agent/stream")
async def stream_agent(request: AgentRequest):
    """Stream the agent's response"""
    thread_id = request.thread_id or str(uuid.uuid4())
    
    # Get the graph
    graph = await get_graph()
    
    # Prepare input
    input_data = {
        "messages": [{"role": "user", "content": request.message}]
    }
    
    # Configure with thread_id
    config = {"configurable": {"thread_id": thread_id}}
    
    async def stream_response() -> AsyncGenerator[str, None]:
        try:
            # Stream agent responses
            async for event in graph.astream(input_data, config, stream_mode="updates"):
                if "__interrupt__" in event:
                    # Handle interrupts by returning a special message format
                    interrupt_data = event["__interrupt__"][0].value
                    yield json.dumps({
                        "type": "interrupt",
                        "data": interrupt_data,
                        "thread_id": thread_id
                    }) + "\n"
                else:
                    # Convert normal events to messages
                    messages = []
                    for key, value in event.items():
                        if key == "messages" and value:
                            for msg in value:
                                if hasattr(msg, "to_dict"):
                                    msg_dict = msg.to_dict()
                                    messages.append({
                                        "role": "assistant" if msg_dict.get("type") == "ai" else msg_dict.get("type", "system"),
                                        "content": msg_dict.get("content", ""),
                                        "name": msg_dict.get("name"),
                                        "tool_calls": msg_dict.get("tool_calls"),
                                        "tool_call_id": msg_dict.get("tool_call_id")
                                    })
                    
                    if messages:
                        yield json.dumps({
                            "type": "message",
                            "data": {"messages": messages},
                            "thread_id": thread_id
                        }) + "\n"
        
        except Exception as e:
            yield json.dumps({
                "type": "error",
                "data": {"error": str(e)},
                "thread_id": thread_id
            }) + "\n"
    
    return StreamingResponse(
        stream_response(),
        media_type="text/event-stream"
    )

@app.post("/story_ip_agent/resume")
async def resume_agent(request: Request):
    """Resume agent after an interrupt"""
    data = await request.json()
    
    thread_id = data.get("thread_id")
    if not thread_id:
        raise HTTPException(status_code=400, detail="thread_id is required")
    
    # Fix how resume_data is extracted - be more flexible with the structure
    resume_data = data.get("data", {})
    # If 'data' isn't found, use the entire payload as resume_data
    if not resume_data and isinstance(data, dict):
        # Filter out special keys
        resume_data = {k: v for k, v in data.items() if k not in ["thread_id", "stream"]}
    
    stream = data.get("stream", False)
    
    # Get the graph
    graph = await get_graph()
    
    # Configure with thread_id
    config = {"configurable": {"thread_id": thread_id}}
    
    try:
        if stream:
            async def stream_resume() -> AsyncGenerator[str, None]:
                try:
                    # Stream agent responses after resuming
                    # Wrap resume_data in the expected format with empty messages array
                    input_data = {
                        "resume": resume_data,
                        "messages": []  # Add empty messages array to satisfy the requirement
                    }
                    
                    async for event in graph.astream(
                        input_data, 
                        config, 
                        stream_mode="updates"
                    ):
                        if "__interrupt__" in event:
                            # Handle interrupts by returning a special message format
                            interrupt_data = event["__interrupt__"][0].value
                            yield json.dumps({
                                "type": "interrupt",
                                "data": interrupt_data,
                                "thread_id": thread_id
                            }) + "\n"
                        else:
                            # Convert normal events to messages
                            messages = []
                            for key, value in event.items():
                                if key == "messages" and value:
                                    for msg in value:
                                        if hasattr(msg, "to_dict"):
                                            msg_dict = msg.to_dict()
                                            messages.append({
                                                "role": "assistant" if msg_dict.get("type") == "ai" else msg_dict.get("type", "system"),
                                                "content": msg_dict.get("content", ""),
                                                "name": msg_dict.get("name"),
                                                "tool_calls": msg_dict.get("tool_calls"),
                                                "tool_call_id": msg_dict.get("tool_call_id")
                                            })
                            
                            if messages:
                                yield json.dumps({
                                    "type": "message",
                                    "data": {"messages": messages},
                                    "thread_id": thread_id
                                }) + "\n"
                
                except Exception as e:
                    yield json.dumps({
                        "type": "error",
                        "data": {"error": str(e)},
                        "thread_id": thread_id
                    }) + "\n"
            
            return StreamingResponse(
                stream_resume(),
                media_type="text/event-stream"
            )
        else:
            # Non-streaming resume
            # Wrap resume_data in the expected format with empty messages array
            # This ensures we're writing to the 'messages' field as required
            input_data = {
                "resume": resume_data,
                "messages": []  # Add empty messages array to satisfy the requirement
            }
            
            result = await graph.ainvoke(input_data, config)
            
            # Convert response format
            messages = []
            for msg in result.get("messages", []):
                if hasattr(msg, "to_dict"):
                    msg_dict = msg.to_dict()
                    messages.append(Message(
                        role="assistant" if msg_dict.get("type") == "ai" else msg_dict.get("type", "system"),
                        content=msg_dict.get("content", ""),
                        name=msg_dict.get("name"),
                        tool_calls=msg_dict.get("tool_calls"),
                        tool_call_id=msg_dict.get("tool_call_id")
                    ))
            
            return AgentResponse(messages=messages, thread_id=thread_id)
    
    except Exception as e:
        error_traceback = traceback.format_exc()
        print(f"ERROR in resume_agent: {str(e)}")
        print(f"Request data: {data}")
        print(f"Resume data: {resume_data}")
        print(f"Thread ID: {thread_id}")
        print(f"Traceback: {error_traceback}")
        raise HTTPException(status_code=500, detail=f"Error resuming agent: {str(e)}")

# Run the server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)