import json
import requests
import asyncio
import aiohttp
from typing import AsyncGenerator, Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field

class Message(BaseModel):
    role: str
    content: Union[str, List[Dict]]
    name: Optional[str] = None
    tool_calls: Optional[List[Dict]] = None
    tool_call_id: Optional[str] = None

class AgentResponse(BaseModel):
    messages: List[Message] = Field(default_factory=list)
    thread_id: str

    def pretty_print(self):
        """Print messages in a readable format"""
        for msg in self.messages:
            print(f"\n{'='*20} {msg.role.capitalize()} Message {'='*20}")
            if isinstance(msg.content, str):
                print(f"\n{msg.content}")
            elif isinstance(msg.content, list):
                for content_item in msg.content:
                    if content_item.get("type") == "text":
                        print(f"\n{content_item.get('text', '')}")
                    else:
                        print(f"\n{content_item}")
            
            if msg.tool_calls:
                print("\nTool Calls:")
                for tool_call in msg.tool_calls:
                    print(f"  - {tool_call.get('name')}: {json.dumps(tool_call.get('args', {}), indent=2)}")

class AgentClient:
    def __init__(self, base_url: str = "http://localhost:8080", agent_name: str = "story_ip_agent"):
        self.base_url = base_url
        self.agent_name = agent_name
        self.thread_id = None

    def invoke(self, message: str, thread_id: Optional[str] = None) -> AgentResponse:
        """Invoke the agent with a non-streaming request"""
        url = f"{self.base_url}/{self.agent_name}/invoke"
        thread_id = thread_id or self.thread_id
        
        payload = {
            "message": message,
            "thread_id": thread_id,
            "stream": False
        }
        
        response = requests.post(url, json=payload)
        response.raise_for_status()
        
        result = response.json()
        self.thread_id = result.get("thread_id")
        
        return AgentResponse(**result)

    async def ainvoke(self, message: str, thread_id: Optional[str] = None) -> AgentResponse:
        """Async invoke the agent with a non-streaming request"""
        url = f"{self.base_url}/{self.agent_name}/invoke"
        thread_id = thread_id or self.thread_id
        
        payload = {
            "message": message,
            "thread_id": thread_id,
            "stream": False
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                if response.status != 200:
                    text = await response.text()
                    raise Exception(f"Error {response.status}: {text}")
                
                result = await response.json()
                self.thread_id = result.get("thread_id")
                
                return AgentResponse(**result)

    async def astream(self, message: str, thread_id: Optional[str] = None) -> AsyncGenerator[Dict, None]:
        """Async stream the agent's response"""
        url = f"{self.base_url}/{self.agent_name}/stream"
        thread_id = thread_id or self.thread_id
        
        payload = {
            "message": message,
            "thread_id": thread_id,
            "stream": True
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                if response.status != 200:
                    text = await response.text()
                    raise Exception(f"Error {response.status}: {text}")
                
                # Process streaming response
                async for line in response.content:
                    if line:
                        line_str = line.decode('utf-8').strip()
                        if line_str:
                            try:
                                data = json.loads(line_str)
                                self.thread_id = data.get("thread_id", self.thread_id)
                                yield data
                            except json.JSONDecodeError:
                                print(f"Error parsing JSON: {line_str}")
    
    async def resume(
        self, 
        resume_data: Dict[str, Any], 
        thread_id: Optional[str] = None, 
        stream: bool = False
    ) -> Union[AgentResponse, AsyncGenerator[Dict, None]]:
        """Resume agent execution after an interrupt"""
        url = f"{self.base_url}/{self.agent_name}/resume"
        thread_id = thread_id or self.thread_id
        
        if not thread_id:
            raise ValueError("thread_id is required")
        
        payload = {
            "data": resume_data,
            "thread_id": thread_id,
            "stream": stream
        }
        
        print(f"CLIENT: Sending resume request to {url}")
        print(f"CLIENT: Payload: {payload}")
        
        if stream:
            async def stream_generator():
                async with aiohttp.ClientSession() as session:
                    print(f"Sending POST to {url} with payload: {payload}")
                    async with session.post(url, json=payload) as response:
                        if response.status != 200:
                            text = await response.text()
                            raise Exception(f"Error {response.status}: {text}")
                        
                        print(f"Resume response status: {response.status}")
                        
                        # Process streaming response
                        line_count = 0
                        async for line in response.content:
                            line_count += 1
                            if line:
                                line_str = line.decode('utf-8').strip()
                                print(f"Resume response line {line_count}: {line_str}")
                                if line_str:
                                    try:
                                        data = json.loads(line_str)
                                        self.thread_id = data.get("thread_id", self.thread_id)
                                        yield data
                                    except json.JSONDecodeError:
                                        print(f"Error parsing JSON: {line_str}")
                        
                        print(f"Total resume response lines: {line_count}")
            
            return stream_generator()
        else:
            # Non-streaming resume
            try:
                response = requests.post(url, json=payload)
                print(f"CLIENT: Resume response status: {response.status_code}")
                if response.status_code != 200:
                    print(f"CLIENT: Error response: {response.text}")
                response.raise_for_status()
                
                result = response.json()
                self.thread_id = result.get("thread_id")
                
                return AgentResponse(**result)
            except Exception as e:
                print(f"CLIENT: Exception in resume: {str(e)}")
                raise