import asyncio
from client import AgentClient
import json

async def main():
    client = AgentClient()
    
    # Streaming example
    print("Starting streaming example with image generation...")
    
    async for event in client.astream("Generate an anime style character"):
        print(f"Received event type: {event.get('type')}")
        
        if event.get("type") == "message":
            for message in event.get("data", {}).get("messages", []):
                print(f"Received message: {message.get('content')}")
        elif event.get("type") == "interrupt":
            # Handle interrupts (like image review)
            interrupt_data = event.get("data", {})
            print(f"Received interrupt with data: {interrupt_data}")
            
            if "image_url" in interrupt_data:
                print(f"Image generated: {interrupt_data['image_url']}")
                print("Approving image and continuing...")
                
                # Resume with feedback and continue streaming
                resume_data = {"action": "continue"}
                print(f"Sending resume with data: {resume_data}")
                
                # Make sure thread_id is properly set and passed explicitly
                thread_id = event.get("thread_id", client.thread_id)
                client.thread_id = thread_id
                await client.resume(resume_data, thread_id=thread_id)
                
                # NOTE: We removed the break statement and the processing of the resume stream
                # We'll continue in the main event loop instead
                
            # Handle negotiation interrupts
            elif "fields" in interrupt_data and any(field.get("name") == "commercial_rev_share" for field in interrupt_data.get("fields", [])):
                print("\nReceived terms negotiation interrupt!")
                explanation = interrupt_data.get("explanation", "")
                print(explanation)
                
                # Set default values
                commercial_rev_share = 20  # Using 20% as our default
                derivatives_allowed = True
                
                print(f"Setting terms: Commercial Revenue Share: {commercial_rev_share}%, Derivatives Allowed: {derivatives_allowed}")
                
                # Resume with our chosen terms
                resume_data = {
                    "commercial_rev_share": commercial_rev_share,
                    "derivatives_allowed": derivatives_allowed
                }
                
                thread_id = event.get("thread_id", client.thread_id)
                client.thread_id = thread_id
                await client.resume(resume_data, thread_id=thread_id)
                
            # Handle any other interrupts with generic response
            else:
                print(f"Received unknown interrupt: {interrupt_data}")
                thread_id = event.get("thread_id", client.thread_id)
                client.thread_id = thread_id
                await client.resume({"data": "Let's continue"}, thread_id=thread_id)

if __name__ == "__main__":
    asyncio.run(main())