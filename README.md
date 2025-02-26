# Story IP Creator Agent

A LangGraph-based agent for creating, minting, and registering IP assets with Story.

## Overview

This agent helps users create AI-generated images, upload them to IPFS, and register them as IP assets on the Story blockchain. The process includes:

1. Generating an image using DALL-E 3
2. Getting user approval for the generated image
3. Uploading the approved image to IPFS
4. Creating IP metadata for the IP asset
5. Negotiating licensing terms with the user
6. Minting and registering the IP on Story
7. Minting license tokens for the IP

![image](https://github.com/user-attachments/assets/31ffda62-2521-4b4d-90f8-5db1cc3f02ea)

## Requirements

- Python 3.9+
- LangGraph
- LangChain
- OpenAI API key (for DALL-E and GPT models)
- Story SDK

## Directory Structure

The agent expects a specific directory structure to function properly:

```
your-root-directory/
├── langgraph-mcp-agent/    # This repository
│   ├── agent.py
│   └── ...
├── story-sdk-mcp/          # The MCP server repository
│   ├── server.py
│   └── ...
```

## Installation

1. Install uv (Universal Versioner for Python):

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Clone this repository and navigate to the project directory

3. Install dependencies using uv:

   ```bash
   uv sync
   ```

4. Set up environment variables:

   ```bash
   cp .env.example .env
   ```

   Then edit the `.env` file with your API keys and configuration.

5. Clone the [story-sdk-mcp](https://github.com/piplabs/story-sdk-mcp) repository **into the same folder that you cloned this repository**, as shown in the above **Directory Structure** section. Follow the [README instructions](https://github.com/piplabs/story-sdk-mcp/blob/main/README.md#setup) to set up and install that mcp server, making sure to set up **all** of the .env variables. You do not have to run it, it just has to be in the same folder so this agent can access it.

## Usage

Run the agent:

```bash
uv run agent.py
```

The agent will guide you through an interactive process to:

1. Enter an image description (e.g., "an anime style image of a person snowboarding")
2. Review the generated image and approve or request a new one
3. Set licensing terms including:
   - Commercial Revenue Share percentage (0-100%)
   - Whether to allow derivative works (yes/no)
4. Complete the minting process on the Story blockchain

### Example Workflow

When you run `agent.py`, you'll experience a workflow like this:

```
=== Story IP Creator ===
This tool will help you create and mint an image as an IP asset in the Story ecosystem.

What image would you like to create? (e.g., 'an anime style image of a person snowboarding'): blob skateboarding on a mountaintop

Starting the creation process...

[Image is generated and a link is given]

Do you like this image? (yes/no + feedback): yes
Uploading image to IPFS...

[Metadata is generated]

Enter Commercial Revenue Share (0-100%, default: 15%): 20
Allow Derivative Works? (yes/no, default: yes): yes

[Minting and registration process]

=== Process Complete ===
Your IP has been successfully created and registered with Story!
```

The agent handles all the complex interactions with DALL-E for image generation, IPFS for storage, and the Story blockchain for minting and registration.
