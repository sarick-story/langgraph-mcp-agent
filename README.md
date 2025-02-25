# Story Protocol NFT Creator Agent

A LangGraph-based agent for creating, minting, and registering NFTs with Story Protocol.

## Overview

This agent helps users create AI-generated images, upload them to IPFS, and register them as IP assets on the Story Protocol blockchain. The process includes:

1. Generating an image using DALL-E 3
2. Getting user approval for the generated image
3. Uploading the approved image to IPFS
4. Creating IP metadata for the NFT
5. Negotiating licensing terms with the user
6. Minting and registering the IP on Story Protocol
7. Minting license tokens for the IP

## Project Structure

```
langgraph-mcp-agent/
├── main.py                    # Main entry point
├── config.py                  # Configuration and constants
├── models/
│   ├── __init__.py
│   └── state.py               # State definitions
├── tools/
│   ├── __init__.py
│   ├── image.py               # Image generation tools
│   ├── feedback.py            # Human feedback tools
│   └── ipfs.py                # IPFS tools wrapper
├── nodes/
│   ├── __init__.py
│   ├── llm.py                 # LLM calling nodes
│   ├── tools.py               # Tool execution nodes
│   ├── review.py              # Human review nodes
│   ├── metadata.py            # Metadata generation/creation
│   ├── negotiation.py         # Terms negotiation
│   └── minting.py             # IP minting and registration
└── graph/
    ├── __init__.py
    ├── builder.py             # Graph construction
    └── runtime.py             # Logic for running with interrupts
```

## Requirements

- Python 3.9+
- LangGraph
- LangChain
- OpenAI API key (for DALL-E and GPT models)
- Story Protocol SDK

## Installation

1. Clone the repository
2. Install dependencies:
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    uv sync
    ```
3. Set up environment variables:
   ```
   cp .env.example .env
   ```

## Usage

Run the agent:

```
python main.py
```

Follow the interactive prompts to:
1. Enter an image description
2. Review the generated image
3. Set licensing terms
4. Complete the minting process

## Development

### Adding New Nodes

To add a new node to the workflow:

1. Create a new class in the appropriate file in the `nodes/` directory
2. Add the node to the graph in `graph/builder.py`
3. Update the edges as needed

### Adding New Tools

To add a new tool:

1. Add the tool function to the appropriate file in the `tools/` directory
2. Update the tools dictionary in `graph/builder.py`

## License

MIT 