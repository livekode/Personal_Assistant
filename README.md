# Overview
Thids is meant to be a voice activated personal assistant, utilizing speech to text, text to speech 
with a large language model. Used pipecat for orchestration as its opensource



###### Environment Installation

### Prerequisites

- Python 3.10 or later
- [uv](https://docs.astral.sh/uv/getting-started/installation/) package manager installed

#### API keys (subject to change as more features are added)



- [Speechmatics] for Speech-to-Text
- [OpenAI] for LLM inference
- [Cartesia] for Text-to-Speech
- [TAVILY] for enabling Tavily Search



### Setup

Utilizing Python 3.12.0 

Navigate to the quickstart directory and set up your environment.

1. Install dependencies:

   ```bash
   uv pip install -r requirements.txt
   ```

2. Configure your API keys:

   Create a `.env` file:


   Then, add your API keys like below :

   ```ini
   SPEECHMATICS_API_KEY=your_speechmatics_api_key
   OPENAI_API_KEY=your_openai_api_key
   SPEECHMATICS_API_KEY=your_cartesia_api_key
   TAVILY_API_KEY=your_tavily_api_key
   ```

### Run your bot locally

```bash
uv run noise_cancellation.py (most recent bot version at the moment)
```

**Open http://localhost:7860 in your browser** and click `Connect` to start talking to your bot.


---

# updates:
Added Langraph, added websearch functionality, now the agent can search the internet via tavily search and return better responses. however, you now require a Taviliy API key

Added voice cancellation to the pipeline, using default noice cancellation parameters.

# Whats Next

1.Add more tools,  a calender tool in order to book meetings, and a gmail tool to send emails 

2. Design a front end application, give the pseronal assistant some personality

3. Containerize and Deploy the application


