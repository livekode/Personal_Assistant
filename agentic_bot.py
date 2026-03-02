#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#
#based. upon bot_logging  you. bot now hhas access to chat history
"""Pipecat Quickstart Example with CSV Logging and Conversation History."""

import os
import csv
import glob
from datetime import datetime
from typing import List, Dict, Optional
from dotenv import load_dotenv
from loguru import logger
from pipecat.frames.frames import Frame, TextFrame, LLMMessagesFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
import asyncio
import time
import openai
import tool_enabled_llm
from tool_enabled_llm import process_query

print("🚀 Starting Pipecat bot...")
print("⏳ Loading models and imports (20 seconds, first run only)\n")

# ==================== CSV LOGGING AND HISTORY SETUP ====================
class ConversationLogger:
    """Logger to store conversations in CSV file and retrieve history."""
    
    def __init__(self, storage_path="./conversations", session_id: Optional[str] = None):
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)
        
        # Use provided session_id or create new one
        if session_id:
            self.session_id = session_id
            self.filename = f"{self.storage_path}/conversation_{self.session_id}.csv"
            # Check if file exists, if not create with headers
            if not os.path.exists(self.filename):
                with open(self.filename, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Timestamp', 'Role', 'Message', 'Session_ID'])
        else:
            self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.filename = f"{self.storage_path}/conversation_{self.session_id}.csv"
            
            # Initialize CSV with headers
            with open(self.filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Timestamp', 'Role', 'Message', 'Session_ID'])
        
        logger.info(f"📝 CSV logging started: {self.filename}")
    
    def log_message(self, role, message):
        """Log a message to the CSV file immediately."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.filename, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, role, message, self.session_id])
            f.flush()  # Force write to disk immediately
        
        # Also log to console for visibility
        role_icon = "👤" if role == "user" else "🤖" if role == "assistant" else "⚙️"
        preview = message[:50] + "..." if len(message) > 50 else message
        logger.info(f"{role_icon} CSV LOG: [{role}] {preview}")
        
    
    def get_conversation_history(self, max_sessions: int = 5, max_messages_per_session: int = 10) -> str:
        """Retrieve conversation history from CSV files.
        
        Args:
            max_sessions: Maximum number of previous sessions to include
            max_messages_per_session: Maximum messages per session to include
            
        Returns:
            Formatted string of conversation history
        """
        history = []
        
        # Get all CSV files in the storage path, sorted by modification time (newest first)
        csv_files = glob.glob(f"{self.storage_path}/conversation_*.csv")
        csv_files.sort(key=os.path.getmtime, reverse=True)
        
        # Skip the current session file
        current_file = os.path.basename(self.filename)
        session_count = 0
        
        for file_path in csv_files:
            file_name = os.path.basename(file_path)
            
            # Skip current session
            if file_name == current_file:
                continue
                
            if session_count >= max_sessions:
                break
                
            try:
                # Read and parse CSV file
                session_messages = []
                with open(file_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        # Only include user and assistant messages, skip system
                        if row['Role'] in ['user', 'assistant']:
                            session_messages.append({
                                'timestamp': row['Timestamp'],
                                'role': row['Role'],
                                'message': row['Message']
                            })
                
                # Take the most recent messages from this session
                recent_messages = session_messages[-max_messages_per_session:]
                
                if recent_messages:
                    session_header = f"\n--- Previous Conversation (Session: {row['Session_ID']}) ---"
                    history.append(session_header)
                    
                    for msg in recent_messages:
                        role_icon = "User" if msg['role'] == 'user' else "Assistant"
                        history.append(f"{role_icon} ({msg['timestamp']}): {msg['message']}")
                    
                    session_count += 1
                    
            except Exception as e:
                logger.error(f"Error reading history file {file_path}: {e}")
        
        if history:
            return "\n".join(history)
        else:
            return "No previous conversation history available."

logger.info("Loading Silero VAD model...")
from pipecat.audio.vad.silero import SileroVADAnalyzer

logger.info("✅ Silero VAD model loaded")

from pipecat.frames.frames import LLMRunFrame

logger.info("Loading pipeline components...")
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.services.speechmatics.stt import SpeechmaticsSTTService

logger.info("✅ All components loaded successfully!")

# ==================== GLOBAL SSL FIX ====================
import ssl
import certifi

try:
    os.environ['SSL_CERT_FILE'] = certifi.where()
    os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    ssl._create_default_https_context = lambda: ssl_context
    print(f"✅ Global SSL context configured")
except Exception as e:
    print(f"⚠️ SSL fix failed: {e}")

load_dotenv(override=True)

# adding a delay. in the llmservice for other stuff
from pipecat.services.openai.llm import OpenAILLMService

import os
import asyncio
import logging
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

import os
import asyncio
import logging
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

class DelayedLLMService(OpenAILLMService):
    def __init__(self, delay_seconds=1.0, **kwargs):
        super().__init__(**kwargs)
        self.delay_seconds = delay_seconds
        self._last_user_message_time = None
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        logger.info("✅ DelayedLLMService initialized")
        
        # Configure logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
    
    async def call_llm(self, prompt):
        """Call OpenAI LLM."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = await self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0.1,
                max_tokens=150
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"❌ Error calling OpenAI: {e}")
            return "NO"
    
    async def get_user_message(self, context):
        """Extract the most recent user message from context."""
        if not context or not hasattr(context, 'messages') or not context.messages:
            return None
        
        for msg in reversed(context.messages):
            if isinstance(msg, dict) and msg.get("role") == "user":
                return msg.get("content")
        return None
    
    async def are_tools_needed(self, context):
        """Determine if tools are needed based on user message."""
        self.logger.info("🔧 CHECKING if tools are needed...")
        
        user_content = await self.get_user_message(context)
        
        if not user_content:
            self.logger.warning("⚠️ No user content found")
            return False
        
        self.logger.info(f"📝 Analyzing: '{user_content[:100]}...'")
        
        tool_check_prompt = f"""
        Determine if this message needs real-time data or tools:
        "{user_content}"
        
        Answer ONLY "YES" or "NO":
        """
        
        response = await self.call_llm(tool_check_prompt)
        result = response.strip().upper() == "YES"
        
        self.logger.info(f"🔧 TOOLS NEEDED: {result}")
        return result
    
    async def _stream_chat_completions_universal_context(self, context):
        """Override the method that actually streams completions."""
        self.logger.info("🚨🚨🚨 _stream_chat_completions_universal_context CALLED! 🚨🚨🚨")
        
        # Check if we have user messages
        if context and hasattr(context, 'messages') and context.messages:
            user_messages = [m for m in context.messages if isinstance(m, dict) and m.get("role") == "user"]
            if user_messages:
                latest_user = user_messages[-1]
                self.logger.info(f"👤 Processing user message: '{latest_user.get('content', '')[:50]}...'")
                
                
            
                current_time = asyncio.get_event_loop().time()
                
                self.logger.info(f"⏰ First message, waiting {self.delay_seconds}s...")
                await asyncio.sleep(self.delay_seconds)
                
                # Check if tools are needed
                self.logger.info("🔍 Running tool check from _stream_chat_completions_universal_context...")
                needs_tools = await self.are_tools_needed(context)
                
                if needs_tools:
                    self.logger.info("🚨 TOOLS REQUIRED - Would trigger tool pipeline here")
                    user_content = await self.get_user_message(context)
                    additional_info=process_query(user_content)
                    self.logger.info(f"🚨 got more info from langraph successfully {additional_info}")
                    context.messages.append({
                    "role": "system",
                    "content": f"REAL-TIME INFORMATION: {additional_info}"
                })
                    
                    # TODO: You could modify the context or take other actions here
                else:
                    self.logger.info("💬 No tools needed - continuing normally")
                
                self._last_user_message_time = current_time
            
        
        # Call the parent method to actually generate the response
        self.logger.debug("➡️ Calling parent _stream_chat_completions_universal_context")
        return await super()._stream_chat_completions_universal_context(context)


# ==================== CUSTOM MESSAGE LIST WITH PERIODIC HISTORY UPDATES ====================
class LoggedMessageList(list):
    """A list that logs messages and periodically updates conversation history."""
    
    def __init__(self, logger, *args, include_history=True, update_frequency=5, **kwargs):
        """
        Args:
            logger: The CSV logger instance
            *args: Positional arguments (like the initial message list)
            include_history: Whether to include conversation history
            update_frequency: How often to update history (in messages)
            **kwargs: Additional keyword arguments
        """
        super().__init__(*args, **kwargs)
        self.csv_logger = logger
        self.include_history = include_history
        self.update_frequency = update_frequency
        self.message_count = 0
        self.last_history_update = 0
        self.history_loaded = False
    
    def _update_history_in_system_message(self):
        """Update conversation history in the system message."""
        # Get updated conversation history
        history_text = self.csv_logger.get_conversation_history(
            max_sessions=3,  # Include last 3 sessions
            max_messages_per_session=5  # Last 5 messages per session
        )
        
        # Find and update the system message
        for i, msg in enumerate(self):
            if msg.get("role") == "system":
                # Extract base prompt (without previous history)
                base_content = msg['content']
                # Remove any existing history section if present
                if "IMPORTANT: You have access to previous conversations" in base_content:
                    base_content = base_content.split("IMPORTANT: You have access to previous conversations")[0].strip()
                
                # Enhanced system prompt with updated history
                enhanced_content = f"""{base_content}

IMPORTANT: You have access to previous conversations with this user. Here is their conversation history (updated as of {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}):

{history_text}

Use this history to provide more personalized and contextual responses. Reference previous topics naturally when relevant. Pay attention to any patterns or recurring topics in their history."""
                
                self[i] = {"role": "system", "content": enhanced_content}
                self.last_history_update = self.message_count
                self.history_loaded = True
                logger.info(f"📚 Conversation history updated at message #{self.message_count}")
                break
    
    def _should_update_history(self):
        """Determine if history should be updated based on message count."""
        if not self.include_history:
            return False
        
        # Always update on first user message
        if not self.history_loaded:
            return True
        
        # Update periodically based on frequency
        messages_since_update = self.message_count - self.last_history_update
        return messages_since_update >= self.update_frequency
    
    def append(self, item):
        """Override append to log messages and periodically update history."""
        # Track message count for user messages
        if item.get("role") == "user":
            self.message_count += 1
            
            # Check if we should update history
            if self._should_update_history():
                self._update_history_in_system_message()
        
        super().append(item)
        
        # Log user and assistant messages immediately
        if item.get("role") in ["user", "assistant"]:
            self.csv_logger.log_message(item["role"], item["content"])
        
        # Also log when history is updated (for debugging)
        if item.get("role") == "system" and "updated as of" in item.get("content", ""):
            logger.debug(f"📚 System message updated with fresh history")

async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info(f"Starting bot")
    
    # Initialize CSV logger
    csv_logger = ConversationLogger()

    stt = SpeechmaticsSTTService(api_key=os.getenv("SPEECHMATICS_API_KEY"))

    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id="71a7ad14-091c-4e8e-a314-022ece01c121",  # British Reading Lady
    )

    llm = DelayedLLMService(
    api_key=os.getenv("OPENAI_API_KEY"),
    delay_seconds=1  # Your 1-second delay
)
    
 

    # Enhanced system prompt that encourages using conversation history
    system_prompt = """You are a friendly AI assistant with access to the user's conversation history. 
Your responses should be natural, conversational, and personalized based on your previous interactions with this user.
If you remember topics from past conversations, feel free to reference them naturally.
Keep your answers helpful and engaging."""

# Use the logged message list with periodic history updates
    messages = LoggedMessageList(
        csv_logger,  # Positional argument
        [  # Positional argument (the initial message list)
            {
                "role": "system",
                "content": system_prompt,
            },
        ],
        include_history=True,  # Keyword argument
        update_frequency=5     # Keyword argument
    )

    context = LLMContext(messages)
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    # Original pipeline - NO processor needed
    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            stt,
            user_aggregator,  # User responses
            llm,  # LLM
            tts,  # TTS
            transport.output(),  # Transport bot output
            assistant_aggregator,  # Assistant spoken responses
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Client connected")
        csv_logger.log_message("system", "Client connected")
        
        # Kick off the conversation with a personalized greeting based on history
        greeting_prompt = """Based on any previous conversations you've had with this user, 
greet them in a personalized way. If you have history with them, reference something from 
a past conversation. If this is their first time, just give a warm welcome."""
        
        messages.append({"role": "system", "content": greeting_prompt})  # This will NOT log (system message)
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Client disconnected")
        csv_logger.log_message("system", "Client disconnected")
        logger.info(f"📊 CSV conversation log saved to: {csv_logger.filename}")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)
    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Main bot entry point for the bot starter."""

    transport_params = {
        "daily": lambda: DailyParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
        ),
        "webrtc": lambda: TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
        ),
    }

    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main
    main()