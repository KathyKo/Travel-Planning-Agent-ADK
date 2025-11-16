import os
import json
from typing import Dict

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.genai import types

import tools_adk as tools

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise EnvironmentError("GEMINI_API_KEY not found in .env file. Please set it before running the server.")

gemini_model = Gemini(
    model_name="gemini-2.5-flash",
    api_key=GEMINI_API_KEY,
)

system_instruction = (
    "You are a helpful and expert travel planning agent. "
    "Your goal is to help the user plan a trip to any city. "
    "First, understand their preferences (like 'vegetarian' or 'museums'). "
    "Then, use your tools (web_search, get_weather, preference tools, knowledge base tools, and search_flight_price) to build a plan. "
    "When the user asks for flight prices between two cities on a specific date, call the search_flight_price tool with the origin, destination, and date. "
    "Then read through the returned results (titles, snippets, and URLs) and infer an approximate lowest price and airline name from those texts if possible. "
    "If you cannot find any clear price, explain that clearly instead of guessing. "
    "When the user asks for hotel prices for a specific date, use the web_search tool with a query that includes the hotel name, city, and date. "
    "Then read through the titles and snippets; if any approximate nightly price is mentioned (for example 'from $X per night' or similar), you should report that price as an approximate web-based price, clearly stating that it may not be exact for that date. "
    "Only say that you cannot find a price if there is truly no price-like information at all in the search results. "
    "After you have presented the plan or pricing information, ALWAYS proactively ask the user if they would also like help finding hotels OR flights for their trip. "
    "CRITICAL RULE: You must never show your internal reasoning, thoughts, or the specific tools you are calling. "
    "You must synthesize the information from your tools and present only the final, helpful answer directly to the user."
)

agent = LlmAgent(
    name="travel_agent",
    model=gemini_model,
    tools=tools.available_tools,
    instruction=system_instruction,
)

runner = InMemoryRunner(agent=agent, app_name="travel_agent_app")

app = FastAPI(
    title="Travel Planning AI Agent (ADK Version)",
    description="An AI agent built with Google ADK, deployed via FastAPI.",
)

class ChatRequest(BaseModel):
    user_id: str
    message: str

user_sessions: Dict[str, str] = {}

async def get_or_create_session_id(user_id: str) -> str:
    if user_id in user_sessions:
        return user_sessions[user_id]
    session = await runner.session_service.create_session(
        app_name=runner.app_name,
        user_id=user_id,
    )
    session_id = session.id
    user_sessions[user_id] = session_id
    return session_id

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    print(f"[Request] user={request.user_id}, message={request.message}")
    session_id = await get_or_create_session_id(request.user_id)
    user_message = types.Content(
        role="user",
        parts=[types.Part(text=request.message)],
    )
    text_parts = []
    try:
        async for event in runner.run_async(
            user_id=request.user_id,
            session_id=session_id,
            new_message=user_message,
        ):
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if getattr(part, "text", None):
                        text_parts.append(part.text)
    except Exception as e:
        print(f"[ERROR] runner.run_async failed: {e}")
        return {"user_id": request.user_id, "response": f"Sorry, I encountered an error: {e}"}
    final_text = "".join(text_parts).strip()
    if not final_text:
        final_text = "Sorry, I could not generate a response."
    print(f"[Response] {final_text[:120]}...")
    return {"user_id": request.user_id, "response": final_text}

@app.get("/")
async def read_root():
    return FileResponse("index.html")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("main_adk:app", host="127.0.0.1", port=port, reload=True)
