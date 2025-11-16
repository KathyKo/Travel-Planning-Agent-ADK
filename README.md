# Intelligent Travel Planning AI Agent (Google ADK Version)

This project is an Intelligent Travel Planning AI Agent. It provides a conversational interface for planning personalized trips to any city in the world, deployed as a robust API and web UI on Google Cloud Run.

This implementation utilizes **Google ADK**, using components like `LlmAgent` and `InMemoryRunner` to create a clean, maintainable, and framework-driven agent.

---

## 🚀 Demo & Testing

The agent is fully deployed and operational. You can interact with it in two ways:

- **Chat UI:**  
  https://travel-agent-adk-349302450067.asia-east1.run.app

- **API Docs:**  
  https://travel-agent-adk-349302450067.asia-east1.run.app/docs

### How to Test (Recommended Flow)

1. **Open the Chat UI**  
   Visit `https://travel-agent-adk-349302450067.asia-east1.run.app`.

2. **Test Long-Term Memory**
   - Send:  
     `Hi, please remember I am vegetarian and I love museums.`
   - The agent will use the `save_preference` tool to store this.

3. **Test Planning & Tool Use**
   - Send:  
     `OK, now please plan a 4-day trip to Paris for me.`
   - Observe as the agent:
     - Recalls your **“vegetarian”** and **“museums”** preferences.
     - Calls the `get_weather` tool for Paris.
     - Calls the `web_search` tool for museums.
     - Synthesizes this information into a personalized plan.

4. **Test Proactive Suggestions & Multi-hop**
   - The agent will proactively ask if you need help with **flights** or **hotels**.
   - Send:  
     `Yes, find me some good hotels near XXX.`
   - The agent will call the `find_hotels` tool, which internally triggers another `web_search` for `best rated hotels in Paris`.

---

## ✅ System Structure

### 1. Tool Use & Decision-Making

- **Tools Integrated:**  
  The agent has access to distinct tools defined in `tools_adk.py`.
- **Dynamic Decision-Making:**  
  The `LlmAgent` (powered by **Gemini 2.5 Flash**) dynamically decides which tool to call.

### 2. Memory Management

- **Short-Term Memory:**  
  Implemented using ADK’s `InMemorySessionService` in `main_adk.py`.
- **Long-Term Memory (Vector Database):**  
  Implemented using a **FAISS** database (`user_prefs.faiss`) and managed by `save_preference` / `load_preferences` tools, which are *primed* (pre-loaded) in the `/chat` endpoint.

### 3. Planning & Reasoning

- **Planning Mechanism (Framework):**  
  This project uses the **Google ADK** framework. The `InMemoryRunner` component automatically handles the entire **Plan-and-Execute** loop, replacing the need for custom `while` loop logic.
- **Multi-hop RAG:**  
  The `InMemoryRunner` automatically performs multi-hop reasoning by repeatedly calling the LLM and tools (`search_knowledge` for RAG, `web_search` for facts) until the goal is achieved.

### 4. Conversational Interface

- **Chat API (FastAPI):**  
  The entire application is a **FastAPI** service that wraps the ADK Runner.
- **Simple Web UI:**  
  A clean HTML/JS UI is served from the root (`/`) route.

### 5. Deployment

- The application is fully containerized (`Dockerfile`) and deployed on **Google Cloud Run**.

---

## 📁 Project File Structure

This project uses a modular structure to separate concerns:

```text
travel-agent/
│
├── .venv/                    # Python virtual environment (ignored)
│
├── data/                     # Stores the persistent vector databases
│   ├── knowledge_base.faiss  # RAG DB for generic travel strategies
│   └── user_prefs.faiss      # Long-Term Memory DB for user preferences
│
├── main.py                   # (Optional) The original SDK version
├── tools.py                  # (Optional) The original SDK tools
│
├── main_adk.py               # ADK Server (FastAPI + Runner)
├── tools_adk.py              # ADK-compatible Tools (use ToolContext)
│
├── build_rag.py              # One-time script to build the .faiss databases
│
├── requirements.txt          # Python package dependencies (now includes ADK)
│
├── .env                      # Local file for API keys (ignored)
│
├── .gitignore                # Files to ignore for Git
│
├── Dockerfile                # Container definition (runs main_adk.py)
│
└── index.html                # Simple HTML/CSS/JS chat UI
````

---

## 🏗️ Project Architecture (ADK)

This project uses the **Google ADK** framework, wrapped in a FastAPI server to provide both a web UI and an API.

```text
[ User (Browser) ]
       |
       |  (1. HTTP Request to / or /chat)
       v
+-----------------------------------------------+
| [ FastAPI Server (main_adk.py) on Cloud Run ] |
|      |                                        |
|      +---- [ / (serves index.html UI) ]       |
|      |                                        |
|      +---- [ /docs (serves API Docs) ]        |
|      |                                        |
|      v                                        |
|  [ /chat (API Endpoint) ]                     |
|      |                                        |
| (Manages Long-Term Memory "Priming")          |
|      |                                        |
|      v (2. await runner.invoke_chat(...))     |
|  [ ADK InMemoryRunner ] --------------------+ |
|    (Handles loops & Short-Term Memory)      | |
|      |                                      | |
|      v (3. Calls LlmAgent)                  | |
|  [ LlmAgent ]                               | |
|    (System Instruction & Tools)             | |
|      |                                      | |
|      v (4. Calls LLM)                       | |
|  [ LLM (Gemini 2.5 Flash) ] <---------------+ |
|    (Planner)                                | |  (7. Final text result)
|      |                                      | |
|      | (5. Returns FunctionCall)            | |
|      v                                      | |
|  [ ADK Runner ]                             | |
|    (Executor)                               | |
|      |                                      | |
| (6. Calls tool with ToolContext)            | |
|      v                                      | |
|  [ Toolbox (tools_adk.py) ] ----------------+ |
|      |                                        |
|      +--> [ get_weather(ctx, ...) ]           |
|      |                                        |
|      +--> [ web_search(ctx, ...) ]           |
|      |                                        |
|      +--> [ load/save_preference(ctx, ...) ] |
|           (Uses ctx.session_id)              |
|                                              |
+-----------------------------------------------+
```

---

## 🏛️ Architecture & Core Logic (ADK Version)

This implementation is cleaner than a manual SDK implementation because the ADK framework abstracts most complex agent logic.

### `main_adk.py` (The Server & Runner)

* Initializes the **FastAPI** app.

* Initializes ADK components:

  * `Gemini(...)`: The LLM "brain".
  * `LlmAgent(...)`: Bundles the Gemini model, the instruction (system prompt), and the `tools.available_tools` list.
  * `InMemorySessionService()`: The built-in short-term memory manager.
  * `InMemoryRunner(...)`: The built-in **Plan-and-Execute** loop.

* **`/chat` Endpoint**

  * Performs a one-time check for new users to **prime** (pre-load) their long-term memory from FAISS.
  * Calls `await runner.invoke_chat(...)` just once.
  * The Runner automatically handles the entire multi-step tool-calling loop in the background, repeatedly calling the LLM and tools until a final text answer is ready.

### `tools_adk.py` (The ADK Tools)

* Refactored based on Kaggle’s **“Day 2: Agent Tools”** pattern.
* Every tool now accepts `context: ToolContext` as its first argument.
* This allows tools like `save_preference` and `load_preferences` to get the `user_id` directly from `context.session_id`, which is cleaner than manually injecting `user_id` in the SDK version.

---

## 🛠️ How to Run Locally

### 1. Clone the Repository

```bash
git clone [your-repo-url]
cd travel-agent
```

### 2. Install Dependencies

* Ensure you have **Python 3.11+** installed.
* (Optional but recommended)

```bash
python -m venv .venv
# Activate your venv here depending on your OS
pip install -r requirements.txt
```

Make sure `requirements.txt` includes:

```text
google-cloud-ai-agent-development-kit
```

### 3. Set Up Environment (`.env`)

Create a file named `.env` in the root of the project and add:

```env
GEMINI_API_KEY="AIzaSy..."
OPENWEATHER_API_KEY="..."
CUSTOM_SEARCH_API_KEY="AIzaSy..."
CUSTOM_SEARCH_CX="..."
```

### 4. Build Vector Databases

This only needs to be run once:

```bash
python build_rag.py
```

This will create the `data/knowledge_base.faiss` and `data/user_prefs.faiss` files.

### 5. Run the Server

```bash
python main_adk.py
```

The server will start on:

* UI: `http://127.0.0.1:8080`
* API Docs: `http://127.0.0.1:8080/docs`



