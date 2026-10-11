✍️ AI Technical Blog Generator

A multi-agent system that writes in-depth technical blog posts from a single topic. Built with LangGraph, LangChain, Groq, Tavily and Streamlit.

Enter a topic, and the pipeline decides whether it needs live web research, plans the article, writes every section in parallel, generates diagrams, and returns a complete Markdown blog post.

📸 Add a screenshot of the app here: ![App Screenshot](screenshots/app.png)

🚀 Features
Smart routing: classifies each topic as closed_book, hybrid or open_book and decides if web research is needed.
Web research: Tavily search with LLM-based synthesis and deduplicated, cited sources.
Structured planning: an orchestrator creates a 5–9 section outline with goals, subpoints and word targets.
Parallel writing: one worker agent per section, run in parallel with LangGraph Send.
Automatic diagrams: Mermaid diagrams and AI images with a multi-tier fallback (Gemini → Mermaid → Pollinations → offline PIL diagram).
Streamlit UI: live progress, plan viewer, sources tab, diagrams tab, raw Markdown and one-click download.
🧠 Architecture
Topic
  │
  ▼
Router ──(needs research?)──► Tavily Research
  │                                │
  └────────────────┬───────────────┘
                   ▼
             Orchestrator (Plan)
                   │
        ┌──────────┼──────────┐
        ▼          ▼          ▼
     Worker 1   Worker 2   Worker N   (parallel)
        └──────────┼──────────┘
                   ▼
        Reducer Subgraph
   merge → decide images → generate & place images
                   │
                   ▼
            Final Markdown blog
🛠️ Tech Stack
Layer	Tools
Orchestration	LangGraph, LangChain
LLM	Groq (openai/gpt-oss-120b), optional OpenAI
Research	Tavily Search
Images	Gemini API, mermaid.ink, Pollinations, Pillow
Schemas	Pydantic
Frontend	Streamlit (Python, CSS, HTML)
Storage	MongoDB (optional)
📁 Project Structure
├── bwa_frontend.py      # Streamlit UI
├── bwa_backend.py       # LangGraph multi-agent pipeline
├── requirements.txt     # Python dependencies
├── .gitignore
├── 1_bwa_basic.ipynb                  # Step 1: basic blog agent
├── 2_bwa_improved_prompting.ipynb     # Step 2: improved prompting
├── 3_bwa_research.ipynb               # Step 3: web research
├── 4_bwa_research_fine_tuned.ipynb    # Step 4: fine-tuned research
└── 5_bwa_image.ipynb                  # Step 5: image generation

The notebooks show how the project evolved step by step. The final app is bwa_frontend.py + bwa_backend.py.

⚙️ Setup
1. Clone the repo
bash
git clone https://github.com/lakshthakre922-hue/blog-writing-adgent.git
cd blog-writing-adgent
2. Create a virtual environment
bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
3. Install dependencies
bash
pip install -r requirements.txt
4. Add your API keys

Create a .env file in the project root (never commit this file):

env
GROQ_API_KEY=your_groq_key
TAVILY_API_KEY=your_tavily_key
GOOGLE_API_KEY=your_google_key
# Optional
OPENAI_API_KEY=your_openai_key
MONGODB_URI=your_mongodb_connection_string
Key	Used for	Get it from
GROQ_API_KEY	LLM	console.groq.com
TAVILY_API_KEY	Web research	app.tavily.com
GOOGLE_API_KEY	Gemini image generation (optional)	aistudio.google.com
5. Run the app
bash
streamlit run bwa_frontend.py

Open http://localhost:8501 in your browser.

💡 Usage
Pick a sample idea or type your own topic, for example "Self Attention in Transformer Architecture".
Click Generate Blog Post.
Watch the agents work in the live progress panel.
Read the article, inspect the plan and sources, then download the .md file.
☁️ Deployment

GitHub only stores the code. To run the app online, connect this repo to Streamlit Community Cloud or Hugging Face Spaces:

Set the main file to bwa_frontend.py.
Add your API keys in the platform's Secrets settings.
Deploy.
🔒 Security Notes
Never commit .env or hardcode API keys in code or notebooks.
If a key is ever exposed, revoke it and create a new one.
If the app is public, add a password so others cannot use your API credits.
📄 License

MIT License. Feel free to use and modify.

🙋 Author

Laksh Thakre GitHub: @lakshthakre922-hue
