**HOW TO RUN THIS PROJECT **


**Clone the Repository**

git clone https://github.com/ankitpremi12/PersonaRAG-Support-Agent.git
cd PersonaRAG-Support-Agent

** (Recommended) Create Virtual Environment**

python -m venv venv
source venv/bin/activate   # Mac/Linux
# OR
venv\Scripts\activate      # Windows


**Install Dependencies**
pip install fastapi uvicorn pydantic python-dotenv numpy


** Add Your Gemini API Key**


Create a .env file in the root directory:
GEMINI_API_KEY=your_actual_api_key_here
Or export manually:
Mac/Linux:
export GEMINI_API_KEY=your_key
Windows:
setx GEMINI_API_KEY "your_key"


 **Ensure Knowledge Base File Exists**

 
Make sure docs.txt is in the root folder:
PersonaRAG-Support-Agent/
├── app.py
├── retriever.py
├── persona.py
├── generator.py
├── escalation.py
├── docs.txt   

**Run the FastAPI Server**


uvicorn app:app --reload   # COMMAND TO RUN THE PROJECT
You should see:
Application startup complete.
Uvicorn running on http://127.0.0.1:8000

 
 **Test the API**

 
Open:
http://127.0.0.1:8000/docs          #  SEARCH THIS ON BROWSER TO SEE RAG INTERFACE ON WEB
Use the /chat endpoint with:
{
  "message": "My nginx is returning 504 errors",
  "attempt_count": 1
}
**Run Demo UI**
Open demo.html in your browser while the server is running.
