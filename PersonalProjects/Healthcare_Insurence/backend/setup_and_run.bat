@echo off
echo ================================
echo Intelligent Policy Assistant Setup
echo ================================

REM Step 1: Create virtual environment
echo Creating virtual environment...
uv venv

REM Step 2: Install dependencies
echo Installing dependencies...
uv add fastapi uvicorn python-dotenv ^
langchain langchain-core langchain-community ^
langchain-text-splitters langchain-openai ^
chromadb tiktoken

REM Step 3: Check .env file
if not exist .env (
    echo Creating .env file...
    echo OPENAI_API_KEY=your_api_key_here > .env
    echo Please update your OPENAI_API_KEY in .env
)

REM Step 4: Create policies folder
if not exist policies (
    mkdir policies
    echo Add your policy files inside /policies folder
)

REM Step 5: Run server
echo Starting FastAPI server...
uv run python main.py

pause