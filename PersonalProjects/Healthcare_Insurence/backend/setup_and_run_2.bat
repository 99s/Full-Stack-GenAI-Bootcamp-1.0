@echo off
echo ================================
echo Installing Dependencies (UV)
echo ================================

REM Step 1: Create virtual environment
echo Creating virtual environment...
uv venv

REM Step 2: Install packages
echo Installing packages...
uv add fastapi uvicorn python-dotenv ^
langchain langchain-core langchain-community ^
langchain-text-splitters langchain-openai ^
chromadb sentence-transformers tiktoken

echo.
echo ✅ Installation Complete!
pause