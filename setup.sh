#!/bin/bash

# Setup script for PDF RAG Chatbot
echo "Setting up PDF RAG Chatbot..."

# Check if virtual environment exists
if [ ! -d "env" ]; then
    echo "Creating virtual environment..."
    python3 -m venv env
fi

# Activate virtual environment
echo "Activating virtual environment..."
source env/bin/activate

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Set environment variable for Google API key
if [ -z "$GOOGLE_API_KEY" ]; then
    echo "Please set your GOOGLE_API_KEY environment variable:"
    echo "export GOOGLE_API_KEY='your_api_key_here'"
    echo ""
    echo "You can also create a .env file with:"
    echo "GOOGLE_API_KEY=your_api_key_here"
    echo ""
fi

echo "Setup complete!"
echo ""
echo "To run the chatbot:"
echo "1. Set your GOOGLE_API_KEY environment variable"
echo "2. Run: python3 chatbot_app.py"./
echo "3. Open http://localhost:8000 in your browser"
