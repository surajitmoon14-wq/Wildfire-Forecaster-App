# Overview

This project is an Anthropic Computer Use Demo running on Replit that showcases Claude's ability to interact with computer interfaces using vision and tool execution capabilities. The demo includes both the core computer use functionality and what appears to be a wildfire risk prediction application with an interactive map interface. The system allows users to send commands to Claude through a web interface while Claude can take screenshots, interact with the desktop, run bash commands, and edit files.

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Frontend Architecture
- **Web Interface**: HTML templates using Leaflet.js for interactive mapping functionality
- **Map Integration**: Leaflet.js library for displaying geographical data and handling user interactions
- **Streamlit Integration**: Web-based interface for real-time interaction with the AI agent
- **Responsive Design**: CSS styling optimized for various screen sizes with full-screen map display

## Backend Architecture
- **Flask Application**: Python web server handling HTTP requests and serving the wildfire prediction interface
- **Anthropic API Integration**: Direct integration with Claude's computer use capabilities through the official Anthropic SDK
- **Async Event Loop**: Asynchronous sampling loop for handling tool execution and API communication
- **Tool System**: Modular tool architecture supporting computer vision, bash execution, and file editing

## Core Tool System
- **Computer Tool**: Screenshot capture, mouse/keyboard interaction, and desktop automation
- **Bash Tool**: Command-line interface with session management and timeout handling
- **Edit Tool**: File system operations including view, create, edit, and undo functionality
- **Tool Collection**: Centralized management system for orchestrating multiple tools

## AI Agent Architecture
- **Beta API Integration**: Uses Anthropic's computer-use-2024-10-22 beta flag for enhanced capabilities
- **Multi-Provider Support**: Compatible with Anthropic, AWS Bedrock, and Google Vertex AI
- **Message Handling**: Structured conversation flow with tool result integration
- **Error Handling**: Comprehensive error management with fallback mechanisms

## Machine Learning Integration
- **Wildfire Prediction Model**: Scikit-learn model loaded via joblib for risk assessment
- **Feature Engineering**: Combines geographical, meteorological, and topographical data
- **Prediction Pipeline**: Real-time risk calculation based on location coordinates and temporal factors

# External Dependencies

## AI and Machine Learning
- **Anthropic SDK**: Core AI capabilities and computer use functionality
- **Claude 3.5 Sonnet**: Primary language model for computer interaction
- **Scikit-learn/Joblib**: Machine learning model persistence and prediction
- **NumPy/Pandas**: Data processing and numerical computation

## Web Framework and UI
- **Flask**: Python web application framework
- **Streamlit**: Interactive web interface for AI agent communication
- **Leaflet.js**: Open-source mapping library for geographical visualization

## System Integration
- **AsyncIO**: Asynchronous programming support for concurrent operations
- **Subprocess Management**: Command execution with timeout and session handling
- **Base64 Encoding**: Image processing for screenshot functionality

## Development and Deployment
- **Replit Platform**: Cloud-based development and hosting environment
- **Environment Variables**: Secure API key management through Replit Secrets
- **Virtual Machine Isolation**: Containerized execution environment for security

## Security Considerations
- **Sandboxed Environment**: Replit provides isolated execution context
- **API Key Protection**: Secure storage of Anthropic API credentials
- **Limited System Access**: Controlled tool execution with safety boundaries
- **User Consent Mechanisms**: Built-in warnings and confirmation prompts for sensitive operations