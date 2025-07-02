# Episodic Memory for Large Language Models

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-Coming%20Soon-orange.svg)]()

🧠 **Persistent Episodic Memory for LLMs using Temporal Knowledge Graphs**

## Overview
This project implements novel memory architectures that enable Large Language Models to maintain coherent, persistent memory across conversations. Unlike traditional approaches, we use Temporal Knowledge Graphs to capture and maintain contextual relationships over time.

## Key Features
- 🔄 **Temporal Knowledge Graphs**: Dynamic memory structures that evolve over time
- 🧠 **Memory Consolidation**: Simulates "sleep" processes to strengthen important memories
- ⚡ **Efficient Retrieval**: Sub-2s response times for production deployment
- 📊 **Rigorous Evaluation**: Comprehensive benchmarking against state-of-the-art

## Project Status
🚧 **In Development** - Summer 2025 Research Project

## Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- 16GB+ RAM

### Setup
```bash
# Clone the repository
git clone https://github.com/tu-usuario/episodic-memory-llm
cd episodic-memory-llm

# Install UV (if not already installed)
pip install uv

# Create environment and install dependencies
uv sync

# Activate environment
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows
