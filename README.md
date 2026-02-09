# Multilingual Ticket Routing (DDP)

A distributed deep learning system for multilingual customer support ticket routing, designed to classify and route tickets across teams, categories, or priority levels using transformer-based NLP models and PyTorch Distributed Data Parallel (DDP) training.

---

## Project Overview

Customer support tickets often arrive in multiple languages and formats. This project aims to:

- Automatically classify and route tickets
- Support multiple languages
- Scale training efficiently using PyTorch Distributed Data Parallel (DDP)
- Enable experimentation with transformer-based models


## Setup 

### 1. Clone the repository
```bash 
git clone https://github.com/PranavShashidhara/multilingual_ticket_routing_ddp.git
```
### 2. Set up the environment
Run the setup script to create and configure the virtual environment:
```bash 
bash setup.sh 
```

### 3. Activate the environment
Whenever you need to activate the environment again, run:
``` bash
source venv/bin/activate
```

Your environment is all setup and ready to go!