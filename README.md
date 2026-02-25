# Blue chicken WFRP AI DM
<div align="center">
<img src="assets\icon1.jpg" width="200" height="200"> 
</div>
An AI agent that helps the Dungeon Master interact with the rule books Warhammer Fantasy Roleplay (WFRP). This repo uses the smolagents package to build an AI agent that can answer questions regarding WFRP rule books.

## Instalation

### Install Ollama on your device:

Ollama allows you to fast and efficiently use an embedding model that is used to create and read a vector store created from the rule book. **Instructions on how to install Ollama can be found here: [Link](https://ollama.com/download)**

###  Pull the embedding model using Ollama.

By default, you can use the mxbai-embed-large [model description](https://ollama.com/library/mxbai-embed-large). This model showed pretty good quality for English texts.  \
Open your terminal and execute this command: \

``
ollama pull mxbai-embed-large
``
\
You can also check for the installed model using this command: \
``
ollama list
``
### Install all requirements for the project:
``
pip install -r requirements.txt
``
### Set up your Hugging Face API token:
By default project use Qwen2.5-Coder-32B-Instruct [model description](https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct) model for thinking and function calling. This model is hosted by HuggingFace and available using hugging face token. \
To create a token, you need to first create a Hugging Face account and then do the next steps:
1) Get your Hugging Face token from https://hf.co/settings/tokens with permission for inference, if you don’t already have one
2) Save your API token as a system variable: for Linux you can do this using this command: ``echo 'export HF_TOKEN="hf....."' >> ~/.bashrc source ~/.bashrc``. For Windows you should do ``$env:HF_TOKEN="hf....."`` in PowerShell and ``setx HF_TOKEN "hf....."`` for cmd
### Runing App
To run the app, you just need to run ``python app.py``, this create Gradio Interface of the Chat in your browser.

## Used Rule books 
I currently used **Warhammer Fantasy Roleplay: Core Rulebook** and **Warhammer Fantasy Roleplay: Up in Arms Rulebook**. The vectorized version of this book you can find in the  ``./databases`` folder. The model was prescribed to use Core Rulebook in all cases unless it is not said to used Up in Arms.

If you want the model to use different rulebooks, you need to create new vector databases from a PDF file using the command: 
 
``python -m src.pdf_reader.vector_store $PATH_TO_YOUR_PDF 'name of the book'``. \
You can also add a description for the new book for the model at ``src.tools.rule_book.py``.

## System prompt 
System prompt for this Agent was copied form this Hugging Face course [repo](https://huggingface.co/spaces/agents-course/First_agent_template) and modifed only slightly. You can find this prompt inside ``./prompts.yaml``.
