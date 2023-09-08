# -*- coding: utf-8 -*-
import logging
from constants import CHROMA_SETTINGS, EMBEDDING_MODEL_NAME, PERSIST_DIRECTORY
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceInstructEmbeddings
import torch
from auto_gptq import AutoGPTQForCausalLM
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    LlamaForCausalLM,
    LlamaTokenizer,
    pipeline,
)
#import streamlit as st

device_type = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s",
    level=logging.INFO,
)

# Load the models and vectorstore
logging.info("Loading models and vectorstore...")

# Load the embedding model
embeddings = HuggingFaceInstructEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs={"device": "cuda"}  # Specify the device type
)

# Load the vectorstore
db = Chroma(
    persist_directory=PERSIST_DIRECTORY,
    embedding_function=embeddings,
    client_settings=CHROMA_SETTINGS,
)

retriever = db.as_retriever()


def load_model(device_type, model_id, model_basename=None):
    """
    Select a model for text generation using the HuggingFace library.
    If you are running this for the first time, it will download a model for you.
    subsequent runs will use the model from the disk.

    Args:
        device_type (str): Type of device to use, e.g., "cuda" for GPU or "cpu" for CPU.
        model_id (str): Identifier of the model to load from HuggingFace's model hub.
        model_basename (str, optional): Basename of the model if using quantized models.
            Defaults to None.

    Returns:
        HuggingFacePipeline: A pipeline object for text generation using the loaded model.

    Raises:
        ValueError: If an unsupported model or device type is provided.
    """

    logging.info(f"Loading Model: {model_id}, on: {device_type}")
    logging.info("This action can take a few minutes!")

    if model_basename is not None:
        # The code supports all huggingface models that ends with GPTQ and have some variation
        # of .no-act.order or .safetensors in their HF repo.
        logging.info("Using AutoGPTQForCausalLM for quantized models")

        if ".safetensors" in model_basename:
            # Remove the ".safetensors" ending if present
            model_basename = model_basename.replace(".safetensors", "")

        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        logging.info("Tokenizer loaded")

        model = AutoGPTQForCausalLM.from_quantized(
            model_id,
            model_basename=model_basename,
            use_safetensors=True,
            trust_remote_code=True,
            device="cuda:0",
            use_triton=False,
            quantize_config=None,
        )
    elif (
        device_type.lower() == "cuda"
    ):  # The code supports all huggingface models that ends with -HF or which have a .bin
        # file in their HF repo.
        logging.info("Using AutoModelForCausalLM for full models")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        logging.info("Tokenizer loaded")

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            # max_memory={0: "15GB"} # Uncomment this line with you encounter CUDA out of memory errors
        )
        model.tie_weights()
    else:
        logging.info("Using LlamaTokenizer")
        tokenizer = LlamaTokenizer.from_pretrained(model_id)
        model = LlamaForCausalLM.from_pretrained(model_id)

    # Load configuration from the model to avoid warnings
    generation_config = GenerationConfig.from_pretrained(model_id)
    # see here for details:
    # https://huggingface.co/docs/transformers/
    # main_classes/text_generation#transformers.GenerationConfig.from_pretrained.returns

    # Create a pipeline for text generation
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=8192,
        temperature=0.3,
        top_p=0.95,
        repetition_penalty=1.15,
        generation_config=generation_config,
    )

    local_llm = HuggingFacePipeline(pipeline=pipe)
    logging.info("Local LLM Loaded")

    return local_llm

# Load the LLM for generating natural language responses

# model_id = "TheBloke/WizardLM-7B-uncensored-GPTQ"
# model_basename = "WizardLM-7B-uncensored-GPTQ-4bit-128g.compat.no-act-order.safetensors"


# model_id= "TheBloke/Llama-2-7B-GPTQ"
# model_basename = "model"

#model_id= "TheBloke/Llama-7B-GPTQ"
#model_id= "TheBloke/llama-2-13B-Guanaco-QLoRA-GPTQ"
model_id= "TheBloke/llama-2-7B-Guanaco-QLoRA-GPTQ"
#model_id= "TheBloke/Llama-7B-GPTQ"
model_basename = "model"


llm = load_model(device_type="cuda", model_id=model_id, model_basename=model_basename)


qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=False) # later set to true


# Interactive questions and answers
# while True:
#     query = input("\nEnter a query: ")
#     if query == "exit":
#         break
#     # Get the answer from the chain
#     res = qa(query)
#     answer, docs = res["result"], res["source_documents"]

#     # Print the result
#     print("\n\n> Question:")
#     print(query)
#     print("\n> Answer:")
#     print(answer)


import gradio as gr


def question_answer(question, history):
    # Get the answer
    res = qa(question)
    answer = res['result']
    return answer

# Create a Gradio interface
# iface = gr.Interface(
#     fn=question_answer,
#     inputs="text",
#     outputs="text",
#     title="Question Answering App - AgriSense ",
#     description="Enter a question and get the answer."
# )


iface = gr.ChatInterface(
    fn=question_answer,
    title="Syngenta ",
    description="DocChat."
)

# Run the Gradio interface
iface.launch(server_name = '0.0.0.0',server_port = 8501)

# 


# st.title("AgriSense® by Syngenta")
# st.write("Everything Agriscience and Agribusiness - simplified and centralized.")

# # Add input components
# question_input = st.text_input("Enter your question:")
# submit_button = st.button("Ask")

# if submit_button and question_input:
#     # Call the question_answer function with the user's question as input
#     res = qa(question_input)
#     answer = res['result']
#     # Display the answer
#     st.text_area("Answer:",answer)
