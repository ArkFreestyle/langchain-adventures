from dotenv import find_dotenv, load_dotenv
# from transformers import pipeline
from langchain.llms import HuggingFaceHub
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import streamlit as st
import requests
import time
import os

load_dotenv()
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Image to text model
def img2text(filename):
    # Call the model using the inference API
    API_URL = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-base"
    headers = {"Authorization": f"Bearer {HUGGINGFACEHUB_API_TOKEN}"}

    with open(filename, "rb") as f:
        data = f.read()
    
    payload = {
        "wait_for_model": True,
    }
    response = requests.post(API_URL, headers=headers, data=data, json=payload)
    print("img2text -> response:", response)
    print("img2text -> response.json():", response.json())
    text = response.json()[0].get("generated_text")

    # Call the model using pipeline, which allows you to download and run the model locally
    # image_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    # text = image_to_text(filename)

    print("\nimg2text output:\n", text, "\n")
    return text


# LLM to generate a short story
def generate_story(scenario):
    template = """
    You are a story teller. You can generate a short story based on a simple narrative, the story should be no more than 20 words:
    CONTEXT: {scenario}
    STORY:
    """
    prompt = PromptTemplate(template=template, input_variables=["scenario"])
    # repo_id = "google/flan-t5-xxl"
    repo_id = "tiiuae/falcon-7b-instruct"
    story_llm = LLMChain(llm=HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature": 1, "max_length": 64}), prompt=prompt, verbose=True)
    story = story_llm.predict(scenario=scenario)

    # Trying out HuggingFace Inference API here too
    # API_URL = "https://api-inference.huggingface.co/models/tiiuae/falcon-7b-instruct"
    # headers = {"Authorization": f"Bearer {HUGGINGFACEHUB_API_TOKEN}"}
    # payload = {"inputs": template}
    # response = requests.post(API_URL, headers=headers, json=payload)
    # print("generate_story() -> response.json():", response.json())
    # story = response.json()[0].get("generated_text")


    print("\nstory_llm output:\n", story, "\n")
    return story


# Text to speech model to convert the story into audio
def text2speech(message):
    API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
    headers = {"Authorization": f"Bearer {HUGGINGFACEHUB_API_TOKEN}"}
    payload = {
        "inputs": message,
        "wait_for_model": True,
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    with open('audio.flac', 'wb') as file:
        file.write(response.content)


def main():
    st.set_page_config(page_title="img 2 audio story", page_icon="💀")
    st.header("Turn img into audio story")
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        print(uploaded_file)

        # Save the file uploaded by the user
        bytes_data = uploaded_file.getvalue()
        with open(uploaded_file.name, "wb") as file:
            file.write(bytes_data)
        
        # Display the uploaded image
        st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
        
        # Call the functions that do the actual work
        scenario = img2text(uploaded_file.name)
        story = generate_story(scenario)
        text2speech(story)

        # Display scenario, story and audio file on the UI
        with st.expander("scenario"):
            st.write(scenario)
        with st.expander("story"):
            st.write(story)
        st.audio("audio.flac")


if __name__ == "__main__":
    main()


# The below code is just for testing different models following the langchain docs: https://python.langchain.com/docs/integrations/llms/huggingface_hub
# Unfortunately, most of them timeout for me :/

# question = "What is sodium chloride?"
# template = """Question: {question}
# Answer: Let's think step by step."""

# prompt = PromptTemplate(template=template, input_variables=["question"])

# # repo_id = "google/flan-t5-xxl"
# # repo_id = "databricks/dolly-v2-3b"
# # repo_id = "Writer/camel-5b-hf" # times out
# # repo_id = "Salesforce/xgen-7b-8k-base" # times out
# # repo_id = "tiiuae/falcon-40b"
# # repo_id = "meta-llama/Llama-2-70b-chat-hf"

# llm = HuggingFaceHub(
#     repo_id=repo_id, model_kwargs={"temperature": 0.5, "max_length": 64}
# )

# start_time = time.time()
# llm_chain = LLMChain(prompt=prompt, llm=llm)
# end_time = time.time()

# print(f"\nLLM {repo_id} output:\n", llm_chain.run(question), "\n", f"Time taken to run chain: {end_time - start_time}")
