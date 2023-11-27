from langchain.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents import load_tools, initialize_agent, AgentType
from dotenv import load_dotenv

import streamlit as st

load_dotenv()
repo_id = "tiiuae/falcon-7b-instruct"


def generate_pet_name(animal_type, pet_color):
    prompt_template_name = PromptTemplate(
        input_variables=['animal_type', 'pet_color'],
        template="I have a pet {animal_type} and I want a cool name for it. It is {pet_color} in color. Suggest me five cool names for my pet."
    )

    llm = HuggingFaceHub(
        repo_id=repo_id,
        model_kwargs={"temperature": 0.7}
    )

    name_chain = LLMChain(llm=llm, prompt=prompt_template_name, output_key="pet_name")

    response = name_chain({'animal_type': animal_type, 'pet_color': pet_color})

    return response


def langchain_agent():
    llm = HuggingFaceHub(
        repo_id=repo_id,
        model_kwargs={"temperature": 0.5}
    )

    tools = load_tools(["llm-math"], llm=llm)
    agent = initialize_agent(
        tools=tools,
        llm=llm, 
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        # handle_parsing_errors=True,
    )

    result = agent.run(
        "If the average age of a dog is 7, multiply the age by 3."
    )
    print(result)


def main():
    st.title("Pets name generator")
    animal_type = st.sidebar.selectbox("What is your pet?", ("Cat", "Dog", "Cow", "Hamster", "Goldfish", "Parrot", "Tortoise", "Rabbit"))
    pet_color = st.sidebar.text_area(f"What color is your {animal_type.lower()}?", max_chars=15)

    if pet_color:
        response = generate_pet_name(animal_type, pet_color)
        st.text(response['pet_name'])
    
    langchain_agent()
    # print(generate_pet_name(animal_type, pet_color))


if __name__ == "__main__":
    main()
