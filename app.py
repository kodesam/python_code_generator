import gradio as gr
import os
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SimpleSequentialChain

def load_chain(api_key):
    os.environ["OPENAI_API_KEY"] = api_key
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.1)
    first_prompt = PromptTemplate(
        input_variables=["user_in"],
        template= "Write the outline of the coding steps to develop the program {user_in} in five steps. Use Python3 and Be concise. \n\n"
    )
    chain = LLMChain(llm=llm, prompt=first_prompt)

    second_prompt = PromptTemplate(
        input_variables=["program"],
        template= '''Write the python3 code for each step of the {program} described. Use python3 style. Be concise in the code and opinionated about framework choice.'''
    )

    chain_two = LLMChain(llm=llm, prompt=second_prompt)

    overall_chain = SimpleSequentialChain(chains=[chain, chain_two], verbose=True)
    return overall_chain

def answer_question(api_key, question):
    chain = load_chain(api_key)
    output = chain.run(input=question)
    return output

ifaces = gr.Interface(
    fn=answer_question,
    inputs=[gr.inputs.Textbox(
    label="Your OpenAI API Key",
    placeholder="e.g. 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"),
    gr.inputs.Textbox(label="Write a python script to:",
    placeholder="e.g. Find the 10th number of the Fibonacci sequence")],
    outputs=gr.outputs.Textbox(label="User guide"),
    title="Python Code Generator",
    description="Enter your OpenAI API key below and a description of your desired python project in 1-2 sentences"
)

ifaces.launch()
