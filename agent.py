import asyncio
import os
from pydantic import BaseModel
from typing import List
import gradio as gr
from langchain_openai import ChatOpenAI
from browser_use import ActionResult, Agent, Controller
import json

# Define a Pydantic model for the structured output
class ApproachesOutput(BaseModel):
    approaches: List[str]

controller = Controller()

@controller.registry.action('Have user login to any website')
async def authenticate_user():
    input("I can't login to this website, please login to the website and press enter to continue: ")
    return ActionResult(extracted_content='User logged in to website. Continue with task.')

async def run_agent(task):
    agent = Agent(
        task=task,
        llm=ChatOpenAI(model="gpt-4o"),
        controller=controller,
    )
    result = await agent.run(100)
    return result.final_result()

async def enrich_lead(email: str, num_agents: int):
    crm_task = f"""
    You are an advanced AI research agent specializing in lead enrichment for RunPod, a GPU cloud provider.
    Your task is to gather detailed, up-to-date information about a given lead based on their current lead information, such as email address or name. 
    Put a focus on potential customers in AI, machine learning, and cloud computing. 
    Prioritize leads working in generative AI, deep learning, LLM training, and high-performance computing (HPC). 
    Use reliable online sources, company websites, LinkedIn, and industry-specific databases to find relevant details. 
    Retrieve the following attributes:
    Basic Lead Information
    Full Name (if not provided)
    Job Title & Role 
    (especially CTOs, ML Engineers, AI Researchers, Founders)
    Company Name & Industry 
    (prioritize AI startups, cloud computing, and HPC-related companies)
    Company Size & Revenue (if available)
    LinkedIn Profile URL
    Company Website & Social Media Links
    Technical & Industry Relevance
    Primary Use Case for GPUs (LLM training, inference, AI research, synthetic data, etc.)
    Tech Stack Used (PyTorch, TensorFlow, JAX, Kubernetes, etc.)
    Hiring Trends (e.g., actively hiring AI engineers or infrastructure specialists)
    Recent Funding Rounds & Investors (to gauge budget for cloud infrastructure)
    Notable AI Projects or Partnerships (e.g., partnerships with OpenAI, NVIDIA, Hugging Face)
    Additional Contact & Insights
    Phone Number (if publicly available)
    Office Location & HQ Address
    Recent News Mentions or Blog Posts
    Potential Buying Intent (based on hiring trends, funding, or product launches that require GPU compute)
    Ensure all information is accurate and from credible sources. Return the results in a structured JSON format suitable for CRM integration.

Current Lead Information:
{email}
"""
    print("lead info: ", email)

    structured_llm = ChatOpenAI(model="gpt-4o-mini").with_structured_output(ApproachesOutput)
    response = structured_llm.invoke(
        input=f"""Come up with 10 different approaches to the task: {crm_task}
        """,
        max_tokens=1000,
        n=10,
        stop=None,  
        temperature=0.5
    )
    approaches = ApproachesOutput(approaches=response.approaches)
    print("Approaches: ", approaches)
    
    agent_tasks = [run_agent(f"task: {crm_task}\napproach: {approach}\nCurrent Lead Information: {email}") for approach in approaches.approaches]

    results = await asyncio.gather(*agent_tasks[:num_agents])

    llm = ChatOpenAI(model="gpt-4o-mini")
    summary = llm.invoke(
        input=f"""Summarize the results of the following approaches: {results}. If there is 
        conflicting information, use your best judgement to determine the most accurate information.
        If there is no information, say so.
        Only return the summary for the given task: {crm_task}.
        """,
        max_tokens=1000,
        temperature=0.3
    )

    return {
        "results": results,
        "summary": json.loads(str(summary.content).replace("```json\n", "").replace("```", "")) if summary.content else {}
    }

def create_ui():
    with gr.Blocks(title='Lead Enrichment') as interface:
        gr.Markdown('# Lead Enrichment Task Automation')

        with gr.Row():
            with gr.Column():
                email = gr.Textbox(label='Contact Lead Info', placeholder='Enter lead email or info here')
                num_agents = gr.Number(label='Number of Agents', value=3, precision=0)
                submit_btn = gr.Button('Run Task')

            with gr.Column():
                result_output = gr.JSON(label='Result Output')
                summary_output = gr.JSON(label='Summary Output')

        def run_task(email, num_agents):
            results = asyncio.run(enrich_lead(email, int(num_agents)))
            return results['results'], results['summary']

        submit_btn.click(
            fn=run_task,
            inputs=[email, num_agents],
            outputs=[result_output, summary_output],
        )

    return interface

if __name__ == '__main__':
    demo = create_ui()
    demo.launch()
