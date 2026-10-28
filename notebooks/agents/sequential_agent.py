
from google.adk.agents import SequentialAgent

# Import your agents
from extractor_agent import extractor_agent
from simplifier_agent import simplifier_agent
from verifier_agent import verifier_agent
from refiner_agent import refiner_agent
from readability_agent import readability_agent


# Run the agents in order using Google ADK
root_agent = SequentialAgent(
    name="FaithfulMedPipeline",
    sub_agents=[
        extractor_agent,
        simplifier_agent,
        verifier_agent,
        refiner_agent,
        readability_agent,
    ],
)
