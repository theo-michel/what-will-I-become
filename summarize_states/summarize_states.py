import vertexai
from vertexai.preview.generative_models import GenerativeModel
from dotenv import load_dotenv
import os

from summarize_states.explo.input_example import EXAMPLE_STATES_2, EXAMPLE_ACTIONS_2


class StateSummarizer:
    def __init__(self, env_file="conf.env"):
        load_dotenv(env_file)
        PROJECT_ID = os.getenv("PROJECT_ID")
        REGION = os.getenv("LOCATION")
        vertexai.init(project=PROJECT_ID, location=REGION)
        self.model = GenerativeModel(
            "gemini-1.5-pro-002",
            system_instruction="You are a helpful assistant that can answer questions and help with tasks.",
        )

    def generate_content(self, text):
        response = self.model.generate_content(
            [text],
            generation_config={
                "max_output_tokens": 512,
                "temperature": 0.3,
                "top_p": 0.95,
            },
        )
        return response.candidates[0].content.parts[0].text

    def summarize_program_states(self, actions: list[str], states: list[str]) -> dict:
        prompt = f"""I present you a series of actions that a person did during the past few weeks : { actions }. The first actions correspond to the ones they made during the first week, and the last ones correspond to the actions they made during the past week. I also present you a series of states that they went through during those weeks : { states }. Your goal is to make a summary of what they did and what they have been through during those weeks. This summary must not exceed 3 sentences. The summary should have a motivational tone, because they feel proud of what they accomplished and the progress they made. You must use the first person. You must output a string containing only the summary as a result."""
        summary = self.generate_content(prompt)
        return summary

    def summarize_habits_states(self, actions: list[str], states: list[str]) -> dict:
        prompt = f"""I present you a series of actions that a person did during the past few weeks : { actions }. The first actions correspond to the ones they made during the first week, and the last ones correspond to the actions they made during the past week. I also present you a series of states that they went through during those weeks : { states }. Your goal is to make a summary of what they did and what they have been through during those weeks. This summary must not exceed 3 sentences. The summary should have a deceptive tone, because they did not make any progress and feel sad about it. You must use the first person. You must output a string containing only the summary as a result."""
        summary = self.generate_content(prompt)
        return summary

    def summarize_states(
        self,
        results: dict,
    ) -> dict:
        results_program = results["program"]
        results_habits = results["habits"]
        summary_program = self.summarize_program_states(
            results_program["actions"], results_program["states"]
        )
        summary_habits = self.summarize_habits_states(
            results_habits["actions"], results_habits["states"]
        )
        return {
            "program": summary_program,
            "habits": summary_habits,
        }


if "name" == "__main__":
    summarizer = StateSummarizer()
    results = summarizer.summarize_program_states(EXAMPLE_ACTIONS_2, EXAMPLE_STATES_2)
