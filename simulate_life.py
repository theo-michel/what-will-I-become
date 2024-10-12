# %%
import vertexai
from vertexai.preview.generative_models import GenerativeModel
import json
from dotenv import load_dotenv
import os

from input_examples import EXAMPLE_INITIAL_STATE, EXAMPLE_PROGRAM, EXAMPLE_TIME_HORIZON


def extract_dict_from_actions(actions: str):
    start_index = actions.find("{")
    end_index = actions.rfind("}")

    if start_index != -1 and end_index != -1 and start_index < end_index:
        return actions[start_index : end_index + 1]
    else:
        return ""


def format_actions_output(actions: str) -> dict:
    dict_actions = extract_dict_from_actions(actions)
    return json.loads(dict_actions)


class LifeSimulator:
    def __init__(self, env_file="conf.env"):
        load_dotenv(env_file)
        PROJECT_ID = os.getenv("PROJECT_ID")
        REGION = os.getenv("LOCATION")
        vertexai.init(project=PROJECT_ID, location=REGION)
        self.model = GenerativeModel(
            "gemini-1.5-pro-002",
            system_instruction="You are a helpful assistant that can answer questions and help with tasks.",
        )
        self.categories_actions = [
            "Sleep",
            "Diet",
            "Exercise",
            "Smoking",
            "Alcohol",
            "Social relationships",
            "Mental health",
            "Motivation",
            "Hydration",
            "Stress management",
            "Screen time",
        ]
        self.prompt_no_program = """Keep doing exactly what you are doing."""

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

    def get_actions_from_program_and_state(self, state: str, program: str) -> dict:
        """
        Get the actions given the program and state.

        Args:
            program (str): Program recommended by the first LLM.
            state (str): State in which the agent is.

        Returns:
            dict: Result with the chosen actions.
        """
        prompt = f"""I present you someone's state that describes their health state and habits : { state }. They received those recommendations from their personal coach: { program }. This is an ideal program, which means that they might not be able to respect each step of the program (it depends on their motivation, their objectives, etc… and all information that you can find in their state. Your goal is to find the realistic actions that they are going to do during the next week, based on their current state and the program they are given. Your goal is not to take the optimal actions but the most realistic ones based on their characteristics. The actions are split into different categories : { self.categories_actions }. For each category, you must choose 1 and only 1 action to take, the one that is the most probable according to you. If you do not have any information on a given category, return 'I do not have any information on that category' and do not invent anything. You must then output your actions as a string with the following json format (without forgetting the brackets) : 'category_1' : 'action_1', 'category_2': 'action_2', etc…"""

        actions = self.generate_content(prompt)
        return actions

    def determine_next_state(self, state: str, actions: dict) -> str:
        """
        Determine the next state given the current state and the taken actions.

        Args:
            state (str): Current state at time t.
            actions (dict): Action that are taken at time t.

        Returns:
            str: Next state at time t+1.
        """
        prompt = f"""I present you someone's state that describes the health state and habits that they had at the beginning of the week : { state }. During this week, they took many actions regarding different categories : { actions }. These actions are all they did during this week. You must not assume that they did something else during this week. Your goal is to determine their state at the end of the week. This new state must take into account their characteristics and the actions that they have taken during the week. Be careful and take into consideration that turning an action into a habit takes times, so their state cannot change drastically in a week. If a category contains 'I do not have any information on that category', do not take it into consideration. You must not invent something for those categories, so do not write something if you do not have any information on it. Your result must then be the realistic and probable one. You should then output the new state as a string. The format must be detailed and precise but as concise as possible."""
        next_state = self.generate_content(prompt)
        return next_state

    def get_evolution_given_program(
        self, initial_state: str, program: str, time_horizon: int
    ) -> dict:
        """
        Get the evolution of the state and the actions given the program and the initial state.

        Args:
            initial_state (str): Initial state at t=0.
            program (str): Program recommended by the first LLM.
            time_horizon (int): Number of time steps to consider.

        Returns:
            tuple: Tuple containing the list of actions and the list of states.
        """
        all_actions = []
        all_states = [initial_state]
        for t in range(time_horizon):
            actions = self.get_actions_from_program_and_state(program, initial_state)
            formatted_actions = format_actions_output(actions)
            next_state = self.determine_next_state(initial_state, formatted_actions)
            all_actions.append(formatted_actions)
            all_states.append(next_state)
            initial_state = next_state
        return {"actions": all_actions, "states": all_states}

    def simulation_pipeline(
        self, initial_state: str, program: str, time_horizon: int
    ) -> tuple:
        """
        Whole pipeline for the simulation model.

        Args:
            initial_state (str): Initial state at t=0.
            program (str): Program recommended by the first LLM.
            current_habits (str): Current habits of the agent.
            time_horizon (int): Number of time steps to consider.

        Returns:
            dict: Dictionary containing the evolution of the habits and the program.
        """
        return {
            "habits": self.get_evolution_given_program(
                initial_state, self.prompt_no_program, time_horizon
            ),
            "program": self.get_evolution_given_program(
                initial_state, program, time_horizon
            ),
        }


if __name__ == "__main__":
    simulator = LifeSimulator()
    results = simulator.simulation_pipeline(
        EXAMPLE_INITIAL_STATE, EXAMPLE_PROGRAM, EXAMPLE_TIME_HORIZON
    )
