"""Functions for the RL model"""

from typing import Any

import vertexai
from vertexai.preview.generative_models import GenerativeModel

PROJECT_ID = "mistral-alan-hack24par-810"
REGION = "us-central1"
vertexai.init(project=PROJECT_ID, location=REGION)
model = GenerativeModel(
    "gemini-1.5-pro-002",
    system_instruction="You are a helpful assistant that can answer questions and help with tasks.",
)

DICT_ACTIONS = {
    "Sleep": [
        "More than 8 hours on average",
        "7 to 8 hours on average",
        "5 to 7 hours on average",
        "Less than 5 hours on average",
    ],
    "Diet": [
        "Balanced diet on average",
        "High-protein diet on average",
        "High-carb diet on average",
        "High-fat diet on average",
        "Vegetarian diet on average",
        "Vegan diet on average",
    ],
    "Number of Workouts": [
        "More than 3 times a week",
        "2 to 3 times a week",
        "1 time a week",
        "No workout",
    ],
    "Duration of Workout": [
        "More than 2 hours",
        "1 to 2 hours",
        "30 to 60 minutes",
        "Less than 30 minutes",
        "No workout",
    ],
    "Smoking": [
        "No smoking at all",
        "1 cigarette a week",
        "1 cigarette a day",
        "More than 5 cigarettes a day",
    ],
}

INITIAL_STATE = "I do not have a good hygiene of life. I do not sleep much, and because of my work I cannot sleep more than 6 hours a night. I smoke a lot of cigarettes (I cannot stop smoking). I eat a lot of processed food but I want to start eating healthy. I really want to start exercising."

PROGRAM = """Lifestyle
    Establish a regular sleep schedule: Maintain a consistent sleep-wake cycle, even on weekends, to regulate your body's natural sleep-wake rhythm.
    Optimize your sleep environment: Create a relaxing bedtime routine to signal to your body that it's time to sleep. This could include taking a warm bath, reading a book, or listening to calming music.
    Short naps: If possible, take short naps (20-30 minutes) during the day to combat daytime sleepiness. Avoid longer naps, as they can disrupt your nighttime sleep.
    Nutrition
    Gradual shift to whole foods: Start by incorporating more whole, unprocessed foods into your diet, such as fruits, vegetables, and lean proteins. Gradually decrease your consumption of processed foods.
    Meal prepping: Prepare meals and snacks in advance to avoid relying on unhealthy convenience foods when you're short on time.
    Hydration: Drink plenty of water throughout the day to stay hydrated and support overall health.
    Mental
    Mindfulness exercises: Practice mindfulness techniques, such as deep breathing or meditation, to manage stress and improve overall well-being.
    Stress management: Find healthy ways to manage stress, such as spending time in nature, pursuing hobbies, or engaging in relaxing activities.
    Sport
    Regular physical activity: Incorporate regular physical activity into your routine, even if it's just a short walk or some stretching. Exercise can improve sleep quality and reduce stress."""


def generate_content(text):
    response = model.generate_content(
        [text],
        generation_config={
            "max_output_tokens": 512,
            "temperature": 0.3,
            "top_p": 0.95,
        },
    )
    return response.candidates[0].content.parts[0].text


def get_actions_from_program_and_state(program: str, state: str) -> dict:
    """
    Get the actions given the program and state.

    Args:
        program (str): Program recommended by the first LLM.
        state (str): State in which the agent is.

    Returns:
        dict: Result with the chosen actions.
    """
    prompt = f"""I present the state of someone that describes their health state and habits : { state }. They received those recommendations from their personal coach: { program }. This is an ideal program, which means that they might not be able to respect each step of the program (it depends on their motivation, their objectives, etc… and all information that you can find in their state. Your goal is to find the realistic actions that they are going to do during the next week, based on their current state and the program they are given. Your goal is not to take the optimal actions but the most realistic ones based on their characteristics. The actions that you can take are predetermined and are split into those categories : {DICT_ACTIONS}.For each category, you must choose 1 and only 1 action to take, the one that is the most probable according to you. You must then output your actions as a string with the following format : sleep action, diet action, number of workouts action, duration of workout action smoking action"""

    actions = generate_content(prompt)
    return actions


def format_actions_output(actions: str) -> dict:
    formatted_actions = {}
    actions = actions.replace("\n", "")
    for i, category in enumerate(DICT_ACTIONS.keys()):
        formatted_actions[category] = actions.split(",")[i]
    return formatted_actions


def determine_next_state(state: str, actions: dict) -> str:
    """
    Determine the next state given the current state and the taken actions.

    Args:
        state (str): Current state at time t.
        actions (dict): Action that are taken at time t.

    Returns:
        str: Next state at time t+1.
    """
    prompt = f"""I present you someone’s state that describes the health state and habits that they had at the beginning of the week : { state }. During this week, they took many actions regarding different categories : { actions }. These actions are all they did during this week. You must not assume that they did something else during this week. Your goal is to determine their state at the end of the week. This new state must take into account their characteristics and the actions that they have taken during the week. Be careful and take into consideration that turning an action into a habit takes times, so their state cannot change drastically in a week. Your result must then be the realistic and probable one. You should then output the new state as a string. The format must be detailed and precise but as concise as possible."""
    next_state = generate_content(prompt)
    return next_state


def evolution_given_program(
    initial_state: str, program: str, time_horizon: int
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
        actions = get_actions_from_program_and_state(program, initial_state)
        formatted_actions = format_actions_output(actions)
        next_state = determine_next_state(initial_state, formatted_actions)
        all_actions.append(formatted_actions)
        all_states.append(next_state)
        initial_state = next_state
    return {"actions": all_actions, "states": all_states}


def rl_pipeline(
    initial_state: str, program: str, current_habits: str, time_horizon: int
) -> tuple:
    """
    Whole pipeline for the RL model.

    Args:
        initial_state (str): Initial state at t=0.
        program (str): Program recommended by the first LLM.
        current_habits (str): Current habits of the agent.
        time_horizon (int): Number of time steps to consider.

    Returns:
        dict: Dictionary containing the evolution of the habits and the program.
    """
    return {
        "habits": evolution_given_program(initial_state, current_habits, time_horizon),
        "program": evolution_given_program(initial_state, program, time_horizon),
    }
