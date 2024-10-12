"""Functions for the RL model"""

from typing import Any

def callapi(prompt: str) -> Any:
    """Call the LLM API with the prompt"""
    return api(prompt)

def get_actions_from_program_and_state(program: str, state: str) -> dict:
    """
    Get the actions given the program and state.

    Args:
        program (str): Program recommended by the first LLM.
        state (str): State in which the agent is.

    Returns:
        dict: Result with the chosen actions.
    """
    prompt = f"You are someone with a given state : { state }.
    This state describes your health state and your habits.
    You are given an ideal health program : { program }.
    This program describes what you should do to improve your health.
    You are asked to choose actions that are consistent with your current state and based on this program.
    These actions are split into 3 categories, and each of these categories has a certain number of actions that you can choose from.
    You can only choose one action per category.
    You can choose to not take any action, in which case you should choose the 'do nothing' action.
    The 3 categories and the actions they contain are the following :
    - Sleep : [9-hour sleep, 8-hour sleep, 7-hour sleep, 6-hour sleep, do nothing]
    - Diet : [healthy diet, unhealthy diet, do nothing]
    - Exercise : [1-hour exercise, 2-hours exercise, do nothing]
    You should then output your choice as a json object with the following format :
    {
        'sleep': 'the sleep action you chose',
        'diet': 'the diet action you chose',
        'exercise': 'the exercise action you chose'
    }
    "
    actions = callapi(prompt)
    return actions

def determine_next_state(state: str, actions: dict) -> str:
    """
    Determine the next state given the current state and the taken actions.

    Args:
        state (str): Current state at time t.
        actions (dict): Action that are taken at time t. 

    Returns:
        str: Next state at time t+1.
    """
    prompt = f"You are someone with a given state : { state }.
    This state describes your health state and your habits.
    You are given a set of actions that you chose to perform : { actions }.
    You should then output your new state as a json object with the following format :
    {
        'state': 'the new state'
    }
    "
    next_state = callapi(prompt)
    return next_state

def evolution_given_program(initial_state: str, program: str, time_horizon: int) -> dict:
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
        next_state = determine_next_state(initial_state, actions)
        all_actions.append(actions)
        all_states.append(next_state)
        initial_state = next_state
    return {'actions': all_actions, 'states': all_states}

def rl_pipeline(initial_state: str, program: str, current_habits: str, time_horizon: int) -> tuple:
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
        "program": evolution_given_program(initial_state, program, time_horizon)
    }









