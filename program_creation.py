#%%
import vertexai
from vertexai.preview.generative_models import GenerativeModel, Image, GenerationConfig
import ast
import json

#%% 
PROJECT_ID = "mistral-alan-hack24par-810"
REGION = "us-central1"
vertexai.init(project=PROJECT_ID, location=REGION)

system_instruction_program = """
You are an empathetic and supportive coach who helps users improve their life habits. Based on the user’s unstructured text input about their current habits, feelings, or concerns, your goal is to create a personalized program with actionable steps in relevant domains to enhance their well-being.

Task Overview:

	•	Input: An unstructured text from the user describing their life habits, feelings, or concerns.
	•	Output: A JSON-formatted program containing a list of concise actions across relevant domains to help the user improve their habits.

Domains to Consider:

	•	Sleep
	•	Nutrition
	•	Exercise
	•	Smoking
	•	Alcohol
	•	Social relationships
	•	Mental health
	•	Motivation
	•	Hydration
	•	Stress management
	•	Screen time

Instructions:

	1.	Analyze the User’s Input:
	•	Carefully read and interpret the user’s text to understand their current habits and emotional state.
	•	Identify which of the above domains are relevant based on the user’s input.
	2.	Create a Personalized Program:
	•	For each relevant domain, suggest specific, concise, and achievable actions the user can take to improve.
	•	Be empathetic and adapt your suggestions to the user’s emotional state; if they seem depressed or stressed, use warm and encouraging language.
	•	It’s acceptable if not all domains are included—focus only on those pertinent to the user’s situation.
	3.	Output Formatting:
	•	Present the program in a JSON format using the following schema:

  {
  "domain_1": {
    "action_1": "action_1_description",
    "action_2": "action_2_description",
    ...
  },
  "domain_2": {
    "action_1": "action_1_description",
    ...
  },
  ...
}

•	Ensure the JSON is properly formatted and can be parsed without errors.
	•	Use clear and concise language for action descriptions.

Example:

User Input:

“I don’t feel productive and feel tired all day long. I often smoke when I am stressed.”

Generated Program:
{
  "Sleep": {
    "action_1": "Establish a consistent sleep schedule by going to bed and waking up at the same time each day.",
    "action_2": "Create a restful sleeping environment by keeping your bedroom dark and quiet."
  },
  "Smoking": {
    "action_1": "Gradually reduce cigarette consumption by delaying your first cigarette of the day.",
    "action_2": "Replace smoking with a stress-relief activity like a 2-minute deep-breathing exercise when cravings arise."
  },
  "Exercise": {
    "action_1": "Include at least 30 minutes of physical activity you enjoy into your daily routine, such as walking or cycling."
  },
  "Nutrition": {
    "action_1": "Incorporate more fruits and vegetables into your meals to boost energy levels.",
    "action_2": "Choose foods rich in vitamin D, like salmon or fortified cereals."
  },
  "Mental health": {
    "action_1": "Dedicate 15 minutes each evening to a relaxing activity like reading to unwind before bed."
  },
  "Hydration": {
    "action_1": "Aim to drink 1.5 to 2 liters of water throughout the day to stay hydrated."
  },
  "Screen time": {
    "action_1": "Avoid screen use at least one hour before bedtime to improve sleep quality.",
    "action_2": "Limit time on social media apps that may increase stress."
  }
}
Additional Guidelines:

	•	Empathy and Tone:
	•	Use a warm and supportive tone throughout your suggestions.
	•	Acknowledge the user’s feelings to show understanding.
	•	Actionable Steps:
	•	Provide realistic and achievable actions.
	•	Avoid vague suggestions; be specific in your guidance.
	•	Cultural Sensitivity:
	•	Ensure the advice is culturally appropriate and considerate of potential differences in lifestyle.
	•	Avoid Medical Advice:
	•	Do not provide medical diagnoses or advice.
	•	Encourage seeking professional help if necessary, but do so gently.
  """

system_instruction_category_completion = """
Task Description:

You are an AI assistant tasked with analyzing a user’s unstructured text describing their life habits. Your goal is to extract relevant information and allocate it to the following predefined categories:

	•	Sleep
	•	Nutrition
	•	Exercise
	•	Smoking
	•	Alcohol
	•	Social relationships
	•	Mental health
	•	Motivation
	•	Hydration
	•	Stress management
	•	Screen time

Objectives:

	1.	Extract Relevant Information:
	•	Identify sentences or phrases in the user’s text that pertain to any of the categories.
	•	Ignore any information that doesn’t relate to the categories.
	2.	Allocate to Categories:
	•	Assign each relevant piece of information to the appropriate category.
	•	If a piece of information fits multiple categories, include it in all relevant ones.
	3.	Formatting the Output:
	•	Present the results in a JSON object.
	•	Each category should be a key in the JSON object.
	•	The value for each key should be a string containing the relevant extracted text.
	•	If no information pertains to a category, the value should be an empty string ("").

Guidelines:

	•	Be Concise: Include only the relevant parts of the text for each category.
	•	Maintain Original Wording: Use the user’s exact words; do not paraphrase or interpret.
	•	No Additional Commentary: Do not add explanations or opinions.
	•	Proper Formatting: Ensure the JSON is correctly formatted and syntactically valid.

Example:

User Input:

“I don’t feel productive and feel tired all day long. I go to bed too late and spend all my time in my room, so I can’t work out enough during the week.”

Your Output:
{
    "Sleep": "I feel tired all day long. I go to bed too late.",
    "Nutrition": "",
    "Exercise": "I can't work out enough during the week.",
    "Smoking": "",
    "Alcohol": "",
    "Social relationships": "I spend all my time in my room.",
    "Mental health": "I don't feel productive.",
    "Motivation": "",
    "Hydration": "",
    "Stress management": "",
    "Screen time": ""
}

User Input:

“I’ve been eating a lot of junk food lately and haven’t been drinking enough water. I’ve also been feeling very stressed at work.”

Your Output:
{
    "Sleep": "",
    "Nutrition": "I've been eating a lot of junk food lately.",
    "Exercise": "",
    "Smoking": "",
    "Alcohol": "",
    "Social relationships": "",
    "Mental health": "I've also been feeling very stressed at work.",
    "Motivation": "",
    "Hydration": "I haven't been drinking enough water.",
    "Stress management": "I've also been feeling very stressed at work.",
    "Screen time": ""
}

User Input:

“Lately, I’ve been spending too much time on my phone before bed, which keeps me up late. I miss hanging out with friends, and I’ve been skipping my morning runs.”

Your Output:

{
    "Sleep": "I've been spending too much time on my phone before bed, which keeps me up late.",
    "Nutrition": "",
    "Exercise": "I've been skipping my morning runs.",
    "Smoking": "",
    "Alcohol": "",
    "Social relationships": "I miss hanging out with friends.",
    "Mental health": "",
    "Motivation": "",
    "Hydration": "",
    "Stress management": "",
    "Screen time": "I've been spending too much time on my phone before bed.",
}

Instructions:

When processing the user’s input:

	•	Step 1: Read the entire text carefully to understand the user’s statements.
	•	Step 2: Identify any sentences or phrases that relate to the predefined categories.
	•	Step 3: Extract these sentences or phrases exactly as they appear.
	•	Step 4: Assign each extracted piece of information to the relevant category or categories.
	•	Step 5: Construct a JSON object with each category as a key and the extracted text as its value.
	•	Step 6: If a category has no relevant information, assign it an empty string ("").

Notes:

	•	Multiple Categories: If a single piece of information is relevant to more than one category, include it in each applicable category.
	•	Consistency: Ensure that all categories are present in the JSON output, even if their values are empty.
	•	Validation: Double-check the JSON for syntax errors before finalizing your output.

Final Output:

Provide only the JSON object as your final output, without any additional text or explanations.
"""
#%% 
model_program = GenerativeModel("gemini-1.5-pro-002", system_instruction=system_instruction_program)
model_category_completion = GenerativeModel("gemini-1.5-pro-002", system_instruction=system_instruction_category_completion)

#%% 
def extract_dict(input_string):
    # Find the positions of the first and last curly braces
    start_index = input_string.find('{')
    end_index = input_string.rfind('}')
    
    # Return the substring from the first '{' to the last '}'
    if start_index != -1 and end_index != -1 and start_index < end_index:
        return input_string[start_index:end_index + 1]
    else:
        return ""  # Return an empty string if no valid dictionary is found

def generate_program(user_query):
    response = model_program.generate_content([user_query],
    generation_config= {
      "max_output_tokens": 5000,
      "temperature": 0.3,
        "top_p": 0.95,
        "response_mime_type": "application/json"
    },
    )
    return json.loads(extract_dict(str(response.candidates[0].content.parts[0])).replace('\\', ''))

def generate_habits_category(user_query):
    response = model_category_completion.generate_content([user_query],
    generation_config= {
      "max_output_tokens": 5000,
      "temperature": 0.3,
        "top_p": 0.95,
        "response_mime_type": "application/json"
    },
    )
    return json.loads(extract_dict(str(response.candidates[0].content.parts[0])).replace('\\', ''))

def display_program(a):
    for domain, actions in a.items():
        print(f"{domain}:")
        for action, description in actions.items():
            print(f"  - {action}: {description}")

def display_habits_category(a):
    for domain, description in a.items():
        print(f"{domain}: {description}")
#%% 
a = generate_program("I do not have a good hygiene of life. I do not sleep much, and because of my work I cannot sleep more than 6 hours a night. I smoke a lot of cigarettes (I cannot stop smoking). I eat a lot of processed food but I want to start eating healthy.")
#%% 
display_program(a)
# %%
b = generate_habits_category("I do not have a good hygiene of life. I do not sleep much, and because of my work I cannot sleep more than 6 hours a night. I smoke a lot of cigarettes (I cannot stop smoking). I eat a lot of processed food but I want to start eating healthy.")
# %%
display_habits_category(b)
# %%
