import vertexai
from vertexai.preview.generative_models import GenerativeModel
import json
from dotenv import load_dotenv
import os

class ProgramGenerator:
    def __init__(self, env_file="../conf.env"):
        load_dotenv(env_file)
        PROJECT_ID = os.getenv('PROJECT_ID')
        REGION = os.getenv('LOCATION')
        vertexai.init(project=PROJECT_ID, location=REGION)

        self.system_instruction_program = open("system_instruction_program.txt", "r").read()
        self.system_instruction_category_completion = open("system_instruction_category_completion.txt", "r").read()

        self.model_program = GenerativeModel("gemini-1.5-pro-002", system_instruction=self.system_instruction_program)
        self.model_category_completion = GenerativeModel("gemini-1.5-pro-002", system_instruction=self.system_instruction_category_completion)

    def extract_dict(self, input_string):
        start_index = input_string.find('{')
        end_index = input_string.rfind('}')
        
        if start_index != -1 and end_index != -1 and start_index < end_index:
            return input_string[start_index:end_index + 1]
        else:
            return ""

    def generate_program(self, user_query):
        response = self.model_program.generate_content([user_query],
        generation_config= {
          "max_output_tokens": 5000,
          "temperature": 0.3,
            "top_p": 0.95,
            "response_mime_type": "application/json"
        },
        )
        return json.loads(self.extract_dict(str(response.candidates[0].content.parts[0])).replace('\\', ''))

    def generate_habits_category(self, user_query):
        response = self.model_category_completion.generate_content([user_query],
        generation_config= {
          "max_output_tokens": 5000,
          "temperature": 0.3,
            "top_p": 0.95,
            "response_mime_type": "application/json"
        },
        )
        return json.loads(self.extract_dict(str(response.candidates[0].content.parts[0])).replace('\\', ''))

    def display_program(self, program):
        for domain, actions in program.items():
            print(f"{domain}:")
            for action, description in actions.items():
                print(f"  - {action}: {description}")

    def display_habits_category(self, habits):
        for domain, description in habits.items():
            print(f"{domain}: {description}")   
  
if __name__ == "__main__":
# Usage example:
  generator = ProgramGenerator()
  program = generator.generate_program("I do not have a good hygiene of life. I do not sleep much, and because of my work I cannot sleep more than 6 hours a night. I smoke a lot of cigarettes (I cannot stop smoking). I eat a lot of processed food but I want to start eating healthy.")
  generator.display_program(program)

  habits = generator.generate_habits_category("I do not have a good hygiene of life. I do not sleep much, and because of my work I cannot sleep more than 6 hours a night. I smoke a lot of cigarettes (I cannot stop smoking). I eat a lot of processed food but I want to start eating healthy.")
  generator.display_habits_category(habits)

# %%
