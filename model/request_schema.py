from pydantic import BaseModel
from typing import List, Optional

class ImageGenerationRequest(BaseModel):
    prompt: str
    input_images_path: str
    num_steps: int = 50
    negative_prompt: Optional[str] = None
    output_directory: str = "output_images"

class ImageGenerationResponse(BaseModel):
    saved_images: List[str]

class ProgramRequest(BaseModel):
    user_query: str

class SimulateLifeRequest(BaseModel):
    initial_state: str
    program: str
    time_horizon: int

    class Config:
        json_schema_extra = {
            "example": {
                "initial_state": "I do not have a good hygiene of life. I do not sleep much, and because of my work I cannot sleep more than 6 hours a night. I smoke a lot of cigarettes (I cannot stop smoking). I eat a lot of processed food but I want to start eating healthy.",
                "program": """Lifestyle
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
    Regular physical activity: Incorporate regular physical activity into your routine, even if it's just a short walk or some stretching. Exercise can improve sleep quality and reduce stress.""",
                "time_horizon": 15
            }
        }