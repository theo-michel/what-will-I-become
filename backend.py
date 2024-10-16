from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from image_generation.generate_replicate import ImageGenerator
from program_creation.program_creation import ProgramGenerator
from model.request_schema import ImageGenerationRequest, ImageGenerationResponse, ProgramRequest, SimulateLifeRequest
from simulate_life.simulate_life import LifeSimulator
import uvicorn

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Add your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/generate-image", response_model=ImageGenerationResponse)
async def generate_image(request: ImageGenerationRequest):
    # Initialize the ImageGenerator
    generator = ImageGenerator()
    try:
        saved_images = generator.generate_image(
            prompt=request.prompt,
            input_images_path=request.input_images_path,
            num_steps=request.num_steps,
            negative_prompt=request.negative_prompt,
            output_directory=request.output_directory
        )
        return ImageGenerationResponse(saved_images=saved_images)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate-program")
async def generate_program(request: ProgramRequest):
    try:
        program_generator = ProgramGenerator()
        program = program_generator.generate_program(request.user_query)
        return {"program": program}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-habits-category")
async def generate_habits_category(request: ProgramRequest):
    try:
        program_generator = ProgramGenerator()
        habits = program_generator.generate_habits_category(request.user_query)
        return {"habits": habits}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/simulate-life")
async def simulate_life(request: SimulateLifeRequest):
    try:
        life_simulator = LifeSimulator()
        life_simulation = life_simulator.get_evolution_given_program(
                request.initial_state, request.program, request.time_horizon
        )
        return {"life_simulation": life_simulation}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
