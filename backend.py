from fastapi import FastAPI, HTTPException
from image_generation.generate_replicate import ImageGenerator
from model.request_schema import ImageGenerationRequest, ImageGenerationResponse, ProgramRequest
from program_creation.program_creation import ProgramGenerator
import uvicorn

app = FastAPI()

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


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)


