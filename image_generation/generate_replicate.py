import replicate
import base64
from dotenv import load_dotenv
import os
import requests
from PIL import Image
from io import BytesIO
from typing import List, Optional

class ImageGenerator:
    def __init__(self, env_file: str = 'conf.env'):
        # Load the environment variables
        load_dotenv(env_file)
        
        # Set up the Replicate API token
        os.environ["REPLICATE_API_TOKEN"] = os.getenv('REPLICATE_API_TOKEN')
        
        # Initialize the Replicate client with a longer timeout
        self.client = replicate.Client(api_token=os.environ["REPLICATE_API_TOKEN"], timeout=300)  # 5 minutes timeout

    def generate_image(self, 
                       prompt: str, 
                       input_images_path: str,
                       num_steps: int = 50, 
                       negative_prompt: Optional[str] = None) -> List[str]:
        # Prepare input dictionary
        input_data = {
            "prompt": prompt,
            "num_steps": num_steps,
        }

        if negative_prompt:
            input_data["negative_prompt"] = negative_prompt
        else:
            input_data["negative_prompt"] = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry"

        # Get the first 4 images from the input directory
        input_images = [os.path.join(input_images_path, f) for f in os.listdir(input_images_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))][:4]

        # Add images to input dictionary (up to 4)
        for i, image_path in enumerate(input_images, start=1):
            with open(image_path, 'rb') as file:
                data = base64.b64encode(file.read()).decode('utf-8')
                image = f"data:application/octet-stream;base64,{data}"
                input_data[f"input_image{'' if i == 1 else i}"] = image

        # Run the model
        output = self.client.run(
            "tencentarc/photomaker:ddfc2b08d209f9fa8c1eca692712918bd449f695dabb4a958da31802a9570fe4",
            input=input_data
        )

        base64_images = []

        # Download and convert each output image to base64
        for image_url in output:
            response = requests.get(image_url)
            if response.status_code == 200:
                img = Image.open(BytesIO(response.content))
                buffered = BytesIO()
                img.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                base64_images.append(img_str)
                print(f"Processed image to base64")
            else:
                print(f"Failed to download image")

        print("All images have been processed to base64.")
        return base64_images

# Example usage:
if __name__ == "__main__":
  generator = ImageGenerator("../conf.env")
  base64_images = generator.generate_image(
      prompt="A photo of a happy and fit man img",
      input_images_path="input_images"
  )

  print(base64_images)