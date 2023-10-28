import requests
import pickle
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration


class ImageToText:


    def __init__(self, image_filename):
        # Load in the image
        with Image.open(image_filename) as im:
            self.image = im.convert('RGB')

        # Unpickle model and processor
        with open('pkl_files/model.pkl', 'rb') as model_file:
            self.model = pickle.load(model_file)
        with open('pkl_files/processor.pkl', 'rb') as processor_file:
            self.processor = pickle.load(processor_file)



    def create_full_description_for_image(self):
        """
        Creates a full description of an image by iteratively calling create description
        """
        intial_prompt = "the item in the image is "
        description = self.create_description(intial_prompt)

        n_iterations = 2

        for i in range(n_iterations):
            description = description.replace(intial_prompt, '')
            prompt = f"The item in the image is {description} "
            description = self.create_description(prompt)

        return description.replace(intial_prompt, '')


    def create_description(self, prompt):
        """
        Creates a description of an image based on a prompt
        Args:
            prompt: str
        Retuns:
            str
        """
        inputs = self.processor(self.image, prompt, return_tensors="pt")

        out = self.model.generate(**inputs)
        return self.processor.decode(out[0], skip_special_tokens=True)



def main(image_filename):
    # initialize the image to text class
    image_to_text = ImageToText(image_filename)
    # use the class to create a description of the image
    description = image_to_text.create_full_description_for_image()

    return description

    


    



