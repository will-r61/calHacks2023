from image_to_text.image_to_text import main as get_image_description

def main():
    user_input_type = input("Are you providing an image or a text description of your object?")
    if user_input_type == "image":
        image_filename = input("Please provide the filename of your image:")
        description = get_image_description(image_filename)
    else:
        description = input("Please provide a description of your object:")
    return description