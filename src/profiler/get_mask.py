from pathlib import Path
from PIL import Image, ImageDraw
import random, json

# Hover net json
def read_coordinates_from_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)

    nuc_coords = data["nuc"]
    all_nuc_outline_coords = []

    for key in nuc_coords.keys():
        coords = []
        for x,y in nuc_coords[key]["contour"]:
            coords.append((x,y))

        all_nuc_outline_coords.append(coords)

    return all_nuc_outline_coords

def generate_mask(img_path, dir_hovernet):

    img = Image.open(img_path)

    file_name = Path(img_path).stem
    
    #outline_coords = read_coordinates_from_json(f"/app_data/hovernet/outputs/json/{file_name}.json")
    outline_coords = read_coordinates_from_json(f"{dir_hovernet}/json/{file_name}.json")

    # Create a blank image with a white background
    mask_img = Image.new("RGB", img.size, (0,0,0))
    draw = ImageDraw.Draw(mask_img)


    for outline_coord in outline_coords:
        color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
        # Draw the outline on the image
        draw.polygon(outline_coord, outline=color, fill=color)
    
    #mask_img.save(f"/app_data/hovernet/outputs/mask/{file_name}.png")  # Save the image to a file
    mask_img.save(f"{dir_hovernet}/mask/{file_name}.png")  # Save the image to a file

    return mask_img
