from PIL import Image
import os
import numpy as np
import svgwrite
#import cairosvg
from wand.image import Image as WandImage


def draw_protractor(filename, center, radius, start_angle, end_angle, scale=1, im_w=1000, im_h=1000):
    dwg = svgwrite.Drawing(filename, profile='tiny',size=(im_w, im_h))

    # Function to draw a semi-circle protractor
    def draw_semi_protractor(offset):
        # Draw degree lines and labels
        for angle in range(start_angle, end_angle+1):
            # Calculate line start and end points
            start_x = center[0] + radius * np.cos(np.radians(angle + offset))
            start_y = center[1] + radius * np.sin(np.radians(angle + offset))
            end_x = center[0] + (radius - 20*scale) * np.cos(np.radians(angle + offset))
            end_y = center[1] + (radius - 20*scale) * np.sin(np.radians(angle + offset))

            # Draw line
            dwg.add(dwg.line(start=(start_x, start_y), end=(end_x, end_y), stroke='black', stroke_width=scale))

            # Draw label and longer line for every fifth degree
            if angle % 10 == 0:
                label_x = center[0] + (radius + 30*scale) * np.cos(np.radians(angle + offset))
                label_y = center[1] + (radius + 30*scale) * np.sin(np.radians(angle + offset))
                dwg.add(dwg.text(str(angle), insert=(label_x, label_y), fill='black', text_anchor='middle', font_size=35*scale))

                # Draw longer line
                long_end_x = center[0] + (radius - 360*scale) * np.cos(np.radians(angle + offset)) 
                long_end_y = center[1] + (radius - 360*scale) * np.sin(np.radians(angle + offset)) 
                dwg.add(dwg.line(start=(start_x, start_y), end=(long_end_x, long_end_y), stroke='blue', stroke_width=scale))
            elif angle % 5 == 0:
                # Draw longer line
                long_end_x = center[0] + (radius - 310*scale) * np.cos(np.radians(angle + offset)) 
                long_end_y = center[1] + (radius - 310*scale) * np.sin(np.radians(angle + offset)) 
                dwg.add(dwg.line(start=(start_x, start_y), end=(long_end_x, long_end_y), stroke='blue', stroke_width=scale))

    # Draw two semi-circle protractors with an offset of 180 degrees
    draw_semi_protractor(0)
    draw_semi_protractor(180)

    
    # Draw center cross
    cross_size = 15*scale
    dwg.add(dwg.line(start=(center[0] - cross_size, center[1]), end=(center[0] + cross_size, center[1]), stroke='red', stroke_width=scale))
    dwg.add(dwg.line(start=(center[0], center[1] - cross_size), end=(center[0], center[1] + cross_size), stroke='red', stroke_width=scale))
    dwg.add(dwg.circle(center=center, r=4*scale, fill='red'))

    dwg.save()


def apply_overlay(image_directory, output_directory):
    

    # Check and create output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Iterate through all .png files in the directory
    for filename in os.listdir(image_directory):
        if filename.endswith(".JPG"):
            # Open the image
            img_path = os.path.join(image_directory, filename)
            img = Image.open(img_path).convert("RGBA")
            
            image_width, image_height = img.size
            center = (image_width/2 + 69, image_height/2 + 10) #Adjust to the center of rotor disc
            
            draw_protractor('protractor_overlay.svg', center, image_width/2.35, -30, 30, scale=6, im_w=image_width, im_h=image_height)
            
            # Convert SVG to PNG using Wand
            with WandImage(filename='./protractor_overlay.svg') as svg_img:
                svg_img.format = 'png'
                svg_img.save(filename='protractor_overlay.png')
            
            # Convert SVG to PNG
            #cairosvg.svg2png(url='protractor_overlay.svg', write_to='protractor_overlay.png')
            
            # Load the overlay image
            # Load the overlay image
            overlay = Image.open('protractor_overlay.png').convert("RGBA")
            
            # Make sure the background of the overlay is transparent
            data = np.array(overlay)
            red, green, blue, alpha = data.T
            white_areas = (red == 255) & (blue == 255) & (green == 255)
            data[..., :-1][white_areas.T] = (255, 255, 255)  # Any color (except white)
            data[..., -1][white_areas.T] = 0  # Transparent
            overlay = Image.fromarray(data)

            # Apply the overlay
            combined = Image.alpha_composite(img, overlay)

            # Save the image to the output directory
            combined = combined.convert("RGB")  # Convert back to RGB to save as PNG
            combined.save(os.path.join(output_directory, filename))

# Usage
image_directory = '.'  # Replace with the path to your image directory
#overlay_path = './overlay_centered.png'  # Replace with the path to your overlay image
output_directory = 'with_overlay'  # Replace with the path to your output directory
apply_overlay(image_directory, output_directory)