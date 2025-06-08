"""
Generate application icon.
"""

import numpy as np
from PIL import Image, ImageDraw

def create_icon(size=256):
    # Create a new image with a white background
    img = Image.new('RGBA', (size, size), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    
    # Draw a camera-like shape
    # Main body
    body_color = (41, 128, 185)  # Blue
    draw.rectangle([size//4, size//4, 3*size//4, 3*size//4], fill=body_color)
    
    # Lens
    lens_color = (52, 152, 219)  # Lighter blue
    draw.ellipse([size//3, size//3, 2*size//3, 2*size//3], fill=lens_color)
    
    # Flash
    flash_color = (241, 196, 15)  # Yellow
    draw.polygon([
        (size//2, size//6),
        (3*size//4, size//4),
        (size//2, size//3)
    ], fill=flash_color)
    
    return img

if __name__ == "__main__":
    # Generate icon in different sizes
    sizes = [16, 32, 48, 64, 128, 256]
    for size in sizes:
        icon = create_icon(size)
        icon.save(f"app_icon_{size}.png")
    
    # Save the main icon
    icon = create_icon(256)
    icon.save("app_icon.png") 