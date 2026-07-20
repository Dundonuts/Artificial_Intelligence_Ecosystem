from PIL import Image, ImageFilter, ImageEnhance, ImageOps, ImageChops # Adding Enhance/Ops for advanced processing
import matplotlib.pyplot as plt
import os

def apply_cel_shading(image_path, output_path="cel_shaded_image.png"):
    try:
        # Load the original image
        img = Image.open(image_path)
        
        # 1. Resize for performance and consistent style.
        # We increase this to 512x512 so we don't lose all details from the murals.
        img_working = img.resize((512, 512))

        # --- PART A: Simplify Colors (Toon Colors) ---

        # 2. Pre-process color/contrast to help the posterization
        enhancer_sat = ImageEnhance.Color(img_working)
        img_vibrant = enhancer_sat.enhance(2.0) # Boost saturation 2x for pop
        enhancer_con = ImageEnhance.Contrast(img_vibrant)
        img_prepped = enhancer_con.enhance(1.3) # Slight contrast boost

        # 3. Simplify colors (Posterize)
        # We use convert to Palette mode with limited colors to create solid blocks
        # 24 colors is often a good sweet spot for complex images.
        img_quantized = img_prepped.convert("P", palette=Image.ADAPTIVE, colors=24)
        
        # Convert back to RGB so we can work with it again
        toon_colors = img_quantized.convert("RGB")

        # --- PART B: Create Outlines (Linework) ---

        # 4. Generate edges
        # Start with a grayscale version
        img_gray = img_working.convert("L")
        
        # Blur slightly so the outlines are cleaner and less noisy
        img_smoothed = img_gray.filter(ImageFilter.GaussianBlur(radius=2))
        
        # Run standard edge finder (most edges appear white on black)
        edges = img_smoothed.filter(ImageFilter.FIND_EDGES)
        
        # --- PART C: Combine Linework and Colors ---

        # 5. Process edges to be useful outlines (invert and clean)
        # We want black lines, so invert the image (makes lines black, background white)
        outlines = ImageOps.invert(edges)
        
        # Optional: Threshold the edges to make them pure black (1) or pure white (0)
        # This gives cleaner, harder lines for a classic look.
        # threshold = 170 # Adjust if lines are too weak or too noisy
        # outlines = outlines.point(lambda p: p > threshold and 255)

        # 6. Overlay outlines using multiplication
        # Where outlines are black (0), the result is black. Where white (1), color passes.
        # Pillow's Multiply method works best here.
        final_img = ImageChops.multiply(toon_colors, outlines.convert("RGB"))

        # --- PART D: Save and Display ---

        # Use the existing Matplotlib logic to save/display cleanly
        plt.imshow(final_img)
        plt.axis('off') # Hide graph axis
        
        # Save output using high-quality parameters
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close() # Free up memory
        print(f"Cel-shaded image successfully saved as '{output_path}'.")

    except Exception as e:
        print(f"Error processing image: {e}")

if __name__ == "__main__":
    print("Cel-Shading Processor (type 'exit' to quit)\n")
    while True:
        # Get filename
        image_path = input("Enter image filename (or 'exit' to quit): ").strip()
        
        # Check for exit
        if image_path.lower() == 'exit':
            print("Goodbye!")
            break
            
        # Check if file exists
        if not os.path.isfile(image_path):
            print(f"File not found: {image_path}")
            continue
            
        # Derive output filename based on original extension (keeps jpg vs png)
        base, ext = os.path.splitext(image_path)
        output_file = f"{base}_cel_shaded{ext}"
        
        # Call the new filter function
        apply_cel_shading(image_path, output_file)