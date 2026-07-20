# Artificial_Intelligence_Ecosystem
Files for AIE course
1. Project Summary
In this project, I set up a Python virtual environment to run an AI image classifier using the pre-trained MobileNetV2 model. I then used Grad-CAM to generate a heatmap to visualize exactly how the AI makes its predictions. In the second part of the lab, I modified a basic image filter script to create a custom "Cel-Shaded" artistic effect that makes photos look like a comic book using the Pillow library.

2. Classifier and Heatmap Findings
I tested the classifier on a photo of colorful, geometric buildings. The AI's top prediction was jigsaw puzzle (31% confidence).

While this seems incorrect at first glance, the Grad-CAM heatmap showed exactly why it guessed that: the heatmap glowed brightly over the dense grid of windows and the sharp, contrasting shapes of the murals. The AI saw the interlocking geometric patterns and sharp edges, and it mathematically mistook them for puzzle pieces.

3. Custom Cel-Shaded Filter
For the second part of the assignment, I built a Cel-Shaded (toon) filter instead of using a standard blur. The Python code achieves this in a few distinct steps:

Color Boost: It increases the saturation and contrast to make the image pop.

Color Simplification: It limits the image to only 24 colors. This turns smooth lighting gradients into flat, cartoon like blocks of color.

Outlines: It uses an edge-detection filter to draw heavy black outlines around the major shapes, which are then overlaid on top of the simplified colors to create a hand-drawn look.

4. AI Collaboration Reflection
Working with the AI made understanding and troubleshooting the Python code much easier. The AI provided line-by-line explanations of how the MobileNetV2 classification script worked behind the scenes. It also helped me quickly debug errors while writing the custom filter, like when I forgot to import the ImageChops module for blending the image layers, and when I ran into a strict Python indentation error that broke the script.
