# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 11:59:19 2021

@author: droes
"""

# Force matplotlib's headless Agg backend BEFORE anything imports pyplot.
# On macOS the default backend opens a native window and retains a graphics
# buffer on every redraw -> unbounded memory growth (the 45 GB leak).
import matplotlib
matplotlib.use('Agg')
from object_detection import ObjectDetector

# You can use this library for oberserving keyboard presses
import keyboard # pip install keyboard

from capturing import VirtualCamera
from overlays import initialize_hist_figure, plot_overlay_to_image, plot_strings_to_image, update_histogram
from basics import (
    histogram_figure_numba,
    image_stats,
    equalize_image,
    linear_transform,
    entropy_per_channel,
    gaussian_blur,
    sobel_edges
)


# The `keyboard` package doesn't work on macOS (needs root, and its darwin
# backend can't map character keys -> "Unrecognized character" crash).
# We replace keyboard.is_pressed on the shared module object with a pynput
# implementation. Both run.py (line 31) and capturing.py (line 70) call
# keyboard.is_pressed(...), so both transparently use this working version.
from pynput import keyboard as _pk

_pressed_keys: set[str] = set()

def _on_press(key) -> None:
    # key.char exists only for character keys; special keys (shift, etc.) lack it
    char = getattr(key, "char", None)
    if char is not None:
        _pressed_keys.add(char.lower())

def _on_release(key) -> None:
    char = getattr(key, "char", None)
    if char is not None:
        _pressed_keys.discard(char.lower())

# daemon listener -> runs in the background, dies automatically with the program
_pk.Listener(on_press=_on_press, on_release=_on_release, daemon=True).start()

# Overwrite the broken function. capturing.py's keyboard.is_pressed('q') now
# resolves to this at call time, so the original file never needs modifying.
keyboard.is_pressed = lambda hotkey: hotkey.lower() in _pressed_keys



# Example function
# You can use this function to process the images from opencv
# This function must be implemented as a generator function
def custom_processing(img_source_generator):
    # use this figure to plot your histogram (created ONCE, outside the loop)
    fig, ax, background, r_plot, g_plot, b_plot = initialize_hist_figure()
    
    #Creating the detector object from our object_detection class.
    detector = ObjectDetector(confidence=0.5)


    # Equalization toggle + a small debounce counter so a single key tap
    #doesn't flip the state on every frame (high fps => key held many frames).
    apply_equalization = False
    apply_linear_transform = False
    show_entropy = False
    apply_gaussian_blur = False
    apply_sobel = False
    apply_object_detection = True
    cooldown = 0
    
    for sequence in img_source_generator:
        # Call your custom processing methods here! (e. g. filters)

        # key handling: press 'h' to toggle histogram equalization
        if cooldown > 0:
            cooldown -= 1
        if cooldown == 0:
            if keyboard.is_pressed('h'):
                apply_equalization = not apply_equalization
                cooldown = 15
                
            elif keyboard.is_pressed('l'):
                apply_linear_transform = not apply_linear_transform
                cooldown = 15
            
            elif keyboard.is_pressed('e'):
                show_entropy = not show_entropy
                cooldown = 15
                
            elif keyboard.is_pressed('g'):
                apply_gaussian_blur = not apply_gaussian_blur
                cooldown = 15
                
            elif keyboard.is_pressed('s'):
                apply_sobel = not apply_sobel
                cooldown = 15
                
            elif keyboard.is_pressed('o'):
                apply_object_detection = not apply_object_detection
                cooldown = 15

        # optional equalization (runs first, so stats/histogram reflect it)
        if apply_equalization:
            sequence = equalize_image(sequence)
        if apply_linear_transform:
            sequence = linear_transform(sequence, a=1.3, b=25)
        if apply_gaussian_blur:
            sequence = gaussian_blur(sequence, kernel_size=9)
        if apply_sobel:
            sequence = sobel_edges(sequence)

        # Object detection can optionally be toggled as YOLO may slow down the demo
        object_count = 0
        if apply_object_detection:
            sequence, object_count = detector.detect_objects(sequence)

        # per channel stats after selected processing, before overlays
        display_text_arr = image_stats(sequence)
        if show_entropy:
            display_text_arr.extend(entropy_per_channel(sequence))
            
        display_text_arr.append(f"Objects detected: {object_count}")
        display_text_arr.append('h: Equalization ' + ('ON' if apply_equalization else 'OFF'))
        display_text_arr.append('l: Linear Transform ' + ('ON' if apply_linear_transform else 'OFF'))
        display_text_arr.append('e: Show Entropy ' + ('ON' if show_entropy else 'OFF'))
        display_text_arr.append('g: Gaussian Blur ' + ('ON' if apply_gaussian_blur else 'OFF'))
        display_text_arr.append('s: Sobel Edges ' + ('ON' if apply_sobel else 'OFF'))
        display_text_arr.append('o: Object Detection ' + ('ON' if apply_object_detection else 'OFF'))
        ###
        ### Histogram overlay example (without data)
        ###
        
        # Load the histogram values
        r_bars, g_bars, b_bars = histogram_figure_numba(sequence)        
        
        # Update the histogram with new data
        update_histogram(fig, ax, background, r_plot, g_plot, b_plot, r_bars, g_bars, b_bars)
        
        # uses the figure to create the overlay
        sequence = plot_overlay_to_image(sequence, fig)
        
        ###
        ### END Histogram overlay example
        ###

        
        # Display text example
        sequence = plot_strings_to_image(sequence, display_text_arr, right_space=750)

        
        # Make sure to yield your processed image
        yield sequence



def main():
    # change according to your settings
    width = 1280
    height = 720
    fps = 30
    
    # Define your virtual camera
    vc = VirtualCamera(fps, width, height)
    
    vc.virtual_cam_interaction(
        custom_processing(
            # either camera stream
            vc.capture_cv_video(0, bgr_to_rgb=True)
            
            # or your window screen
            # vc.capture_screen()
        )
    )

if __name__ == "__main__":
    main()