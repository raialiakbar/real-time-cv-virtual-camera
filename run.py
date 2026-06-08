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

# You can use this library for oberserving keyboard presses
import keyboard # pip install keyboard

from capturing import VirtualCamera
from overlays import initialize_hist_figure, plot_overlay_to_image, plot_strings_to_image, update_histogram
from basics import histogram_figure_numba,image_stats, equalize_image


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

    # Equalization toggle + a small debounce counter so a single key tap
    #doesn't flip the state on every frame (high fps => key held many frames).
    apply_equalization = False
    cooldown = 0
    
    for sequence in img_source_generator:
        # Call your custom processing methods here! (e. g. filters)

        # key handling: press 'h' to toggle histogram equalization
        if cooldown > 0:
            cooldown -= 1
        if cooldown == 0 and keyboard.is_pressed('h'):
            apply_equalization = not apply_equalization
            cooldown = 15  # ignore further presses for ~15 frames (~0.5 s)

        # optional equalization (runs first, so stats/histogram reflect it)
        if apply_equalization:
            sequence = equalize_image(sequence)

        # per-channel statistics on the clean image (before overlays)
        display_text_arr = image_stats(sequence)
        display_text_arr.append('EQ ON' if apply_equalization else 'EQ OFF')
            

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