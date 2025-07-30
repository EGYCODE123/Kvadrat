#!/usr/bin/env python3
"""
Kvadrat Shade Fabric Simulation Automator - Complete Industrial Version
Author: Aly Marzouk
Description: Full-featured GUI tool for industrial-scale simulation control in Plataine
Features: Image-based automation, dark mode, preview systems, export functionality, retry mechanisms
Requires: pyautogui, opencv-python, tkinter, pandas, matplotlib, numpy, tqdm, pillow, pygetwindow
"""

import os
import sys
import time
import logging
import threading
import subprocess
import re
import shutil
import json
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button, TextBox
import ezdxf
from tqdm import tqdm
import traceback

# Auto-install missing packages
def install_missing_packages():
    import importlib
    required = ["pyautogui", "opencv-python", "pandas", "matplotlib", 
                "numpy", "tqdm", "pillow", "pygetwindow", "openpyxl", "ezdxf"]
    for pkg in required:
        try:
            importlib.import_module(pkg.replace("-", "_"))
        except ImportError:
            print(f"Installing {pkg}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

install_missing_packages()

import pyautogui
import pygetwindow as gw

def save_plan_with_tabs(plan_name):
    """
    Helper function to save a plan using tab-based keyboard navigation with image recognition.
    
    Args:
        plan_name (str): The name of the plan to save (simulation number)
    """
    def log(message, level="INFO"):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] [{level}] {message}")
    
    try:
        log(f"Starting save process for plan: {plan_name}")
        
        # 1. First, try to find the save dialog using image recognition
        save_dialog_img = os.path.join("images", "save_dialog.png")
        save_button_img = os.path.join("images", "save_button.png")
        
        log(f"Looking for save dialog using image: {save_dialog_img}")
        
        # Wait for save dialog to appear
        max_wait = 10  # seconds
        start_time = time.time()
        save_dialog = None
        
        while time.time() - start_time < max_wait:
            try:
                save_dialog = pyautogui.locateOnScreen(save_dialog_img, confidence=0.8)
                if save_dialog:
                    log(f"Save dialog found at position: {save_dialog}")
                    break
            except Exception as e:
                log(f"Error while looking for save dialog: {e}", "WARNING")
            
            time.sleep(0.5)
        
        if not save_dialog:
            log("Save dialog not found after waiting. Attempting to proceed anyway...", "WARNING")
        
        # 2. Ensure the save dialog has focus by clicking on it
        if save_dialog:
            dialog_center = pyautogui.center(save_dialog)
            pyautogui.click(dialog_center)
            log(f"Clicked on save dialog at {dialog_center}")
            time.sleep(0.5)
        
        # 3. Clear any existing text in the filename field
        log("Clearing filename field...")
        pyautogui.hotkey('ctrl', 'a')
        pyautogui.press('delete')
        time.sleep(0.5)
        
        # 4. Type the plan name
        log(f"Typing plan name: {plan_name}")
        pyautogui.typewrite(str(plan_name), interval=0.1)
        time.sleep(0.5)
        
        # 5. Try to find and click the save button using image recognition
        log(f"Looking for save button using image: {save_button_img}")
        save_button = None
        max_attempts = 3
        
        for attempt in range(1, max_attempts + 1):
            try:
                save_button = pyautogui.locateCenterOnScreen(save_button_img, confidence=0.8)
                if save_button:
                    log(f"Save button found at position: {save_button}")
                    pyautogui.click(save_button)
                    log("Clicked save button")
                    break
                else:
                    log(f"Save button not found (attempt {attempt}/{max_attempts})", "WARNING")
                    
                    # If we can't find the save button, try tabbing to it
                    log("Attempting to tab to save button...")
                    pyautogui.press('tab', presses=3, interval=0.3)
                    time.sleep(0.5)
                    
            except Exception as e:
                log(f"Error while looking for save button: {e}", "ERROR")
        
        # 6. If we still couldn't find the save button, try pressing Enter as fallback
        if not save_button:
            log("Save button not found after multiple attempts. Trying Enter key as fallback...", "WARNING")
            pyautogui.press('enter')
        
        # 7. Wait for the save operation to complete
        time.sleep(2.0)
        
        # 8. Check if save was successful by looking for the save dialog to disappear
        save_complete = False
        for _ in range(5):
            if not pyautogui.locateOnScreen(save_dialog_img, confidence=0.8):
                save_complete = True
                break
            time.sleep(0.5)
        
        if save_complete:
            log("Save operation completed successfully")
        else:
            log("Warning: Save dialog still visible after save attempt", "WARNING")
        
        return save_complete
        
    except Exception as e:
        log(f"Critical error in save_plan_with_tabs: {str(e)}", "ERROR")
        log(f"Stack trace: {traceback.format_exc()}", "ERROR")
        return False

# Configure pyautogui
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.1

# Logging setup
LOG_PATH = "simulation_log.txt"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler()
    ]
)

import os
script_dir = os.path.dirname(__file__)           # if your script lives in Sim_App/
images_dir = os.path.join(script_dir, "images")
for img in ("save_dialog.png", "save_button.png", "ok_button.png"):
    path = os.path.join(images_dir, img)
    print(f"{img:15s} →", "FOUND" if os.path.exists(path) else "MISSING")


# Constants
TEMPLATE_IMAGES = {
    "load_library": "images/load_from_library.png",
    "by_steps": "images/by_steps.png", 
    "run_button": "images/run_button.png",
    "filter_field": "images/filter_field.png",
    "from_field": "images/from_field.png",
    "to_field": "images/to_field.png",
    "step_field": "images/step_field.png",
    "qmix_from": "images/qmix_from.png",
    "qmix_to": "images/qmix_to.png",
    "qmix_steps": "images/qmix_steps.png",
    "ok_button": "images/ok_button.png",
    "order_entry": "images/order_entry.png"
}

# Color schemes
KVADRAT_PRIMARY = '#3498db'
KVADRAT_SECONDARY = '#e74c3c'
KVADRAT_SUCCESS = '#27ae60'
KVADRAT_WARNING = '#f39c12'
KVADRAT_DARK = '#2c3e50'
KVADRAT_LIGHT = '#ecf0f1'

LIGHT_THEME = {
    'bg': '#f8f8ff',
    'fg': '#2c3e50',
    'accent': KVADRAT_PRIMARY,
    'secondary': KVADRAT_SECONDARY,
    'success': KVADRAT_SUCCESS,
    'warning': KVADRAT_WARNING,
    'button_bg': KVADRAT_PRIMARY,
    'button_fg': 'white',
    'entry_bg': 'white',
    'text_bg': 'white'
}

DARK_THEME = {
    'bg': KVADRAT_DARK,
    'fg': KVADRAT_LIGHT,
    'accent': KVADRAT_PRIMARY,
    'secondary': KVADRAT_SECONDARY,
    'success': KVADRAT_SUCCESS,
    'warning': KVADRAT_WARNING,
    'button_bg': '#34495e',
    'button_fg': KVADRAT_LIGHT,
    'entry_bg': '#34495e',
    'text_bg': '#34495e'
}

TITLE_FONT = ('Arial', 16, 'bold')

class PlataineSimulator:
    """Core simulation automation class with image recognition and error handling"""
    
    def _save_csv_method_direct(self, output_path):
        """Method 1: Direct typing of path"""
        pyautogui.typewrite(output_path, interval=0.05)
        time.sleep(0.5)
        pyautogui.press("enter")
        return True
    
    def _save_csv_method_saveas(self, output_path):
        """Method 2: Use Save As dialog"""
        pyautogui.hotkey('alt', 'f')
        time.sleep(0.5)
        pyautogui.press('a')
        time.sleep(1)
        pyautogui.typewrite(output_path, interval=0.05)
        time.sleep(0.5)
        pyautogui.press("enter")
        return True
    
    def _save_csv_method_clipboard(self, output_path):
        """Method 3: Use clipboard for path"""
        pyperclip.copy(output_path)
        pyautogui.hotkey('ctrl', 'v')
        time.sleep(0.5)
        pyautogui.press("enter")
        return True
    
    def __init__(self, dxf_folder, output_folder, width_min, width_max, width_step, retry_count=3):
        self.dxf_folder = dxf_folder
        self.output_folder = output_folder
        self.width_min = width_min
        self.width_max = width_max
        self.width_step = width_step
        self.retry_count = retry_count
        self.results = []
        self.failed_orders = []
        self.paused = False
        self.stopped = False
        
    def log(self, msg, level="INFO"):
        """Enhanced logging with levels"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_msg = f"[{timestamp}] {msg}"
        print(formatted_msg)
        
        if level == "ERROR":
            logging.error(msg)
        elif level == "WARNING":
            logging.warning(msg)
        else:
            logging.info(msg)
    
    def wait_for_window(self, title, timeout=60):
        """Wait for a window with specific title to appear"""
        start = time.time()
        while time.time() - start < timeout:
            if self.stopped:
                raise InterruptedError("Simulation stopped by user")
            for w in gw.getAllTitles():
                if title.lower() in w.lower():
                    win = gw.getWindowsWithTitle(w)[0]
                    win.activate()
                    win.maximize()
                    return win
            time.sleep(0.5)
        raise TimeoutError(f"Window '{title}' not found in {timeout}s.")
    
    def wait_for_image(self, image_path, timeout=20, confidence=0.8, region=None):
        """Wait for an image to appear on screen with retry logic and fallback to pixel-based positioning"""
        start = time.time()
        last_error = None
        
        # Try with decreasing confidence levels
        conf_levels = [confidence, max(0.6, confidence - 0.1), max(0.5, confidence - 0.2)]
        
        for conf in conf_levels:
            while time.time() - start < timeout:
                if self.stopped:
                    raise InterruptedError("Simulation stopped by user")
                try:
                    location = pyautogui.locateCenterOnScreen(
                        image_path, 
                        confidence=conf,
                        region=region
                    )
                    if location:
                        return location
                except Exception as e:
                    last_error = str(e)
                time.sleep(0.5)
        
        # If we get here, image wasn't found - log the error and try fallback positions
        self.log(f"Warning: Could not locate image '{image_path}' with confidence {confidence}. Last error: {last_error}", "WARNING")
        
        # Try fallback positions if available
        fallback_positions = self._get_fallback_positions(image_path)
        if fallback_positions:
            self.log(f"Trying {len(fallback_positions)} fallback positions for {image_path}", "WARNING")
            for pos in fallback_positions:
                try:
                    pyautogui.moveTo(pos[0], pos[1], duration=0.2)
                    pyautogui.click()
                    time.sleep(0.5)
                    return pos
                except Exception as e:
                    self.log(f"Fallback position failed: {e}", "WARNING")
        
        raise TimeoutError(f"Image '{image_path}' not found in {timeout}s with confidence {confidence}.")
    
    def _get_fallback_positions(self, image_name):
        """Get fallback pixel positions for critical UI elements"""
        fallbacks = {
            "save_dialog.png": [(100, 100), (500, 300)],  # Common save dialog positions
            "run_button.png": [(100, 100)],
            "ok_button.png": [(500, 400)],
        }
        
        # Extract just the filename for matching
        filename = os.path.basename(image_name)
        return fallbacks.get(filename, [])
    
    def click_and_type(self, image_path, value, clear=True, timeout=10, retries=3):
        """Click on image location and type value with error handling and retries"""
        last_exception = None
        
        for attempt in range(retries):
            try:
                # Try to find and click the target
                loc = self.wait_for_image(image_path, timeout)
                if not loc:
                    raise Exception("Image not found")
                
                # Move to the location with some randomness to avoid detection
                x, y = loc
                pyautogui.moveTo(
                    x + random.randint(-2, 2),
                    y + random.randint(-2, 2),
                    duration=random.uniform(0.1, 0.3)
                )
                pyautogui.click()
                time.sleep(random.uniform(0.1, 0.3))
                
                # Clear field if needed
                if clear:
                    pyautogui.hotkey("ctrl", "a")
                    time.sleep(0.1)
                    pyautogui.press("backspace")
                    time.sleep(0.1)
                
                # Type the value with human-like delays
                for char in str(value):
                    pyautogui.typewrite(char, interval=random.uniform(0.03, 0.1))
                    if random.random() < 0.1:  # 10% chance of a tiny pause
                        time.sleep(random.uniform(0.05, 0.2))
                
                time.sleep(0.2)
                return True
                
            except Exception as e:
                last_exception = e
                self.log(f"Attempt {attempt + 1}/{retries} failed for {image_path}: {e}", "WARNING")
                if attempt < retries - 1:
                    time.sleep(1)  # Wait before retry
        
        self.log(f"All {retries} attempts failed for {image_path}: {last_exception}", "ERROR")
        return False
    
    def extract_vnumber(self, filename):
        """Extract V-number from filename"""
        match = re.search(r"V\d+", filename, re.IGNORECASE)
        return match.group(0) if match else None
    
    def handle_save_dialog(self, filename, default_dir=None, timeout=30):
        """
        Handle Windows save dialog with robust detection and typing.
        Includes multiple fallback methods and detailed diagnostics.
        """
        script_dir = os.path.dirname(os.path.abspath(__file__))
        save_dialog_img = os.path.join(script_dir, "images", "save_dialog.png")
        ok_button_img = os.path.join(script_dir, "images", "ok_button.png")
        
        print(f"[DEBUG] Looking for save dialog with images:\n  - {save_dialog_img}\n  - {ok_button_img}")
        
        # Ensure images exist
        for img_path in [save_dialog_img, ok_button_img]:
            if not os.path.exists(img_path):
                print(f"[WARNING] Image not found: {img_path}")
        
        start_time = time.time()
        w, h = pyautogui.size()
        search_region = (w//2 - 400, h//2 - 200, 800, 400)  # Center of screen
        
        print(f"[DEBUG] Screen size: {w}x{h}, Search region: {search_region}")
        
        while time.time() - start_time < timeout:
            if self.stopped:
                raise InterruptedError("Simulation stopped by user")
                
            try:
                # Method 1: Try to find save dialog by image with multiple approaches
                print("[DEBUG] Attempting to locate save dialog...")
                
                # Try with grayscale and lower confidence
                save_loc = None
                for confidence in [0.7, 0.6, 0.5]:
                    print(f"[DEBUG] Trying with confidence: {confidence}")
                    
                    # Try with region restriction
                    try:
                        save_loc = pyautogui.locateCenterOnScreen(
                            save_dialog_img,
                            confidence=confidence,
                            region=search_region,
                            grayscale=True
                        )
                        if save_loc:
                            print(f"[SUCCESS] Found save dialog at {save_loc} with confidence {confidence}")
                            break
                    except Exception as e:
                        print(f"[DEBUG] Error with region search: {e}")
                
                # If dialog found, handle it
                if save_loc:
                    print("[DEBUG] Save dialog found, attempting to save...")
                    
                    # Type the full path
                    full_path = os.path.join(default_dir, filename) if default_dir else filename
                    print(f"[DEBUG] Typing save path: {full_path}")
                    
                    # Try to activate the window first
                    try:
                        for window in gw.getWindowsWithTitle('Save As'):
                            window.activate()
                            time.sleep(0.5)
                    except Exception as e:
                        print(f"[DEBUG] Could not activate window: {e}")
                    
                    # Type the filename with some delay
                    pyautogui.typewrite(full_path, interval=0.1)
                    time.sleep(0.5)
                    
                    # Try to find and click OK button
                    ok_clicked = False
                    for confidence in [0.7, 0.6, 0.5]:
                        try:
                            ok_btn = pyautogui.locateCenterOnScreen(
                                ok_button_img,
                                confidence=confidence,
                                region=search_region,
                                grayscale=True
                            )
                            if ok_btn:
                                print(f"[DEBUG] Found OK button at {ok_btn} with confidence {confidence}")
                                pyautogui.click(ok_btn)
                                ok_clicked = True
                                time.sleep(1)
                                return True
                        except Exception as e:
                            print(f"[DEBUG] Error finding OK button: {e}")
                    
                    # If OK button not found, try pressing Enter
                    if not ok_clicked:
                        print("[DEBUG] OK button not found, trying Enter key")
                        pyautogui.press('enter')
                        time.sleep(1)
                        return True
                
                # Method 2: Try direct keystroke fallback
                print("[DEBUG] Trying direct keystroke fallback...")
                try:
                    pyautogui.hotkey('alt', 'n')  # Focus filename field
                    time.sleep(0.5)
                    full_path = os.path.join(default_dir, filename) if default_dir else filename
                    pyautogui.typewrite(full_path, interval=0.1)
                    time.sleep(0.5)
                    pyautogui.press("enter")
                    print("[SUCCESS] Used direct keystroke fallback")
                    time.sleep(1)
                    return True
                except Exception as e:
                    print(f"[DEBUG] Direct keystroke fallback failed: {e}")
                
                # Method 3: Try pywinauto if available
                try:
                    from pywinauto import Application
                    app = Application().connect(path="TPO FabricOptimizer.exe")
                    dlg = app.window(title_re="Choose Simulation Plan file name")
                    dlg.Edit1.set_edit_text(filename)
                    dlg.Button1.click()
                    print("[SUCCESS] Used pywinauto for save dialog")
                    return True
                except Exception as e:
                    print(f"[DEBUG] pywinauto fallback failed: {e}")
                
            except Exception as e:
                print(f"[ERROR] Error in save dialog handling: {e}")
            
            # Wait before retrying
            time.sleep(1)
        
        raise TimeoutError(f"Could not handle save dialog within {timeout} seconds")

    def simulate_one_order(self, vnumber, index, progress_callback=None):
        """Simulate a single order with comprehensive error handling and retry logic"""
        # Create a state file for this order to track progress
        state_file = os.path.join(self.output_folder, f"{vnumber}_state.json")
        
        # Try to load previous state if exists
        state = {
            'attempts': 0,
            'completed_steps': [],
            'last_error': None,
            'start_time': time.time(),
            'output_path': os.path.join(self.output_folder, f"{vnumber}.csv")
        }
        
        if os.path.exists(state_file):
            try:
                with open(state_file, 'r') as f:
                    saved_state = json.load(f)
                    state.update(saved_state)
                self.log(f"Resuming {vnumber} from previous state (attempt {state['attempts'] + 1})", "INFO")
            except Exception as e:
                self.log(f"Error loading state for {vnumber}: {e}", "WARNING")
        
        # Skip if already completed
        if os.path.exists(state['output_path']) and os.path.getsize(state['output_path']) > 0:
            self.log(f"Skipping {vnumber} - output already exists", "INFO")
            self.results.append(vnumber)
            return True
            
        for attempt in range(state['attempts'], self.retry_count):
            state['attempts'] = attempt + 1
            state['last_error'] = None
            
            try:
                self.log(f"Simulating order {vnumber} (#{index}) - Attempt {attempt + 1}")
                
                if progress_callback:
                    progress_callback(f"Simulating {vnumber} - Attempt {attempt + 1}")
                
                # Save state before each major step
                def save_state(step_name):
                    state['last_step'] = step_name
                    if step_name not in state['completed_steps']:
                        state['completed_steps'].append(step_name)
                    with open(state_file, 'w') as f:
                        json.dump(state, f, indent=2)
                
                # Wait for pause/resume
                while self.paused and not self.stopped:
                    time.sleep(0.5)
                
                if self.stopped:
                    raise InterruptedError("Simulation stopped by user")
                
                # Step 1: Open Plataine and navigate to simulation
                self.wait_for_window("TPO FabricOptimizer", timeout=30)
                pyautogui.hotkey("alt", "t")
                pyautogui.press("f")
                
                # Step 2: Open Material Purchase Simulation
                self.wait_for_window("Material Purchase Simulation", timeout=30)
                
                # Step 3: Load from library
                if not self.click_and_type(TEMPLATE_IMAGES["order_entry"], str(index)):
                    raise Exception("Failed to enter order index")
                
                pyautogui.press("enter")
                time.sleep(2)
                
                # Step 4: Filter by order number
                if not self.click_and_type(TEMPLATE_IMAGES["filter_field"], vnumber):
                    raise Exception("Failed to enter filter")
                
                pyautogui.press("enter")
                time.sleep(1)
                pyautogui.press("down")
                pyautogui.press("enter")
                time.sleep(1)
                
                # Step 5: Switch to "By Steps" mode
                by_steps_loc = self.wait_for_image(TEMPLATE_IMAGES["by_steps"])
                pyautogui.click(by_steps_loc)
                
                # Step 6: Input width parameters
                if not self.click_and_type(TEMPLATE_IMAGES["from_field"], self.width_min):
                    raise Exception("Failed to set width minimum")
                
                if not self.click_and_type(TEMPLATE_IMAGES["to_field"], self.width_max):
                    raise Exception("Failed to set width maximum")
                
                if not self.click_and_type(TEMPLATE_IMAGES["step_field"], self.width_step):
                    raise Exception("Failed to set width step")
                
                # Step 7: Input Quantity Mix parameters
                if not self.click_and_type(TEMPLATE_IMAGES["qmix_from"], 1):
                    raise Exception("Failed to set quantity mix from")
                
                if not self.click_and_type(TEMPLATE_IMAGES["qmix_to"], 1):
                    raise Exception("Failed to set quantity mix to")
                
                if not self.click_and_type(TEMPLATE_IMAGES["qmix_steps"], 0):
                    raise Exception("Failed to set quantity mix step")
                
                # Step 8: Run simulation
                run_loc = self.wait_for_image(TEMPLATE_IMAGES["run_button"])
                pyautogui.click(run_loc)
                pyautogui.press("enter")
                
                # Step 9: Wait for simulation to complete (with dynamic timeout)
                self.log(f"Waiting for simulation to complete for {vnumber}...")
                simulation_timeout = 600  # 10 minutes defaults
                start_wait = time.time()
                while time.time() - start_wait < simulation_timeout:
                     if self.stopped:
                        raise InterruptedError("Simulation stopped by user")
                    
                    # Handle save plan dialog with tab-based navigation (3 tabs to Save button)
                try:
                        plan_name = vnumber  # Use the simulation number (vnumber) as the plan name
                        self.log(f"Saving simulation plan as {plan_name}")
                        if save_plan_with_tabs(plan_name):
                            save_state("saved_simulation_plan")
                        time.sleep(2)  # Wait for dialog to close
                        break  # Exit the loop after successful 
                    
                except Exception as e:
                        self.log(f"Error saving simulation plan: {e}", "WARNING")
                        time.sleep(1)  # Small delay before retry
                    
                # Check if simulation completed by looking for the Marker tab
                try:
                        marker_tab = pyautogui.locateCenterOnScreen("images/marker_tab.png", confidence=0.7)
                        if marker_tab:
                            self.log("Detected Marker tab - simulation completed")
                            save_state("simulation_completed")
                            break
                except Exception as e:
                        self.log(f"Error checking for marker tab: {e}", "DEBUG")
                    
                time.sleep(5)  # Check every 5 seconds
                
                # Step 10: Navigate to Marker tab and export
                time.sleep(3)
                pyautogui.hotkey("ctrl", "pagedown")  # Navigate to Marker tab
                pyautogui.hotkey("ctrl", "pagedown")  # Ensure we're on Marker tab
                
                # Step 11: Export to CSV
                pyautogui.hotkey("ctrl", "a")  # Select all
                pyautogui.hotkey("alt", "a")   # Alt+A for export menu
                pyautogui.press("e")           # E for export
                
                # Step 12: Save CSV file with robust handling
                output_path = os.path.join(self.output_folder, f"{vnumber}.csv")
                self.log(f"Attempting to save results to {output_path}")
                
                # Ensure output directory exists
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                # Try multiple methods to save the file
                save_success = False
                save_methods = [
                    self._save_csv_method_direct,
                    self._save_csv_method_saveas,
                    self._save_csv_method_clipboard
                ]
                
                for method in save_methods:
                    try:
                        if method(output_path):
                            save_success = True
                            break
                    except Exception as e:
                        self.log(f"Save method {method.__name__} failed: {e}", "WARNING")
                
                if not save_success:
                    raise Exception("All CSV save methods failed")
                
                # Verify file was saved
                time.sleep(2)  # Give time for save to complete
                if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
                    raise Exception(f"CSV file was not created or is empty: {output_path}")
                
                self.log(f"Successfully saved results to {output_path}")
                save_state("saved_results")
                
                # Step 13: Close simulation tab
                pyautogui.hotkey("ctrl", "w")
                
                # Verify CSV was created
                if os.path.exists(output_path):
                    self.results.append(vnumber)
                    self.log(f"Successfully completed simulation for {vnumber}")
                    return True
                else:
                    raise Exception("CSV file was not created")
                
            except Exception as e:
                self.log(f"Attempt {attempt + 1} failed for {vnumber}: {e}", "ERROR")
                if attempt < self.retry_count - 1:
                    self.log(f"Retrying {vnumber} in 5 seconds...")
                    time.sleep(5)
                else:
                    self.failed_orders.append((vnumber, str(e)))
                    self.log(f"All attempts failed for {vnumber}", "ERROR")
                    return False
        
        return False
    
    def run_all_simulations(self, progress_callback=None, status_callback=None):
        """Run simulations for all DXF files with progress tracking"""
        try:
            # Get all DXF files
            dxf_files = [f for f in os.listdir(self.dxf_folder) if f.lower().endswith(".dxf")]
            
            if not dxf_files:
                raise Exception("No DXF files found in the specified folder")
            
            self.log(f"Starting batch simulation for {len(dxf_files)} orders")
            
            # Create output directory
            os.makedirs(self.output_folder, exist_ok=True)
            
            # Process each file
            for i, filename in enumerate(dxf_files, start=1):
                if self.stopped:
                    self.log("Simulation stopped by user")
                    break
                
                vnumber = self.extract_vnumber(filename)
                if not vnumber:
                    self.log(f"Skipping {filename} - no V-number found", "WARNING")
                    continue
                
                if status_callback:
                    status_callback(f"Processing {vnumber} ({i}/{len(dxf_files)})")
                
                success = self.simulate_one_order(vnumber, i, progress_callback)
                
                if progress_callback:
                    progress_callback(f"Completed {i}/{len(dxf_files)} orders")
            
            # Generate final summary
            self.generate_summary()
            
        except Exception as e:
            self.log(f"Batch simulation failed: {e}", "ERROR")
            raise
    
    def generate_summary(self):
        """Generate and save simulation summary"""
        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_orders": len(self.results) + len(self.failed_orders),
            "successful": len(self.results),
            "failed": len(self.failed_orders),
            "success_rate": len(self.results) / (len(self.results) + len(self.failed_orders)) * 100 if (len(self.results) + len(self.failed_orders)) > 0 else 0,
            "successful_orders": self.results,
            "failed_orders": [{"order": order, "error": error} for order, error in self.failed_orders],
            "settings": {
                "width_min": self.width_min,
                "width_max": self.width_max,
                "width_step": self.width_step,
                "retry_count": self.retry_count
            }
        }
        
        # Save as JSON
        summary_path = os.path.join(self.output_folder, "simulation_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save as CSV
        summary_csv_path = os.path.join(self.output_folder, "simulation_summary.csv")
        summary_df = pd.DataFrame({
            'Order': self.results + [order for order, _ in self.failed_orders],
            'Status': ['Success'] * len(self.results) + ['Failed'] * len(self.failed_orders),
            'Error': [''] * len(self.results) + [error for _, error in self.failed_orders]
        })
        summary_df.to_csv(summary_csv_path, index=False)
        
        self.log(f"Summary: {len(self.results)} successful, {len(self.failed_orders)} failed")
        return summary

class PreviewWindow:
    """Enhanced preview window with export functionality for order previews"""
    
    def __init__(self, data, parent=None):
        self.data = data
        self.parent = parent
        self.index = 0
        self.search_text = ""
        
        # Create window
        self.window = tk.Toplevel(parent) if parent else tk.Tk()
        self.window.title("Order Preview")
        self.window.geometry("1200x800")
        
        # Create matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        plt.subplots_adjust(bottom=0.15, right=0.82)
        
        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, self.window)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Create control frame
        control_frame = ttk.Frame(self.window)
        control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Navigation buttons
        ttk.Button(control_frame, text="Previous", command=self.prev).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Next", command=self.next).pack(side=tk.LEFT, padx=5)
        
        # Search entry
        ttk.Label(control_frame, text="Search Order #:").pack(side=tk.LEFT, padx=(20, 5))
        self.search_var = tk.StringVar()
        search_entry = ttk.Entry(control_frame, textvariable=self.search_var, width=20)
        search_entry.pack(side=tk.LEFT, padx=5)
        search_entry.bind('<Return>', self.search)
        
        # Export button
        ttk.Button(control_frame, text="Export PNG", command=self.export_png).pack(side=tk.RIGHT, padx=5)
        
        self.render()
    
    def render(self):
        """Render the current preview"""
        self.ax.clear()
        self.render_order()
        self.canvas.draw()
    
    def render_order(self):
        """Render order preview"""
        try:
            order, widths, drops, filename = self.data[self.index]
            num_blinds = len(widths)
            
            # Draw grid
            self.ax.grid(True, which='major', axis='x', linestyle=':', alpha=0.25, color='#337ab7')
            
            # Draw bars
            for i, w in enumerate(widths):
                rect = Rectangle((0, i), w, 0.8, facecolor='#dd6b00', edgecolor="#337ab7", linewidth=0.25)
                self.ax.add_patch(rect)
            
            self.ax.set_xlim(0, max(widths) * 1.10 if widths else 1000)
            self.ax.set_ylim(0, num_blinds if num_blinds > 0 else 1)
            self.ax.set_aspect("auto")
            
            # Title and labels
            self.ax.set_title(f"Order Preview [{self.index+1}/{len(self.data)}] — Order #{order}",
                            fontsize=18, fontweight='bold', pad=24, color='#337ab7')
            self.ax.set_xlabel("Width (mm)", fontsize=15, labelpad=18, color='#23527c')
            self.ax.set_ylabel("Blind #", fontsize=15, labelpad=18, color='#23527c')
            
            # Statistics box
            if widths:  # Check if widths list is not empty
                stats_text = (
                    f"Order: {order}\n"
                    f"Max Width: {max(widths):.0f} mm\n"
                    f"Min Width: {min(widths):.0f} mm\n"
                    f"Avg Width: {np.mean(widths):.1f} mm\n"
                    f"Blinds: {num_blinds}\n"
                    f"File: {filename}"
                )
            else:
                stats_text = f"Order: {order}\nNo width data available\nFile: {filename}"
                
            self.ax.text(1.03, 0.98, stats_text, transform=self.ax.transAxes,
                        fontsize=14, verticalalignment="top", horizontalalignment="left",
                        bbox=dict(boxstyle="round,pad=0.8", facecolor="#f8f8ff", 
                                 edgecolor="#337ab7", alpha=0.98))
        except Exception as e:
            self.ax.clear()
            self.ax.text(0.5, 0.5, f"Error rendering preview: {str(e)}", 
                        ha='center', va='center', transform=self.ax.transAxes)
    
    def next(self):
        """Go to next item"""
        if self.data:
            self.index = (self.index + 1) % len(self.data)
            self.render()
    
    def prev(self):
        """Go to previous item"""
        if self.data:
            self.index = (self.index - 1) % len(self.data)
            self.render()
    
    def search(self, event=None):
        """Search for specific order"""
        search_term = self.search_var.get().strip()
        if not search_term or not self.data:
            return
        
        for idx, item in enumerate(self.data):
            try:
                identifier = str(item[0])  # Order number
                if search_term.lower() in identifier.lower():
                    self.index = idx
                    self.search_var.set("")  # Clear search box after successful search
                    self.render()
                    return
            except (IndexError, AttributeError):
                continue
        
        messagebox.showinfo("Search", f"Order '{search_term}' not found")
    
    def export_png(self):
        """Export current preview as PNG"""
        try:
            if not self.data:
                messagebox.showerror("Export Error", "No data to export")
                return
                
            order = self.data[self.index][0]
            filename = f"order_{order}_preview.png"
            filepath = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("All files", "*.*")],
                initialfile=filename
            )
            
            if filepath:
                self.fig.savefig(filepath, dpi=300, bbox_inches='tight', 
                               facecolor='white', edgecolor='none')
                messagebox.showinfo("Export", f"Preview exported to {filepath}")
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export preview: {str(e)}")

class SimulationApp:
    """Main application class with comprehensive GUI and functionality"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Kvadrat Shade - Efficiency Simulation Tool")
        self.root.geometry("1000x700")
        
        # Set application icon if available
        logo_path = os.path.join("images", "Kvadrat_logo.png")
        try:
            if os.path.exists(logo_path):
                self.root.iconphoto(True, tk.PhotoImage(file=logo_path))
            else:
                print(f"Logo not found at: {os.path.abspath(logo_path)}")
        except Exception as e:
            print(f"Could not load application icon: {e}")
        
        # Theme management
        self.dark_mode = tk.BooleanVar(value=False)
        self.current_theme = LIGHT_THEME
        
        # Simulation control
        self.simulator = None
        self.simulation_thread = None
        self.paused = False
        
        # Variables
        self.setup_variables()
        
        # Create widgets and apply theme
        self.create_widgets()
        self.apply_theme()
        
        # Load settings
        self.load_settings()
    
    def setup_variables(self):
        """Initialize all tkinter variables"""
        self.dxf_folder = tk.StringVar(value=os.path.join(os.getcwd(), "output_order_dxfs"))
        self.output_folder = tk.StringVar(value=os.path.join(os.getcwd(), "Sim_Output"))
        self.width_min = tk.IntVar(value=1900)
        self.width_max = tk.IntVar(value=3200)
        self.width_step = tk.IntVar(value=50)
        self.retry_count = tk.IntVar(value=3)
        self.status_text = tk.StringVar(value="Ready")
        self.progress_text = tk.StringVar(value="")
    
    def create_widgets(self):
        """Create all GUI widgets"""
        # Configure the root window
        self.root.configure(bg=KVADRAT_PRIMARY)
        
        # Main container with Kvadrat branding
        main_frame = ttk.Frame(self.root, style='TFrame')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Header frame with logo and title
        header_frame = ttk.Frame(main_frame, style='TFrame')
        header_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Try to load and display the Kvadrat logo
        try:
            logo_img = tk.PhotoImage(file="Kvadrat_logo.png")
            logo_label = ttk.Label(header_frame, image=logo_img)
            logo_label.image = logo_img  # Keep a reference
            logo_label.pack(side=tk.LEFT, padx=(0, 20))
        except Exception as e:
            print(f"Could not load logo image: {e}")
        
        # Title with custom font
        title_label = ttk.Label(header_frame, 
                              text="Kvadrat Shade - Fabric Simulation Tool",
                              font=TITLE_FONT,
                              style='TLabel')
        title_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Settings frame
        settings_frame = ttk.LabelFrame(main_frame, text="Simulation Settings", padding=10)
        settings_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Folder settings
        folder_frame = ttk.Frame(settings_frame)
        folder_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(folder_frame, text="DXF Input Folder:").grid(row=0, column=0, sticky="w", pady=2)
        ttk.Entry(folder_frame, textvariable=self.dxf_folder, width=60).grid(row=0, column=1, padx=(10, 5), pady=2)
        ttk.Button(folder_frame, text="Browse", command=self.browse_dxf_folder).grid(row=0, column=2, pady=2)
        
        ttk.Label(folder_frame, text="Output Folder:").grid(row=1, column=0, sticky="w", pady=2)
        ttk.Entry(folder_frame, textvariable=self.output_folder, width=60).grid(row=1, column=1, padx=(10, 5), pady=2)
        ttk.Button(folder_frame, text="Browse", command=self.browse_output_folder).grid(row=1, column=2, pady=2)
        
        # Width settings
        width_frame = ttk.Frame(settings_frame)
        width_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(width_frame, text="Width Range:").grid(row=0, column=0, sticky="w")
        ttk.Label(width_frame, text="From:").grid(row=0, column=1, padx=(20, 5))
        ttk.Entry(width_frame, textvariable=self.width_min, width=8).grid(row=0, column=2, padx=5)
        ttk.Label(width_frame, text="To:").grid(row=0, column=3, padx=(10, 5))
        ttk.Entry(width_frame, textvariable=self.width_max, width=8).grid(row=0, column=4, padx=5)
        ttk.Label(width_frame, text="Step:").grid(row=0, column=5, padx=(10, 5))
        ttk.Entry(width_frame, textvariable=self.width_step, width=8).grid(row=0, column=6, padx=5)
        
        # Advanced settings
        advanced_frame = ttk.Frame(settings_frame)
        advanced_frame.pack(fill=tk.X)
        
        ttk.Label(advanced_frame, text="Retry Count:").grid(row=0, column=0, sticky="w")
        ttk.Entry(advanced_frame, textvariable=self.retry_count, width=8).grid(row=0, column=1, padx=(10, 20))
        
        # Control buttons frame
        control_frame = ttk.LabelFrame(main_frame, text="Simulation Control", padding=10)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        button_frame = ttk.Frame(control_frame)
        button_frame.pack()
        
        self.run_button = ttk.Button(button_frame, text="Run All Simulations", 
                                    command=self.start_simulation, style="Accent.TButton")
        self.run_button.grid(row=0, column=0, padx=5)
        
        self.pause_button = ttk.Button(button_frame, text="Pause", 
                                      command=self.toggle_pause, state="disabled")
        self.pause_button.grid(row=0, column=1, padx=5)
        
        self.stop_button = ttk.Button(button_frame, text="Stop", 
                                     command=self.stop_simulation, state="disabled")
        self.stop_button.grid(row=0, column=2, padx=5)
        
        # Preview buttons
        preview_frame = ttk.LabelFrame(main_frame, text="Preview & Analysis", padding=10)
        preview_frame.pack(fill=tk.X, pady=(0, 10))
        
        preview_button_frame = ttk.Frame(preview_frame)
        preview_button_frame.pack()
        
        ttk.Button(preview_button_frame, text="Preview Orders", 
                  command=self.preview_orders).grid(row=0, column=0, padx=5)
        ttk.Button(preview_button_frame, text="Toggle Dark Mode", 
                  command=self.toggle_dark_mode).grid(row=0, column=1, padx=5)
        ttk.Button(preview_button_frame, text="Export Logs",
                  command=self.export_logs).grid(row=0, column=2, padx=5)
        
        # Progress and status frame
        progress_frame = ttk.LabelFrame(main_frame, text="Progress & Status", padding=10)
        progress_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Progress bar
        self.progress_bar = ttk.Progressbar(progress_frame, mode='determinate')
        self.progress_bar.pack(fill=tk.X, pady=(0, 5))
        
        # Status labels
        status_frame = ttk.Frame(progress_frame)
        status_frame.pack(fill=tk.X)
        
        ttk.Label(status_frame, text="Status:").pack(side=tk.LEFT)
        status_label = ttk.Label(status_frame, textvariable=self.status_text, foreground="blue")
        status_label.pack(side=tk.LEFT, padx=(5, 0))
        
        progress_label = ttk.Label(status_frame, textvariable=self.progress_text, foreground="green")
        progress_label.pack(side=tk.RIGHT)
        
        # Log frame
        log_frame = ttk.LabelFrame(main_frame, text="Simulation Log", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True)
        
        # Log text widget with scrollbar
        log_text_frame = ttk.Frame(log_frame)
        log_text_frame.pack(fill=tk.BOTH, expand=True)
        
        self.log_text = scrolledtext.ScrolledText(log_text_frame, height=10, wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        # Status bar at bottom
        self.status_bar = ttk.Label(self.root, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def apply_theme(self):
        """Apply current theme to all widgets"""
        theme = DARK_THEME if self.dark_mode.get() else LIGHT_THEME
        self.current_theme = theme
        
        # Configure root window
        self.root.configure(bg=theme['bg'])
        
        # Configure styles
        style = ttk.Style()
        
        # Configure button styles
        style.configure("Accent.TButton",
                       background=theme['accent'],
                       foreground=theme['button_fg'])
        
        # Update log text colors
        if hasattr(self, 'log_text'):
            self.log_text.configure(bg=theme['text_bg'], fg=theme['fg'])
    
    def browse_dxf_folder(self):
        """Browse for DXF input folder"""
        folder = filedialog.askdirectory(title="Select DXF Input Folder")
        if folder:
            self.dxf_folder.set(folder)
    
    def browse_output_folder(self):
        """Browse for output folder"""
        folder = filedialog.askdirectory(title="Select Output Folder")
        if folder:
            self.output_folder.set(folder)
    
    def toggle_dark_mode(self):
        """Toggle between light and dark themes"""
        self.dark_mode.set(not self.dark_mode.get())
        self.apply_theme()
        self.save_settings()
    
    def log_message(self, message, level="INFO"):
        """Add message to log widget"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_msg = f"[{timestamp}] {level}: {message}\n"
        
        self.log_text.insert(tk.END, formatted_msg)
        self.log_text.see(tk.END)
        
        # Also log to file
        if level == "ERROR":
            logging.error(message)
        elif level == "WARNING":
            logging.warning(message)
        else:
            logging.info(message)
    
    def update_status(self, message):
        """Update status bar and status text"""
        self.status_text.set(message)
        self.status_bar.configure(text=message)
        self.log_message(message)
    
    def update_progress(self, message):
        """Update progress text"""
        self.progress_text.set(message)
    
    def start_simulation(self):
        """Start the simulation process"""
        try:
            # Validate inputs
            if not os.path.exists(self.dxf_folder.get()):
                messagebox.showerror("Error", "DXF folder does not exist")
                return
            
            if not os.path.exists(self.output_folder.get()):
                try:
                    os.makedirs(self.output_folder.get(), exist_ok=True)
                except Exception as e:
                    messagebox.showerror("Error", f"Cannot create output folder: {e}")
                    return
            
            # Create simulator instance
            self.simulator = PlataineSimulator(
                dxf_folder=self.dxf_folder.get(),
                output_folder=self.output_folder.get(),
                width_min=self.width_min.get(),
                width_max=self.width_max.get(),
                width_step=self.width_step.get(),
                retry_count=self.retry_count.get()
            )
            
            # Update UI state
            self.run_button.configure(state="disabled")
            self.pause_button.configure(state="normal")
            self.stop_button.configure(state="normal")
            
            # Clear log
            self.log_text.delete(1.0, tk.END)
            
            # Start simulation in separate thread
            self.simulation_thread = threading.Thread(
                target=self.run_simulation_thread,
                daemon=True
            )
            self.simulation_thread.start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start simulation: {e}")
            self.log_message(f"Failed to start simulation: {e}", "ERROR")
    
    def run_simulation_thread(self):
        """Run simulation in separate thread"""
        try:
            self.update_status("Starting simulation...")
            
            # Get DXF files for progress tracking
            dxf_files = [f for f in os.listdir(self.dxf_folder.get())
                        if f.lower().endswith(".dxf")]
            
            if dxf_files:
                self.progress_bar.configure(maximum=len(dxf_files))
                self.progress_bar.configure(value=0)
            
            # Run simulation with callbacks
            self.simulator.run_all_simulations(
                progress_callback=self.update_progress,
                status_callback=self.update_status
            )
            
            # Update final status
            summary = self.simulator.generate_summary()
            success_count = summary['successful']
            failed_count = summary['failed']
            
            self.update_status(f"Simulation complete: {success_count} successful, {failed_count} failed")
            
            # Show completion dialog
            messagebox.showinfo("Simulation Complete",
                              f"Simulation finished!\n"
                              f"Successful: {success_count}\n"
                              f"Failed: {failed_count}\n"
                              f"Success Rate: {summary['success_rate']:.1f}%")
            
        except Exception as e:
            self.log_message(f"Simulation error: {e}", "ERROR")
            messagebox.showerror("Simulation Error", str(e))
        
        finally:
            # Reset UI state
            self.root.after(0, self.reset_ui_state)
    
    def reset_ui_state(self):
        """Reset UI to initial state"""
        self.run_button.configure(state="normal")
        self.pause_button.configure(state="disabled", text="Pause")
        self.stop_button.configure(state="disabled")
        self.paused = False
    
    def toggle_pause(self):
        """Toggle pause/resume simulation"""
        if self.simulator:
            self.paused = not self.paused
            self.simulator.paused = self.paused
            
            if self.paused:
                self.pause_button.configure(text="Resume")
                self.update_status("Simulation paused")
            else:
                self.pause_button.configure(text="Pause")
                self.update_status("Simulation resumed")
    
    def stop_simulation(self):
        """Stop the simulation"""
        if self.simulator:
            self.simulator.stopped = True
            self.update_status("Stopping simulation...")
    
    def preview_orders(self):
        """Show order preview window"""
        try:
            # Load order data
            order_data = self.load_order_data()
            if order_data:
                PreviewWindow(order_data, self.root)
            else:
                messagebox.showinfo("No Data", "No order data found. Generate DXF files first.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load order data: {e}")
    
    def load_order_data(self):
        """Load order data from Excel and DXF files"""
        try:
            # Load Excel data
            if not os.path.exists("data.xlsx"):
                return None
            
            df = pd.read_excel("data.xlsx")
            df.columns = df.columns.str.upper().str.strip()
            df = df[df["WIDTH"].notna() & df["DROP"].notna() & df["ORDER"].notna()]
            df["ORDER"] = df["ORDER"].astype(str)
            
            # Group by order and create preview data
            order_data = []
            for order, group in df.groupby("ORDER"):
                widths = group["WIDTH"].tolist()
                drops = group["DROP"].tolist()
                filename = f"ORDER_{order}.dxf"
                order_data.append((order, widths, drops, filename))
            
            return order_data
        except Exception as e:
            self.log_message(f"Error loading order data: {e}", "ERROR")
            return None
    
    def export_logs(self):
        """Export logs to file"""
        try:
            # Get log content
            log_content = self.log_text.get(1.0, tk.END)
            
            # Ask for save location
            filename = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("CSV files", "*.csv"), ("All files", "*.*")],
                initialvalue=f"simulation_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            )
            
            if filename:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(log_content)
                
                messagebox.showinfo("Export Complete", f"Logs exported to {filename}")
                self.log_message(f"Logs exported to {filename}")
        
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export logs: {e}")
    
    def save_settings(self):
        """Save current settings to file"""
        try:
            settings = {
                "dxf_folder": self.dxf_folder.get(),
                "output_folder": self.output_folder.get(),
                "width_min": self.width_min.get(),
                "width_max": self.width_max.get(),
                "width_step": self.width_step.get(),
                "retry_count": self.retry_count.get(),
                "dark_mode": self.dark_mode.get()
            }
            
            with open("settings.json", 'w') as f:
                json.dump(settings, f, indent=2)
        
        except Exception as e:
            self.log_message(f"Failed to save settings: {e}", "ERROR")
    
    def load_settings(self):
        """Load settings from file"""
        try:
            if os.path.exists("settings.json"):
                with open("settings.json", 'r') as f:
                    settings = json.load(f)
                
                self.dxf_folder.set(settings.get("dxf_folder", self.dxf_folder.get()))
                self.output_folder.set(settings.get("output_folder", self.output_folder.get()))
                self.width_min.set(settings.get("width_min", self.width_min.get()))
                self.width_max.set(settings.get("width_max", self.width_max.get()))
                self.width_step.set(settings.get("width_step", self.width_step.get()))
                self.retry_count.set(settings.get("retry_count", self.retry_count.get()))
                self.dark_mode.set(settings.get("dark_mode", self.dark_mode.get()))
        
        except Exception as e:
            self.log_message(f"Failed to load settings: {e}", "WARNING")
    
    def on_closing(self):
        """Handle application closing"""
        # Stop simulation if running
        if self.simulator:
            self.simulator.stopped = True
        
        # Save settings
        self.save_settings()
        
        # Close application
        self.root.destroy()
    
    def run(self):
        """Start the application"""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()

# DXF Generation Functions (integrated from separate files)
def load_data():
    """Load data from Excel file"""
    df = pd.read_excel("data.xlsx")
    df.columns = df.columns.str.upper().str.strip()
    df = df[df["WIDTH"].notna() & df["DROP"].notna()]
    return df

def generate_order_dxfs(df, output_folder="output_order_dxfs"):
    """Generate DXF files grouped by order"""
    os.makedirs(output_folder, exist_ok=True)
    
    # Clear existing files
    for f in Path(output_folder).glob("*.dxf"):
        f.unlink()
    
    result = []
    print("📦 Generating Order DXFs...")
    
    for order, group in tqdm(list(df.groupby("ORDER"))):
        # Use AutoCAD 2000 version for compatibility
        doc = ezdxf.new(dxfversion="AC1015")
        msp = doc.modelspace()
        x_pos = 0
        widths, drops = [], []
        
        for _, row in group.sort_values("WIDTH", ascending=False).iterrows():
            width = row["WIDTH"]
            drop = row["DROP"]
            # Rectangle as closed LWPOLYLINE
            points = [
                (x_pos, 0),
                (x_pos + width, 0),
                (x_pos + width, drop),
                (x_pos, drop),
                (x_pos, 0)  # Close the polyline
            ]
            msp.add_lwpolyline(points, close=True)
            widths.append(width)
            drops.append(drop)
            x_pos += width
        
        filename = f"ORDER_{order}.dxf"
        doc.saveas(os.path.join(output_folder, filename))
        result.append((order, widths, drops, filename))
    
    return result

def generate_fabric_dxfs(df, output_folder="output_fabric_dxfs"):
    """Generate DXF files grouped by fabric"""
    os.makedirs(output_folder, exist_ok=True)
    
    # Clear existing files
    for f in Path(output_folder).glob("*.dxf"):
        f.unlink()
    
    result = []
    print("🧵 Generating Fabric DXFs...")
    
    for fabric, group in tqdm(list(df.groupby("FABRIC"))):
        doc = ezdxf.new()
        msp = doc.modelspace()
        y_pos = 0  # Start stacking from y=0
        
        # Sort by width ascending (smallest to largest)
        sorted_group = group.sort_values("WIDTH", ascending=True).reset_index(drop=True)
        
        for _, row in sorted_group.iterrows():
            width = row["WIDTH"]
            drop = row["DROP"] * 0.0001  # Scale factor
            # Draw each blind at the next y position
            msp.add_lwpolyline([
                (0, y_pos), (width, y_pos), (width, y_pos + drop), (0, y_pos + drop), (0, y_pos)
            ])
            y_pos += drop  # Stack the next blind above
        
        filename = f"FABRIC_{fabric}.dxf"
        doc.saveas(os.path.join(output_folder, filename))
        result.append((fabric, sorted_group["WIDTH"].tolist(), filename))
    
    return result

def main():
    """Main entry point"""
    try:
        # Create and run the application
        app = SimulationApp()
        app.run()
    
    except Exception as e:
        print(f"Application error: {e}")
        logging.error(f"Application error: {e}")

if __name__ == "__main__":
    main()