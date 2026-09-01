PARTICLE FLOW VIDEO ANALYSIS TOOL
Portable local package

WHAT THIS PACKAGE DOES
----------------------
This package runs the Streamlit interface locally on the user's own computer.
The interface opens in the default web browser, but the video analysis runs on the local computer.
No fixed Mac or Windows project-folder path is required by the application.

WINDOWS - FIRST USE
-------------------
1. Extract the ZIP file to a normal folder (for example Desktop\ParticleFlow).
2. Make sure Python 3.10 or newer is installed.
3. Double-click: Start_ParticleFlow_Windows.bat
4. On first use, the launcher creates a private .venv folder and installs the required Python packages.
   An internet connection is needed for this first installation.
5. Streamlit then starts and the interface should open automatically in the default browser.
6. Keep the launcher window open while using the tool.
7. To stop the tool, close the launcher window or press Ctrl+C in it.

MAC - FIRST USE
---------------
1. Extract the ZIP file.
2. Make sure Python 3.10 or newer is installed.
3. Double-click: Start_ParticleFlow_Mac.command
4. If macOS blocks it the first time, right-click the file, choose Open, then confirm Open.
5. On first use, the launcher creates a private .venv folder and installs the required packages.
6. Keep the Terminal window open while using the tool.


USING THE ROI WORKFLOW
----------------------
The drag boxes are placed inside Streamlit forms. Moving or resizing a box does not rerun the whole app; coordinates are sent to Python only when the single Continue button is pressed.
1. Step 1: move/resize the blue crop box, then click Apply Crop & Continue.
2. Step 2 always displays the applied cropped frame. Move/resize the yellow Reference ROI, then click Apply Reference ROI & Continue.
3. Step 3: choose Number of zones and move/resize the green Analysis Area. Both settings are inside the same form, so changing the zone count does not refresh or move the green box. Click Apply Analysis Area & Continue once when both are ready.

LATER USE
---------
After the first successful setup, simply use the same launcher again. The existing .venv will be reused.

FILES
-----
app.py                          Main Streamlit application
requirements.txt                Python package dependencies
.streamlit/config.toml          Streamlit configuration (450 MB upload limit)
Start_ParticleFlow_Windows.bat  Windows launcher
Start_ParticleFlow_Mac.command  macOS launcher
README.txt                      Setup and usage notes

IMPORTANT NOTES
---------------
- Python is still required. This is a portable source package, not a standalone .exe.
- The first run needs an internet connection to install Python dependencies.
- Later runs reuse the local .venv environment.
