# Particle Flow Video Analysis Tool

**Portable local package**

## What This Package Does

This package runs the Streamlit interface locally on the user's own computer.

The interface opens in the default web browser, but the video analysis runs on the local computer.

No fixed Mac or Windows project-folder path is required by the application.

---

## Windows — First Use

1. Extract the ZIP file to a normal folder, for example:

   `Desktop\ParticleFlow`

2. Make sure **Python 3.10 or newer** is installed.

3. Double-click:

   `Start_ParticleFlow_Windows.bat`

4. On first use, the launcher creates a private `.venv` folder and installs the required Python packages.

   **An internet connection is needed for this first installation.**

5. Streamlit then starts and the interface should open automatically in the default browser.

6. Keep the launcher window open while using the tool.

7. To stop the tool, close the launcher window or press `Ctrl+C` in it.

---

## macOS — First Use

1. Extract the ZIP file.

2. Make sure **Python 3.10 or newer** is installed.

3. Double-click:

   `Start_ParticleFlow_Mac.command`

4. If macOS blocks it the first time, right-click the file, choose **Open**, then confirm **Open**.

5. On first use, the launcher creates a private `.venv` folder and installs the required Python packages.

6. Keep the Terminal window open while using the tool.

---

## Using the ROI Workflow

The draggable boxes are placed inside Streamlit forms. Moving or resizing a box does not rerun the whole application. Coordinates are sent to Python only when the corresponding **Continue** button is pressed.

### Step 1 — Crop Video

Move or resize the **blue crop box**, then click:

**Apply Crop & Continue**

### Step 2 — Reference ROI

Step 2 always displays the applied cropped frame.

Move or resize the **yellow Reference ROI**, then click:

**Apply Reference ROI & Continue**

### Step 3 — Analysis Area

Choose the **Number of zones** and move or resize the **green Analysis Area**.

Both settings are inside the same form, so changing the zone count does not refresh or move the green box.

When both settings are ready, click:

**Apply Analysis Area & Continue**

---

## Later Use

After the first successful setup, simply use the same launcher again.

The existing `.venv` environment will be reused, so the required Python packages do not need to be installed again.

---

## Files

| File                             | Description                                                |
| -------------------------------- | ---------------------------------------------------------- |
| `app.py`                         | Main Streamlit application                                 |
| `requirements.txt`               | Python package dependencies                                |
| `.streamlit/config.toml`         | Streamlit configuration, including the 450 MB upload limit |
| `Start_ParticleFlow_Windows.bat` | Windows launcher                                           |
| `Start_ParticleFlow_Mac.command` | macOS launcher                                             |
| `README.md`                      | Setup and usage instructions                               |

---

## Important Notes

* **Python is still required.** This is a portable source package, not a standalone `.exe`.
* The first run requires an **internet connection** to install the Python dependencies.
* Later runs reuse the local `.venv` environment.
* Keep the Windows launcher or macOS Terminal window open while the application is running.

