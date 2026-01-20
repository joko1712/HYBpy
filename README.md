# HYBpy<sup></sup> <img align="left" src="Website/hybpy/src/Image/hybpy_logo.png" width="50" />

![Latest Release](https://img.shields.io/github/v/release/joko1712/HYBpy)

## Overview

HYBpy is a python-based tool for building and evaluating hybrid models of bioprocesses and biological systems.
This repository contains the source code for the web-based application available at the www.hybpy.com website.

### Installing and Running HYBpy Locally

You can run HYBpy locally in two ways:

1. **Recommended (no Python setup):** Download a pre-built executable (Windows/macOS).
2. **Developer setup:** Run from source using venv or Conda.

---
# Option A — Windows Executable (Recommended)

### Step A1: Download the executable
Go to the **Latest Release** on GitHub and download the Windows package:

- `HYBpy-LocalTrainer-windows-x64.zip` (Windows 10/11, 64-bit)

### Step A2: Extract the ZIP
Extract the `.zip` file to a folder of your choice.

### Step A3: Run the executable
Double-click the executable (or run from PowerShell):

```powershell
.\HYBpy_LocalTrainer.exe
```
### Step A4: Verify it is running
After a few seconds the console should say:

```bash
    [READY] Local trainer starting
```

# Option B — macOS Executable (Recommended)

### Step B1: Download the executable
Go to the Latest Release on GitHub and download the macOS package if matches your Mac:

- Apple Silicon (M1/M2/M3/M4): HYBpy-LocalTrainer-macos-arm64.zip

### Step B2: Extract and run
Extract the `.zip` and run:
```bash
chmod +x HYBpy_LocalTrainer
./HYBpy_LocalTrainer
```

### Step B3: If macOS blocks the app (Gatekeeper)
If macOS prevents execution because the binary is not notarized/signed:

- System Settings → Privacy & Security → "HYBpy_LocalTrainer" was blocked to protect your Mac. → “Open Anyway”
- Then run again, or execute via Terminal.

### Step B4: Verify it is running
After a few seconds the console should say:

```bash
    [READY] Local trainer starting
```

# Option C — Using Python Virtual Environment (venv)

### Step 1: Obtain the HYBpy Repository

First, you need to get the HYBpy code onto your computer. There are two primary methods for doing this:

-   **Using Git:** Clone the repository directly from GitHub using the command line. This is the recommended method as it makes it easier to update the code in the future.

    ```bash
    git clone https://github.com/joko1712/HYBpy.git
    ```

-   **Downloading a ZIP file:** Alternatively, you can download the repository as a compressed file from the GitHub page. Navigate to `github.com/joko1712/HYBpy` and click on the "Code" button, then select "Download ZIP."

Once you have the code, use your terminal or command prompt to navigate into the project's root directory.

```bash
cd HYBpy
```

---

## Option C1 — Using Python Virtual Environment (venv)

### Step 2: Create a Virtual Environment

To avoid conflicts with other Python projects and their dependencies, it's best practice to create a dedicated **virtual environment**. This isolates the project's required packages.

Use the following command to create a new virtual environment named `HYBpyEnv`:

```bash
python -m venv HYBpyEnv
```

---

### Step 3: Activate the Virtual Environment

Before installing any packages, you must **activate** the virtual environment. The commands differ slightly based on your operating system:

-   **Windows:**

    ```bash
    HYBpyEnv\Scripts\activate
    ```

-   **macOS / Linux:**

    ```bash
    source HYBpyEnv/bin/activate
    ```

You'll know the environment is active when the name `(HYBpyEnv)` appears at the beginning of your command prompt.

---

### Step 4: Install Required Packages

With the virtual environment active, you can now install all the necessary dependencies. It's a good practice to first upgrade `pip` (the package installer) to its latest version.

```bash
pip install --upgrade pip
```

Next, install the required libraries. This can be done with a single command:

```bash
pip install numpy scipy matplotlib torch scikit-learn h5py torchdiffeq pandas
```

---

### Step 5: Run the tool

After installing the dependencies, you can run the tool using the following command:

```bash
python run_hybtrain_local.py
```

---

### Step 6: Deactivate the Virtual Environment

Once you are finished working on the project, you should deactivate the virtual environment. This will return you to your system's global Python environment.

```bash
deactivate
```

---

## Option C2 — Using Conda Environment

### Step 2: Create a Conda Environment

If you prefer Conda (Anaconda or Miniconda), you can create an isolated environment with Python 3.10 or later:

```bash
conda create -n hybpy python=3.10
```

### Step 3: Activate the Conda Environment

Activate the newly created Conda environment:

```bash
conda activate hybpy
```

### Step 4: Install Required Packages

Install the necessary packages using Conda and pip:

```bash
conda install numpy scipy matplotlib scikit-learn pandas h5py pytorch -c pytorch
pip install torchdiffeq
```

### Step 5: Run the tool

After installing the dependencies, you can run the tool using the following command:

```bash
python run_hybtrain_local.py
```

### Step 6: Deactivate the Conda Environment

When you are done working, you can deactivate the Conda environment:

```bash
conda deactivate
```

---

## ⸎ Developed at

-   HYBpy is developed and maintained at UCIBIO - Applied Molecular Biosciences Unit, NOVA School of Science and Technology, Universidade NOVA de Lisboa, 2829-516 Caparica, Portugal

_Authors:_ [José Pereira](https://github.com/joko1712), [Rafael Costa](https://github.com/r-costa), José Pinto, Rui Oliveira

## License

This work is licensed under a <a href="https://www.gnu.org/licenses/gpl-3.0.html"> GNU Public License (version 3.0).</a>
