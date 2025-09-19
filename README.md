# HYBpy<sup></sup> <img align="left" src="Website/hybpy/src/Image/hybpy_logo.png" width="50" />

![Latest Release](https://img.shields.io/github/v/release/joko1712/HYBpy)

## Overview

HYBpy is a python-based tool for building and evaluating hybrid models of bioprocesses and biological systems.
This repository contains the source code for the web-based application available at the www.hybpy.com website.

### Installing and Running HYBpy Locally

This guide provides a comprehensive, step-by-step process for setting up and running **HYBpy** on your local machine. Following these instructions will ensure a clean and isolated environment for your project.

---

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

## ⸎ Developed at

-   HYBpy is developed and maintaned at UCIBIO - Applied Molecular Biosciences Unit, NOVA School of Science and Technology, Universidade NOVA de Lisboa, 2829-516 Caparica, Portugal

_Authors:_ [José Pereira](https://github.com/joko1712), [Rafael Costa](https://github.com/r-costa), José Pinto, Rui Oliveira

## License

This work is licensed under a <a href="https://www.gnu.org/licenses/gpl-3.0.html"> GNU Public License (version 3.0).</a>
