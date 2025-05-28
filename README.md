# Random Forest Image Classification System 🌳📷

This project implements a client-server application for image classification using a Random Forest model. The system allows users to register, log in, and send images to a server for classification. Communication between the client and server is secured using RSA and AES encryption.

---

## Overview ✨

The project consists of a server that hosts the Random Forest model and handles client requests, and a client with a graphical user interface (GUI) for user interaction. Users can classify images of vehicles (Cars, Motorbikes, Planes, Ships). The server handles user authentication, image processing, and classification, while the client provides an interface for users to interact with the system.

---

## Features 🚀

* **User Authentication:** Secure user registration and login functionality[cite: 1].
* **Secure Communication:** Implements RSA for initial key exchange and AES for session data encryption between client and server[cite: 1].
* **Image Classification:** Classifies images into predefined categories (Cars, Motorbikes, Planes, Ships) using a Random Forest model[cite: 1].
* **Client GUI:** A Tkinter-based graphical user interface for easy interaction[cite: 1].
* **Database Integration:** Uses SQLite to store user credentials and preferences.
* **Model Training & Loading:** The server can train a new Random Forest model or load a pre-trained one.
* **Feature Extraction:** Implements a custom feature extraction pipeline tailored for vehicle images.

---

## Architecture 🏗️

The system follows a client-server architecture:

1.  **RFServer (`RFServer.py`):**
    * Handles client connections and requests in separate threads.
    * Manages user authentication via `DatabaseHandler.py`.
    * Loads or trains a Random Forest model (`Random_Forest.py`, `Decision_Tree.py`) for image classification.
    * Performs feature extraction and prediction on images sent by clients.
    * Manages secure communication by establishing encrypted channels with clients using `SecurityUtils.py`.

2.  **RFClient (`RFClient.py`):**
    * Provides a GUI (built with Tkinter) for users to interact with the server[cite: 1].
    * Allows users to connect to the server, register, and log in[cite: 1].
    * Enables users to select an image, send it to the server, and view classification results[cite: 1].
    * Handles secure communication with the server using `SecurityUtils.py`[cite: 1].

3.  **Machine Learning Model:**
    * **Decision Tree (`Decision_Tree.py`):** Implements a decision tree classifier which serves as the base learner for the Random Forest. It includes methods for calculating entropy, Gini impurity, finding the best split, and building the tree.
    * **Random Forest (`Random_Forest.py`):** Implements a Random Forest classifier by creating an ensemble of decision trees. It supports bootstrapping and aggregation of predictions from multiple trees.
    * **TreeNode (`treenode.py`):** A helper class representing a node in the decision tree.

4.  **Database (`DatabaseHandler.py`):**
    * Uses SQLite to store user information, including usernames, hashed passwords, salts, and email addresses.
    * Manages user settings such as last login time and preferred server.
    * Provides functions for user registration, authentication, and updating user preferences.

5.  **Security (`SecurityUtils.py`):**
    * Provides static methods for generating RSA key pairs.
    * Encrypts messages using RSA public keys and decrypts them using RSA private keys.
    * Generates AES keys and initialization vectors (IVs) for symmetric encryption.
    * Encrypts and decrypts data using AES-256-CBC with PKCS7 padding.

---

## Prerequisites 📋

Before running the project, ensure you have Python installed (version 3.x recommended).

---

## Setup and Installation ⚙️

1.  **Clone the Repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Install Dependencies:**
    Install the required Python libraries using the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```
    The dependencies are:
    * Pillow
    * numpy
    * cryptography
    * opencv-python

3.  **Image Data:**
    * The server expects image data for training and testing to be located in a directory named `Model_pics`.
    * Inside `Model_pics`, create two subdirectories: `train` and `test`.
    * Within both `train` and `test`, create subdirectories for each class: `Cars`, `Motorbikes`, `Planes`, `Ships`. Place the respective images in these folders.
    * The server loads images from these directories to train the model if a pre-trained model is not found or if re-training is triggered.

4.  **Database:**
    * The `DatabaseHandler.py` script will automatically create a SQLite database file named `user_database.db` in the same directory where the server is run, if it doesn't already exist.

5.  **Pre-trained Model:**
    * The server attempts to load a pre-trained model named `vehicle_rf_model.npy`.
    * If this file is not found, or if it's corrupted, the server will automatically initiate the model training process using the images in the `Model_pics/train` directory and evaluate it using images from `Model_pics/test`. The trained model will then be saved as `vehicle_rf_model.npy`.

---

## Running the Project ▶️

You need to run the server first, and then the client.

### 1. Start the Server

Navigate to the project directory and run the server script:
```bash
python RFServer.py [options]
```
**Server command-line options:**

* `--host <hostname>`: Server host address (default: `0.0.0.0`).
* `--port <port_number>`: Server port number (default: `9000`).
* `--model <model_file_path>`: Path to the Random Forest model file (default: `vehicle_rf_model.npy`).
* `--train_dir <path_to_train_data>`: Directory containing training data (default: `train`, relative to `Model_pics`).
* `--test_dir <path_to_test_data>`: Directory containing test data (default: `test`, relative to `Model_pics`).
* `--training_limit <number>`: Maximum number of training samples to use per class (default: `10000`).

**Example:**

```bash
python RFServer.py --port 9000
```
## Using the Client GUI 🖱️

Once the client application starts:

### Connect to Server:

* The GUI will open on the "Login/Register" tab.
* The server address and port are displayed. Click the "**Connect**" button to establish a connection with the server.
* The status label will indicate "Connected" if successful. The initial connection also involves a security handshake where RSA public keys are exchanged and an AES session key is established.

### Register (New Users):

* In the "Register" section, enter a desired username, password, confirm the password, and optionally provide an email address.
* Click the "**Register**" button. You will receive a confirmation message upon successful registration.

### Login (Existing Users):

* In the "Login" section, enter your username and password.
* Click the "**Login**" button.
* Upon successful login, the "Image Classification" tab will be enabled and automatically selected. Your username will be displayed in this tab.

### Image Classification:

* On the "Image Classification" tab, click the "**Select Image**" button to choose an image file from your computer (supports formats like JPG, PNG, BMP, GIF).
* The selected image will be displayed in the GUI.
* Click the "**Classify Image**" button. The image will be preprocessed (resized to 32x32, normalized), sent securely to the server, and classified.
* The classification results, including the predicted class, confidence score, and class probabilities, will be displayed in the "Classification Results" area.

### Logout:

* Click the "**Logout**" button in the "Image Classification" tab to log out. This will disable the "Image Classification" tab and return you to the "Login/Register" tab.

### Disconnect:

* On the "Login/Register" tab, click the "**Disconnect**" button (which was previously the "Connect" button) to close the connection to the server.

## Modules 🧩

* `RFClient.py`: Implements the client-side application with a Tkinter GUI for user interaction, image selection, and communication with the server for classification.
* `RFServer.py`: Implements the server-side application that handles client connections, user authentication, model loading/training, image feature extraction, and classification.
* `Random_Forest.py`: Contains the implementation of the Random Forest classifier algorithm, including bootstrapping and ensemble prediction.
* `Decision_Tree.py`: Provides the implementation for a single decision tree, which is used as a base learner in the Random Forest. Includes logic for splitting nodes based on Gini impurity or information gain.
* `treenode.py`: Defines the `TreeNode` class used to build the decision trees.
* `DatabaseHandler.py`: Manages all database operations using SQLite, including creating tables, registering users with hashed passwords and salts, and authenticating users.
* `SecurityUtils.py`: A utility class providing methods for RSA and AES encryption/decryption to secure client-server communication.
* `requirements.txt`: Lists the project dependencies.
