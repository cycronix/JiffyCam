# JiffyCam

JiffyCam is a portable, all-python real-time video capture webapp designed to efficiently record and manage video streams. It features a circular buffer for continuous recording and allows users to save clips on demand, making it ideal for surveillance, event recording, and more.

![What is this](jiffycam.jpg)

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Features

- **Real-Time Video Capture**: Capture video from various camera sources in real-time.
- **Circular Buffer**: Continuously records video, allowing for easy retrieval of recent footage.
- **Clip Saving**: Save clips on demand with a simple user interface.
- **Health Monitoring**: Includes a watchdog mechanism to ensure system stability.
- **User-Friendly Interface**: Intuitive controls for managing recordings and settings.

## Installation

To install JiffyCam, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your_username/jiffycam.git
   cd jiffycam
   ```

2. **Install the required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

To run the application, use the following command:

```bash
streamlit run jiffycam.py
```

### Controls

- **Toggle Real-Time Capture**: Use the button in the interface to start or stop real-time capture.
- **Save Clips**: Clips can be saved by triggering the save functionality in the UI.

## Configuration

JiffyCam allows for configuration through a configuration file. You can load and save your settings using the following methods:

- **Load Configuration**: Automatically loads settings on startup.
- **Save Configuration**: Save your current settings to a file.

## Contributing

Contributions are welcome! If you would like to contribute to JiffyCam, please follow these steps:

1. **Fork the repository**.
2. **Create a new branch** (`git checkout -b feature/YourFeature`).
3. **Make your changes and commit them** (`git commit -m 'Add some feature'`).
4. **Push to the branch** (`git push origin feature/YourFeature`).
5. **Open a pull request**.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For any inquiries or issues, please contact:

- GitHub: [cycronix](https://github.com/cycronix)

---

Thank you for using JiffyCam! We hope you find it useful for your video capture needs.
