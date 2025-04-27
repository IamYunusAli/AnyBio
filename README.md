# AnyBio: Bioinformatics Research AI Assistant

AnyBio is a Streamlit-based application that allows users to interact with a bioinformatics knowledge base. It uses Google Generative AI and FAISS for document embedding and retrieval.

## Prerequisites

Before running the application, ensure you have the following installed:

1. **Python 3.8 or higher**: [Download Python](https://www.python.org/downloads/)
2. **pip**: Python's package manager (comes with Python installation).
3. **Virtual Environment (optional)**: Recommended to isolate dependencies.

## Setup Instructions

### 1. Clone the Repository
Clone the repository to your local machine:
```bash
git clone <repository-url>
cd AnyBio
```

### 2. Create a Virtual Environment (Optional but Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
Install the required Python packages:
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a `.env` file in the root directory and add the following:
```
GOOGLE_API_KEY=<your-google-api-key>
```
Replace `<your-google-api-key>` with your actual Google AI API key.

### 5. Add PDF Files
Place the PDF files you want to process in the `data` directory. Create the directory if it does not exist:
```bash
mkdir data
```

### 6. Run the Application
Start the Streamlit app:
```bash
streamlit run app.py
```

### 7. Access the Application
Open your browser and navigate to:
```
http://localhost:8501
```

## Features

- **Multi-language Support**: Supports English, Spanish, Amharic, and Arabic.
- **PDF Processing**: Extracts and indexes text from PDF files.
- **Knowledge Base Management**: Recreate or clear the FAISS vector store.
- **Chat Interface**: Ask bioinformatics-related questions and get responses based on the knowledge base.

## Troubleshooting

- **Missing API Key**: Ensure the `.env` file contains a valid Google API key.
- **PDF Directory Not Found**: Create the `data` directory and add PDF files.
- **Dependencies Issue**: Ensure all dependencies are installed using `pip install -r requirements.txt`.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
