\# Mental Health Prediction Using Social Media Analytics



SocialSense is a mental health support and stress prediction project that combines machine learning, natural language processing, and interactive support features. It analyzes text to predict potential stress indicators and provides users with mental wellness resources.



\## Features



\- \*\*Stress Prediction:\*\* Uses Logistic Regression, Linear SVM, and RoBERTa-based text classification.

\- \*\*AI Companion:\*\* An AI-powered chatbot for supportive conversations.

\- \*\*Help \& Hope:\*\* Searchable mental wellness questions and answers.

\- \*\*Calm Corner:\*\* Includes breathing exercises, a grounding activity, and a session-only journal.

\- \*\*Reach Out:\*\* Provides mental health support contacts and emergency resources.



\## Technology Stack



\*\*Frontend\*\*

\- React

\- Vite

\- JavaScript

\- CSS



\*\*Backend\*\*

\- Python

\- FastAPI

\- Uvicorn



\*\*Machine Learning\*\*

\- Scikit-learn

\- PyTorch

\- Hugging Face Transformers

\- RoBERTa



\*\*AI Integration\*\*

\- OpenAI API



\## Project Structure



```text

mental\_health\_prediction-main/

├── app/

│   └── predictor.py

├── backend/

│   └── main.py

├── frontend/

│   └── src/

│       ├── App.jsx

│       ├── Features.jsx

│       └── Features.css

├── data/

│   └── raw/

│       └── dreaddit/

├── models/

│   ├── baseline\_stress/

│   └── roberta\_stress\_new/

├── scripts/

├── .gitignore

└── README.md

```



\*The structure above highlights key files and folders; your local project may contain additional files.\*



\## Getting Started



\### 1. Clone the Repository



```bash

git clone https://github.com/manasasaragalla/Mental-health-prediction-using-social-media-analytics.git

cd Mental-health-prediction-using-social-media-analytics

```



\### 2. Set Up the Python Environment



Create and activate a virtual environment:



```powershell

python -m venv .venv

.\\.venv\\Scripts\\Activate.ps1

```



Install the Python dependencies using your project's dependency file, if provided:



```powershell

pip install -r requirements.txt

```



If your project does not have a root-level `requirements.txt`, install the dependencies listed in your project's actual dependency files.



\### 3. Configure Environment Variables



Create a `.env` file in the project root and add your OpenAI API key:



```dotenv

OPENAI\_API\_KEY=your\_openai\_api\_key

OPENAI\_CHAT\_MODEL=gpt-4o-mini

```



Replace the placeholder with your own API key. \*\*Never commit `.env` or expose your API key publicly.\*\*



\### 4. Start the Backend



From the project root, run:



```powershell

python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000

```



\### 5. Start the Frontend



Open a second terminal:



```powershell

cd frontend

npm install

npm run dev

```



Open the local URL printed by Vite, usually `http://localhost:5173`.



\## Dataset



The project uses the Dreaddit dataset for stress-related text classification. Confirm the dataset's original source and licensing before redistributing it.



\## Machine Learning Models



\- Logistic Regression

\- Linear Support Vector Machine (SVM)

\- RoBERTa transformer model



Model predictions indicate potential stress-related language and are not medical diagnoses.



\## Mental Health Support



The application includes support resources intended to help users find further assistance. In India, Tele-MANAS can be reached at \*\*14416\*\*. For immediate emergencies, call \*\*112\*\*.



\## Security Notes



\- Keep API keys and secrets out of source control.

\- Large model-weight files may need to be downloaded or trained separately.

\- Do not upload private datasets or sensitive user information.



\## Future Improvements



\- Improve model evaluation and compare classification metrics.

\- Add automated tests and error handling.

\- Improve accessibility and responsive design.

\- Document dataset provenance and model limitations.



\## Disclaimer



This project is intended for educational and informational purposes. Stress predictions are experimental and should not be used as a substitute for professional mental health assessment or care.



\## Author



\*\*Manasa Saragalla\*\*



GitHub: \[@manasasaragalla](https://github.com/manasasaragalla)

