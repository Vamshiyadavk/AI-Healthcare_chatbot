🩺 HealthBot – AI Powered Healthcare Diagnosis Chatbot

Python | Flask | Machine Learning | HTML |

An intelligent healthcare chatbot that predicts possible diseases based on symptoms and provides basic remedies with severity analysis.

---

📌 Project Overview

HealthBot is an AI-powered healthcare assistant that predicts possible diseases based on user-entered symptoms.

The project was developed to provide a simple and user-friendly way for users to get preliminary health insights before consulting a medical professional.

Using Machine Learning and Flask, the application analyzes symptoms, predicts diseases, displays confidence levels, and suggests basic remedies along with severity analysis.

This project reflects my interest in Artificial Intelligence, Healthcare Technology, and building real-world Machine Learning applications.

---

🚀 Features

✅ Predicts diseases from user-entered symptoms
✅ Uses Machine Learning algorithms for diagnosis
✅ Displays prediction confidence percentage
✅ Provides home remedies and health suggestions
✅ Shows severity level of the disease
✅ Alerts users for serious conditions
✅ Simple and user-friendly interface
✅ Fast and responsive web application
✅ Built using Flask and Machine Learning

---

🧠 How It Works

1. User enters symptoms in the web interface.
2. Symptoms are processed and cleaned.
3. The trained Machine Learning model analyzes the symptoms.
4. The model predicts the most probable disease.
5. Confidence score is calculated.
6. Remedy and severity details are displayed.
7. For serious conditions, the system advises consulting a healthcare professional immediately.

---

🛠️ Tech Stack

Frontend

- HTML

Backend

- Python
- Flask

Machine Learning

- Scikit-learn
- Pandas
- NumPy

Algorithms Used

- TF-IDF N-Gram Feature Extraction (1-3 word n-grams)
- Calibrated Support Vector Classifier (LinearSVC with Sigmoid calibration)
- Regularized Logistic Regression Ensemble

---

📂 Project Structure

AI-Healthcare_chatbot/
│
├── templates/              # HTML Templates (index, result, error)
│   ├── index.html          # Main symptom input interface
│   ├── result.html         # Diagnosis and remedy report
│   └── error.html          # User-friendly error page
├── disease_symptoms.csv    # Dataset covering 281 conditions and symptoms
├── home_remedies.csv       # Evidence-based home care & severity ratings
├── app.py                  # Flask Web UI & RESTful API Service
├── train_model.py          # Data augmentation, training & calibration script
├── run.py                  # Direct runner script
├── requirements.txt        # Cross-platform dependencies
└── README.md               # Documentation

---

💡 Sample Input & Output

Input Symptoms

wheezing, shortness of breath, chest tightness, coughing

Predicted Result

Disease: Asthma
Confidence: 70%+
Severity: Serious
Remedy: Practice breathing exercises, use steam inhalation, avoid triggers

---

📸 Application Screenshots

Symptom Input Page

- Clean and simple interface for entering symptoms.
- Users can provide multiple symptoms separated by commas.

Diagnosis Result Page

- Displays predicted disease.
- Shows confidence percentage and differential diagnoses.
- Provides remedies and severity analysis.
- Alerts users if immediate medical attention is required.

---

⚙️ Installation & Setup

1️⃣ Clone the Repository

git clone https://github.com/Vamshiyadavk/AI-Healthcare_chatbot.git

2️⃣ Navigate to the Project Directory

cd AI-Healthcare_chatbot

3️⃣ Create a Virtual Environment (Optional)

python -m venv venv

4️⃣ Activate Virtual Environment

Windows

venv\Scripts\activate

Linux / Mac

source venv/bin/activate

5️⃣ Install Dependencies

pip install -r requirements.txt

6️⃣ Train the Machine Learning Model

python train_model.py

7️⃣ Run the Application

python app.py

8️⃣ Open in Browser

http://127.0.0.1:5000/

---

🧪 Future Improvements

- Integration with advanced NLP models
- Multi-language support
- User authentication system
- Doctor appointment integration
- Conversational chatbot interface
- Cloud deployment
- Improved dataset accuracy
- Integration with medical APIs

---

🎯 What I Learned

Through this project, I gained hands-on experience in:

- Building end-to-end Machine Learning applications
- Flask web development
- Frontend and backend integration
- Data preprocessing and model training
- Real-world AI application development
- Software deployment concepts

---

📈 Why This Project Stands Out

This project demonstrates:

- Practical use of Machine Learning in healthcare
- Full-stack application development
- Problem-solving using AI
- Data preprocessing and predictive analytics
- User-focused application design

---

⚠️ Disclaimer

This application is developed for educational and research purposes only. It is not intended to replace professional medical advice, diagnosis, or treatment. Users should always consult qualified healthcare professionals for medical concerns.

---

👨‍💻 Author

Vamshi k

B.Tech Data Science Student
Python Developer | Machine Learning Enthusiast

I enjoy building AI-driven applications that solve real-world problems and improve user experiences through technology.

This project represents my interest in combining Machine Learning with practical software development to create impactful solutions.

GitHub

https://github.com/Vamshiyadavk

⭐ Support

If you found this project useful or interesting, consider giving it a ⭐ on GitHub.

Your support motivates future improvements and more AI-based projects.
