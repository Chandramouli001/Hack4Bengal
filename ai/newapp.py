import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import random
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

# ======================
# 1. Create Career Dataset
# ======================
career_data = '''career_id,career_name,description,category,required_skills,avg_salary,demand
0,Computer Science Engineer,Design and develop software systems,Engineering,"Programming, Algorithms, Math",80000,High
1,Data Scientist,Analyze complex datasets,Tech,"Statistics, ML, Python",95000,Very High
2,Digital Marketer,Manage online campaigns,Business,"SEO, Social Media, Analytics",60000,High
3,Biomedical Engineer,Develop medical equipment,Medical,"Biology, Engineering, Design",75000,Medium
4,Graphic Designer,Create visual content,Arts,"Creativity, Adobe Tools, Design",50000,Medium
5,Financial Analyst,Analyze financial data,Business,"Accounting, Excel, Analysis",65000,High
6,AI Engineer,Build intelligent systems,Tech,"Python, ML, Deep Learning",110000,Very High
7,Environmental Scientist,Study environmental issues,Science,"Biology, Chemistry, Research",70000,Medium
8,Robotics Engineer,Build automated systems,Engineering,"Robotics, CAD, Electronics",90000,High
'''

with open('career_dataset.csv', 'w') as f:
    f.write(career_data)

# ======================
# 2. AI/ML Model Setup
# ======================
df = pd.read_csv('career_dataset.csv')

questions = [
    "Do you enjoy working with numbers and statistics?",
    "Are you interested in creative design?",
    "Do you like working with hardware/electronics?",
    "Are you passionate about environmental issues?",
    "Do you enjoy programming/coding?",
    "Are you interested in medical/biological fields?",
    "Do you prefer working with business strategies?",
    "Are you comfortable with advanced mathematics?"
]

def generate_synthetic_data(num_samples=1000):
    data = []
    for _ in range(num_samples):
        answers = []
        profile = random.choice(df['career_id'])
        
        if df.iloc[profile]['category'] == 'Tech':
            answers = [random.choice([1,1,1,0]), 0, 0, 0, 1, 0, 0, 1]
        elif df.iloc[profile]['category'] == 'Engineering':
            answers = [1, 0, 1, 0, 1, 0, 0, 1]
        elif df.iloc[profile]['category'] == 'Business':
            answers = [1, 0, 0, 0, 0, 0, 1, 0]
        elif df.iloc[profile]['category'] == 'Medical':
            answers = [0, 0, 0, 0, 0, 1, 0, 0]
        elif df.iloc[profile]['category'] == 'Arts':
            answers = [0, 1, 0, 0, 0, 0, 0, 0]
        elif df.iloc[profile]['category'] == 'Science':
            answers = [0, 0, 0, 1, 0, 1, 0, 1]
        
        answers = [a if random.random() > 0.1 else 1-a for a in answers]
        data.append(answers + [profile])
    
    return pd.DataFrame(data, columns=questions + ['career_id'])

synthetic_df = generate_synthetic_data(5000)
X = synthetic_df.iloc[:, :-1].values
y = synthetic_df.iloc[:, -1].values

le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# ======================
# 3. Neural Network Model
# ======================
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(len(df), activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train, y_train, 
                    epochs=50, 
                    validation_split=0.2,
                    batch_size=32,
                    verbose=1)

model.save('career_ai_model.h5')

# ======================
# 4. Career Counseling Agent (Modified)
# ======================
class CareerCounselor:
    def __init__(self, model_path, career_db_path):
        self.model = tf.keras.models.load_model(model_path)
        self.df = pd.read_csv(career_db_path)
        self.questions = questions
        self.le = LabelEncoder().fit(self.df['career_id'])
    
    def predict_career(self, answers):
        probabilities = self.model.predict(answers)[0]
        top_5 = np.argsort(probabilities)[-5:][::-1]
        return [(self.df.iloc[idx]['career_name'], probabilities[idx]) for idx in top_5]

# ======================
# 5. FastAPI Integration
# ======================
app = FastAPI()
counselor = CareerCounselor('career_ai_model.h5', 'career_dataset.csv')

@app.get("/", response_class=HTMLResponse)
async def serve_questionnaire(request: Request):
    html_content = """
   <html>
<head>
    <title>AI Career Counselor</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');

    body {
        font-family: 'Inter', sans-serif;
        margin: 0;
        padding: 0;
        background-color: #f8f9fa;
        color: #333;
    }

    .container {
        max-width: 800px;
        margin: 40px auto;
        background: white;
        border-radius: 12px;
        padding: 40px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }

    h1, h2, h3 {
        color: #2c3e50;
    }

    .question {
        margin-bottom: 24px;
        padding-bottom: 10px;
        border-bottom: 1px solid #e0e0e0;
    }

    .question p {
        font-size: 18px;
        font-weight: 500;
    }

    label {
        display: inline-block;
        margin-right: 20px;
        font-size: 16px;
    }

    button {
        margin-top: 20px;
        padding: 12px 28px;
        font-size: 16px;
        font-weight: 600;
        background-color: #007bff;
        border: none;
        border-radius: 8px;
        color: white;
        cursor: pointer;
        transition: background-color 0.3s ease;
    }

    button:hover {
        background-color: #0056b3;
    }

    .result {
        margin-top: 24px;
        padding: 20px;
        background-color: #e9f7ef;
        border-left: 5px solid #28a745;
        border-radius: 8px;
    }

    .result h3 {
        margin: 0;
        font-size: 20px;
        color: #2e7d32;
    }

    .result p {
        margin: 4px 0;
    }

    .hidden {
        display: none;
    }

    #resultsDiv h2 {
        margin-top: 40px;
        margin-bottom: 16px;
        font-size: 24px;
        border-bottom: 2px solid #007bff;
        display: inline-block;
    }
</style>

</head>
<body>
<div class="container">
    <h1>AI Career Counselor Questionnaire</h1>
    
    <!-- Div for questions -->
    <div id="questionDiv">
        <form id="careerForm">
            <!-- Questions will be dynamically inserted here by JavaScript -->
        </form>
        <!-- Use regular button here, not type="submit" -->
        <button id="submitBtn" type="button">Get Recommendations</button>
    </div>
    
    <!-- Div for results (recommendations) -->
    <div id="resultsDiv" class="hidden">
        <!-- Recommendations will be dynamically inserted here by JavaScript -->
    </div>
   </div>
    <script>
        const questions = [
            "Do you enjoy working with numbers and statistics?",
            "Are you interested in creative design?",
            "Do you like working with hardware/electronics?",
            "Are you passionate about environmental issues?",
            "Do you enjoy programming/coding?",
            "Are you interested in medical/biological fields?",
            "Do you prefer working with business strategies?",
            "Are you comfortable with advanced mathematics?"
        ];

        const form = document.getElementById('careerForm');
        const questionDiv = document.getElementById('questionDiv');
        const resultsDiv = document.getElementById('resultsDiv');

        // Dynamically add questions
        questions.forEach((q, index) => {
            const div = document.createElement('div');
            div.className = 'question';
            div.innerHTML = `
                <p>${q}</p>
                <label>
                    <input type="radio" name="q${index}" value="1" required> Yes
                </label>
                <label>
                    <input type="radio" name="q${index}" value="0" required> No
                </label>
            `;
            form.appendChild(div);
        });

        // Handle the button click (not form submission)
        document.getElementById('submitBtn').addEventListener('click', async (e) => {
            e.preventDefault();  // Prevent form submission
            const formData = new FormData(form);
            const answers = [];
            for (let i = 0; i < questions.length; i++) {
                const val = formData.get(`q${i}`);
                answers.push(val !== null ? parseInt(val) : 0);
            }

            // Hide the questions and show the results
            questionDiv.style.display = 'none';
            resultsDiv.classList.remove('hidden');

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ answers }),
                });
                const data = await response.json();
                resultsDiv.innerHTML = '<h2>Top Career Recommendations</h2>';
                data.predictions.forEach((career, index) => {
                    resultsDiv.innerHTML += `
                        <div class="result">
                            <h3>${index + 1}. ${career.career_name} (${(career.probability * 100).toFixed(1)}%)</h3>
                            <p><strong>Category:</strong> ${career.category}</p>
                            <p><strong>Skills Required:</strong> ${career.required_skills}</p>
                            <p><strong>Average Salary:</strong> $${career.avg_salary}</p>
                            <p><strong>Demand:</strong> ${career.demand}</p>
                        </div>
                    `;
                });
            } catch (error) {
                console.error('Error:', error);
                resultsDiv.innerHTML = '<p>Error fetching recommendations. Please try again.</p>';
            }
        });
    </script>
</body>
</html>

    """
    return HTMLResponse(content=html_content)

@app.post("/predict")
async def predict_careers(request_data: dict):
    answers = request_data.get('answers', [])
    if len(answers) != 8:
        return JSONResponse(
            content={"error": "Exactly 8 answers are required."},
            status_code=400
        )
    input_data = np.array([answers], dtype=np.int32)
    predictions = counselor.predict_career(input_data)
    results = []
    for career_name, prob in predictions:
        career_details = counselor.df[counselor.df['career_name'] == career_name].iloc[0]
        results.append({
            "career_name": career_name,
            "probability": float(prob),
            "category": career_details['category'],
            "required_skills": career_details['required_skills'],
            "avg_salary": int(career_details['avg_salary']),
            "demand": career_details['demand']
        })
    return {"predictions": results}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
