from flask import Flask, render_template, request, jsonify
import random

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index_v2.html')

@app.route('/index_v2')
def index_v2():
    return render_template('index_v2.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    skills = data.get('skills', '')
    interests = data.get('interests', '')
    aptitude = int(data.get('aptitude', 0))

    sample_careers = ['Data Scientist', 'AI Engineer', 'Web Developer', 'Cloud Architect', 'Cybersecurity Analyst']
    recommendations = random.sample(sample_careers, 3)
    scores = [random.randint(70, 100) for _ in recommendations]

    return jsonify({'recommendations': recommendations, 'scores': scores})

if __name__ == '__main__':
    app.run(debug=True)
