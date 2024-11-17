#export_HF_token = "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"
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from transformers import pipeline
import random
import uvicorn
from pydantic import BaseModel
from transformers import pipeline

# Initialize FastAPI app
app = FastAPI()


# Model Pipeline Generated
qa_pipeline = pipeline("question-answering")
quote_pipeline = pipeline("text-generation", model="huggingtweets/_buddha_quotes")

class QuestionRequest(BaseModel):
    question: str


from fastapi.middleware.cors import CORSMiddleware
# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to your specific frontend origin if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)




# Asynchronous endpoint to QuestionAnswer
@app.post("/question-answer")
async def question_answer(request: QuestionRequest):
    question = request.question
    context = """Nestled in the vibrant city of Bhubaneswar, Pradeep Parida, fondly known as Gopal, embodies an extraordinary blend of curiosity, determination, and versatility. Raised in a supportive family alongside his five elder sisters—each carving their own path to success—Pradeep always believed in dreaming big while staying grounded. With a father, AB Parida, and a mother, Ahalya Parida, instilling in him the values of resilience and continuous learning, his journey is a testament to how passion and hard work can open doors to countless possibilities.
Pradeep's academic journey began at Govt. High School, Bhubaneswar, followed by Rajdhani College, where he pursued science in his 12th grade. This foundational love for technology led him to earn his B.Tech in Electrical Engineering from BPUT and later a Master's in Automation and Process Control from CTTC. During his engineering years, his zeal for innovation saw him contributing to projects certified by DRDO, laying the groundwork for his technical expertise.
Pradeep's career began in the field of sales engineering at Alfa India Pvt. Ltd., where he achieved a remarkable 45% revenue growth in a single year. He even spearheaded an energy audit for Coca-Cola India, identifying cost-saving opportunities in their manufacturing plant. However, his curiosity for data and analytics steered him toward a new path.
At Remotasks, he delved into data annotation and analysis, mastering LiDAR techniques for self-driving cars and collaborating with tech giants like Uber and P&G. This experience sharpened his ability to derive meaningful insights from data, setting the stage for his role as a Data Scientist.
Today, Pradeep thrives as a Data Scientist at CSW Alliance Pvt. Ltd., where his work speaks volumes. From achieving 89.3% accuracy in sentiment analysis to predicting customer buying behavior with a 15% improvement in accuracy, his contributions have directly impacted marketing and business strategies. His proficiency in Python, Tableau, and NLP has enabled him to craft solutions that resonate with stakeholders and enhance decision-making processes.
Pradeep’s passion for learning is evident in his virtual internships with British Airways, TATA Insights & Quants, and KPMG, where he gained hands-on experience in cost analysis, predictive modeling, and customer segmentation.
Beyond his professional achievements, Pradeep has donned multiple hats, including volunteering as an IT Specialist for DAV Public School and supporting startups like Desire IT Solutions with project management. These roles reflect his innate ability to adapt, learn, and lead in diverse environments.
Pradeep is an avid tech enthusiast with expertise spanning AutoCAD, PLC programming, and cutting-edge technologies like Deep Learning and AI. His certifications, including those from KPMG, freeCodeCamp, and HackerRank, underscore his commitment to staying ahead in the fast-evolving world of data science.
When he's not working, Pradeep loves indulging in hobbies like painting and exploring thriller and action movies. These interests not only fuel his creativity but also provide a well-rounded perspective that he channels into his professional life.
Driven by a passion for leveraging data to solve real-world challenges, Pradeep envisions contributing to innovative projects that blend technical prowess with strategic insights. Whether it’s enhancing customer satisfaction, optimizing processes, or pushing the boundaries of AI, his journey reflects a relentless pursuit of excellence.
Pradeep "Gopal" Parida is not just a professional—he is a storyteller, a visionary, and a changemaker, ready to leave an indelible mark in the world of data science."""
    panswer = qa_pipeline(question=question, context=context)
    return JSONResponse(content={"answer": panswer['answer']})



# Asynchronous endpoint to generate quotes
@app.get("/generate-quote")
async def generate_quote():
    topics = ["Life", "Career", "Goal", "Achievement", "Love", "Nature", "Technology", "Belief"]
    chosen_topic = random.choice(topics)
    result = quote_pipeline(chosen_topic, num_return_sequences=1)
    return JSONResponse(content={"quote": result[0]['generated_text'] } )


# Run the server when executed directly
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
