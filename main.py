from fastapi import FastAPI
from mangum import Mangum
from compareVoice import compute_similarity

app = FastAPI()
handler = Mangum(app)

@app.get("/")
async def test():
    similarity_score = compute_similarity()
    formatted_score = f"{similarity_score:.4f}"  
    return {"Similarity_Score": float(formatted_score)}
