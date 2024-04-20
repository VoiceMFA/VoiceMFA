import boto3
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
import starlette.status as status

app = FastAPI()
templates = Jinja2Templates(directory="templates")

dynamodb = boto3.client('dynamodb')

TABLE_NAME = 'Users'

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/signup")
async def signup(username: str = Form(...), password: str = Form(...)):
    response = dynamodb.put_item(
        TableName=TABLE_NAME,
        Item={
            'username': {'S': username},  # Assuming email is the username
            'Password': {'S': password}
        }
    )
    
    if response['ResponseMetadata']['HTTPStatusCode'] == 200:
        return RedirectResponse("/samplePage", status_code=status.HTTP_302_FOUND)  
    else:
        return {"message": "Error occurred while signing up."}
    

@app.get("/samplePage", response_class=HTMLResponse)
async def samplePage(request: Request):
    return templates.TemplateResponse("samplePage.html", {"request": request})

@app.get("/record", response_class=HTMLResponse)
async def record(request: Request):
    return {"message": "Record page"}

@app.get("/compare", response_class=HTMLResponse)
async def compare(request: Request):
    return {"message": "Compare page"}


