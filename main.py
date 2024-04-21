import boto3
from fastapi import FastAPI, Request, Form, File, UploadFile, Cookie, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import starlette.status as status
import os


app = FastAPI()
templates = Jinja2Templates(directory="templates")

dynamodb = boto3.client('dynamodb')
s3 = boto3.client('s3')


TABLE_NAME = 'Users'
BUCKET_PREFIX = 'testbucket'


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
        # Set a cookie containing the username
        return RedirectResponse("/samplePage", status_code=status.HTTP_302_FOUND, headers={"Set-Cookie": f"username={username}; Path=/;"})
    else:
        return {"message": "Error occurred while signing up."}  
    

@app.get("/samplePage", response_class=HTMLResponse)
async def samplePage(request: Request):
    return templates.TemplateResponse("samplePage.html", {"request": request})


@app.post("/record")
async def record(request: Request, audio: UploadFile = File(...)):
    username = request.cookies.get("username")  
    if username:
        bucket_name = f"{BUCKET_PREFIX}-{username.lower().replace('_', '-')}"
        try:
            s3.create_bucket(Bucket=bucket_name, CreateBucketConfiguration={'LocationConstraint': 'eu-west-1'})
            file_extension = os.path.splitext(audio.filename)[-1]
            file_name = f"real{file_extension}"  # Use the appropriate file extension
            audio_bytes = await audio.read()
            s3.put_object(Bucket=bucket_name, Key=file_name, Body=audio_bytes)
            return JSONResponse(content={"message": "Audio uploaded successfully."})
        except Exception as e:
            return JSONResponse(content={"message": f"Failed to upload audio: {str(e)}"}, status_code=500)
    else:
        return JSONResponse(content={"message": "Username cookie not found."}, status_code=400)

@app.get("/compare", response_class=HTMLResponse)
async def compare(request: Request):
    return {"message": "Compare page"}


