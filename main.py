import boto3
from fastapi import FastAPI, Request, Form, File, UploadFile, Cookie, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import starlette.status as status
import soundfile as sf
from scipy.io.wavfile import write
import io
from resemblyzer import preprocess_wav, VoiceEncoder
import numpy as np
import librosa
import logging
import sounddevice as sd
from botocore.exceptions import ClientError
import bcrypt
from starlette.middleware import Middleware
from starlette.middleware.sessions import SessionMiddleware
from starlette.authentication import (
    AuthenticationBackend, AuthCredentials, SimpleUser
)
from starlette.middleware.authentication import AuthenticationMiddleware
from starlette.authentication import requires
import os

class SessionAuthBackend(AuthenticationBackend):
    async def authenticate(self, conn):
        username = conn.session.get("username")
        if not username:
            return None

        scopes = []
        if conn.session.get("password_verified"):
            scopes.append("password")
        if conn.session.get("voice_verified"):
            scopes.append("voice")

        if not scopes:
            return None

        return AuthCredentials(scopes), SimpleUser(username)

app = FastAPI()
templates = Jinja2Templates(directory="templates")

dynamodb = boto3.client('dynamodb')
s3 = boto3.client('s3')
fs = 44100
secretkey = os.environ["SESSION_SECRET"]

app.add_middleware(AuthenticationMiddleware, backend=SessionAuthBackend())
app.add_middleware(SessionMiddleware, secret_key=secretkey)
TABLE_NAME = 'Users'
BUCKET_PREFIX = 'testbucket'

def fetch_audio_from_s3(bucket_name, folder_name):
    s3 = boto3.client('s3')
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=folder_name)
    wav_objects = [obj['Key'] for obj in response.get('Contents', []) if obj['Key'].endswith('.wav')]
    print(wav_objects)
    
    wavs = []
    for wav_object in wav_objects:
        obj = s3.get_object(Bucket=bucket_name, Key=wav_object)
        audio_bytes = obj['Body'].read()
        audio_data = io.BytesIO(audio_bytes)
        
        # Convert the BytesIO object to a numpy array
        wav, sr = librosa.load(audio_data, sr=None)
        
        # Preprocess the audio data using resemblyzer
        wav = preprocess_wav(wav, source_sr=sr)
        wavs.append(wav)
    
    return wavs


def upload_audio_to_s3(audio_bytes, bucket_name, file_name):
    s3 = boto3.client('s3')
    try:
        s3.put_object(Bucket=bucket_name, Key=file_name, Body=audio_bytes)
        return True
    except Exception as e:
        logging.error(f"Upload Failed: {str(e)}")
        return False
    
async def record_audio(seconds: int):
    # Record audio
    logging.debug("Recording...")
    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
    sd.wait()  # Wait until recording is finished
    logging.debug("Recording finished")

    # Convert recorded audio to bytes
    wav_bytes = io.BytesIO()
    write(wav_bytes, fs, myrecording)
    
    return wav_bytes.getvalue()


def delete_folder_from_s3(bucket_name, folder_path):
    s3 = boto3.client('s3')
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=folder_path)

    # Check if the folder exists
    if 'Contents' in response:
        for obj in response['Contents']:
            key = obj['Key']
            s3.delete_object(Bucket=bucket_name, Key=key)
        
        # Delete the folder itself
        s3.delete_object(Bucket=bucket_name, Key=folder_path)
        
        return True  # Deletion successful
    else:
        return False  # Folder does not exis



def compute_similarity(real_wavs, fake_wavs):
    encoder = VoiceEncoder()
    real_embeds = np.array([encoder.embed_utterance(wav) for wav in real_wavs])
    fake_embeds = np.array([encoder.embed_utterance(wav) for wav in fake_wavs])

    # Compute similarity scores between real and fake voices
    similarity_scores = (real_embeds @ fake_embeds.T).mean(axis=1)
    print(similarity_scores)
    return similarity_scores



@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.post("/signup")
async def signup(
    username: str = Form(...),
    password: str = Form(...),
    psw_repeat: str = Form(..., alias="psw-repeat"),
):
    if password != psw_repeat:
        return JSONResponse(content={"message": "Passwords do not match."}, status_code=400)
    
    password_bytes = password.encode("utf-8")
    hashed = bcrypt.hashpw(password_bytes, bcrypt.gensalt())

    try:
        dynamodb.put_item(
        
        TableName=TABLE_NAME,
        Item={
            'username': {'S': username},
            'Password': {'S': hashed.decode("utf-8")}
        },
        ConditionExpression="attribute_not_exists(username)"
    )
    except ClientError   as e:
        if e.response['Error']['Code'] == 'ConditionalCheckFailedException':
            return JSONResponse(content={"message": "Username already exists."}, status_code=409)
        logging.exception("Signup failed")
        return JSONResponse(content={"message": "Error occurred while signing up."}, status_code=500)

    

     
    return RedirectResponse("/login", status_code=status.HTTP_302_FOUND)
 
    

@app.get("/record", response_class=HTMLResponse)
@requires("password")
async def record_page(request: Request):
    return templates.TemplateResponse(request, "samplePage.html")



@app.post("/record")
@requires("password")
async def record(request: Request, duration: int = Form(...)):
    username = request.user.display_name  
    try:
        # Record audio
        audio_bytes = await record_audio(duration)
        
        # Upload audio to S3
        bucket_name = f"{BUCKET_PREFIX}-{username.lower().replace('_', '-')}"
        file_name = f"real.wav"  
        folder_path = "real/"  

        # Create bucket if it doesn't exist
        try:
            s3.create_bucket(Bucket=bucket_name, CreateBucketConfiguration={'LocationConstraint': 'eu-west-1'})
        except ClientError as e:
            if e.response['Error']['Code'] == 'BucketAlreadyOwnedByYou':
                pass  # Bucket already exists, proceed with the upload
            else:
                return JSONResponse(content={"message": f"Failed to create bucket: {str(e)}"}, status_code=500)
        
        # Upload the WAV data to S3
        if upload_audio_to_s3(audio_bytes, bucket_name, f"{folder_path}{file_name}"):
            return JSONResponse(content={"message": "Audio uploaded successfully."})
        else:
            return JSONResponse(content={"message": "Failed to upload audio."}, status_code=500)
    except Exception as e:
        # Log the exception details
        logging.exception("Error occurred while processing audio upload:")
        return JSONResponse(content={"message": f"Failed to upload audio: {str(e)}"}, status_code=500)

    
@app.get("/compare", response_class=HTMLResponse)
@requires("password", redirect="login_page")
async def compare_page(request: Request):
    return templates.TemplateResponse(request, "popup.html")

@app.post("/compare")
async def compare(request: Request, duration: int = Form(...)):
    username = request.cookies.get("username")
    if username:
        try:
            # Record audio
            audio_bytes = await record_audio(duration)
            
            # Upload audio to S3 in the fake folder
            bucket_name = f"{BUCKET_PREFIX}-{username.lower().replace('_', '-')}"
            folder_path = "fake/"
            file_name = "fake.wav"
            
            if upload_audio_to_s3(audio_bytes, bucket_name, f"{folder_path}{file_name}"):
                # Fetch real and fake audio files from S3
                real_wavs = fetch_audio_from_s3(bucket_name, "real/")
                fake_wavs = fetch_audio_from_s3(bucket_name, folder_path)

                # Compute similarity scores
                similarity_scores = compute_similarity(real_wavs, fake_wavs)

                if any(score < 0.8 for score in similarity_scores):
                    logging.debug("Deleting fake audio folder...")
                    delete_folder_from_s3(bucket_name, folder_path)
                    return JSONResponse(content={"redirect": "/login"})
                else:
                    delete_folder_from_s3(bucket_name, folder_path)
                    return JSONResponse(content={"redirect": "/record"})
            else:
                return JSONResponse(content={"message": "Failed to upload audio."}, status_code=500)
        except Exception as e:
            logging.error(f"Error occurred: {str(e)}")
            return JSONResponse(content={"message": f"Failed to upload audio: {str(e)}"}, status_code=500)
    else:
        return JSONResponse(content={"message": "Username cookie not found."}, status_code=400)



@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse(request, "login.html")
@app.post("/login")
async def login(request: Request, username: str = Form(...), password: str = Form(...)):
    # Query DynamoDB to check if user exists and password matches
    response = dynamodb.get_item(
        TableName=TABLE_NAME,
        Key={'username': {'S': username}}
    )
    
    if 'Item' in response:
        stored_hash = response['Item']['Password']['S'].encode("utf-8")
        if bcrypt.checkpw(password.encode("utf-8"), stored_hash):
            request.session.clear()
            request.session["username"] = username
            request.session["password_verified"] = True
            bucket_name = f"{BUCKET_PREFIX}-{username.lower().replace('_', '-')}"
            try:
                s3.head_bucket(Bucket=bucket_name)
                return RedirectResponse("/compare", status_code=status.HTTP_302_FOUND)
            except ClientError as e:
                if e.response['Error']['Code'] == '404':
                    return RedirectResponse("/record", status_code=status.HTTP_302_FOUND)
                else:
                    return JSONResponse(content={"message": f"Failed to check bucket: {str(e)}"}, status_code=500)
    
    # Authentication failed or user not found
    return JSONResponse(content={"message": "Invalid username or password"}, status_code=401)