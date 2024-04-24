import boto3
from io import BytesIO
from resemblyzer import preprocess_wav, VoiceEncoder
import numpy as np

def fetch_audio_from_s3(bucket_name, folder_name):
    s3 = boto3.client('s3')
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=folder_name)
    wav_objects = [obj['Key'] for obj in response.get('Contents', []) if obj['Key'].endswith('.wav')]
    print(wav_objects)
    
    wavs = []
    for wav_object in wav_objects:
        obj = s3.get_object(Bucket=bucket_name, Key=wav_object)
        audio_bytes = obj['Body'].read()
        audio_data = BytesIO(audio_bytes)
        wav, sr = librosa.load(audio_data, sr=None)
        wav = preprocess_wav(wav, source_sr=sr)
        wavs.append(wav)
    
    return wavs

def compute_similarity():
    bucket_name = 'realvoices'
    real_wavs = fetch_audio_from_s3(bucket_name, 'real')
    fake_wavs = fetch_audio_from_s3(bucket_name, 'fake')

    if len(real_wavs) == 0 or len(fake_wavs) == 0:
        return "Insufficient audio files in either 'real' or 'fake' folder."

    encoder = VoiceEncoder()
    real_embed = encoder.embed_utterance(real_wavs[0])  # Assuming there's only one file in each folder
    fake_embed = encoder.embed_utterance(fake_wavs[0])

    similarity_score = np.dot(real_embed, fake_embed)
    
    return similarity_score


def handler(event, context):
    # Main Lambda handler function
    bucket_name = 'realvoices'
    real_wavs = fetch_audio_from_s3(bucket_name, 'real')
    fake_wavs = fetch_audio_from_s3(bucket_name, 'fake')

    similarity_score = compute_similarity(real_wavs, fake_wavs)
    print(f"Similarity score: {similarity_score}")

    return {
        'statusCode': 200,
        'body': f"Similarity score: {similarity_score}"
    } 