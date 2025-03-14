import os
import boto3
from botocore.exceptions import ClientError
from fastapi import UploadFile, HTTPException

# env 파일 로드
from dotenv import load_dotenv
load_dotenv()

class S3Repository:
   def __init__(self):
      self.bucket_name = os.getenv("S3_BUCKET_NAME")
      self.region = os.getenv("AWS_REGION")
      self.s3_client = boto3.client(
         's3', 
         aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID"),
         aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY"),
         region_name = self.region
      )
   
   # S3에 파일 업로드
   async def upload_file(self, file: UploadFile, s3_key: str) -> str:
      try:
         self.s3_client.upload_fileobj(
            file.file,
            self.bucket_name,
            s3_key,
            ExtraArgs={'ContentType': file._headers.get('content-type')}
         )
         return f"https://{self.bucket_name}.s3.{self.region}.amazonaws.com/{s3_key}"
      except ClientError as e:
         raise HTTPException(status_code=500, detail=str(e))
   

   # S3에서 파일 가져오기
   async def download_file(self, s3_key: str) -> tuple[bytes, str]:
      try:
         file_obj = self.s3_client.get_object(
            Bucket=self.bucket_name,
            Key=s3_key
         )
         return file_obj['Body'].read(), file_obj.get('ContentType', 'image/jpeg')
      except ClientError as e:
         if e.response['Error']['Code'] == 'NoSuchKey':
            raise HTTPException(status_code=404, detail="이미지가 없음")
         raise HTTPException(status_code=500, detail=str(e))
