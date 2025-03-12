import io
import uuid
from fastapi import UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from src.repository.S3Repository import S3Repository

class S3Service:
   def __init__(self):
      self.s3_model = S3Repository()
      self.allowed_types = ['jpg', 'jpeg', 'png', 'gif']

   # 이미지 업로드 처리
   async def upload_image(self, file: UploadFile):
      # 파일 유효성 검사
      file_type = file.filename.split('.')[-1].lower()
      if file_type not in self.allowed_types:
         raise HTTPException(status_code=400, detail="허용되지 않은 파일 형식")

      # 고유한 S3 키 생성
      # images/324f9fb1-9a56-4304-baad-c40d443db886.png
      s3_key = f"images/{uuid.uuid4()}.{file_type}"
      print("S3Service:s3_key: ", s3_key)

      # S3에 업로드
      try:
         file_url = await self.s3_model.upload_file(file, s3_key)
         return {
            "message": "upload success",
            "file_url": file_url,
            "file_name": file.filename
         }
      except Exception as e:
         raise HTTPException(status_code=500, detail=str(e))
   

   # 이미지 다운로드 처리
   async def get_image(self, image_name: str):
      s3_key = f"images/{image_name}"
      try:
         file_content, content_type = await self.s3_model.download_file(s3_key)
         return StreamingResponse(
            io.BytesIO(file_content),
            media_type = content_type,
            headers = {"Content-Disposition": f"attachment; filename={image_name}"}
         )
      except Exception as e:
         raise HTTPException(status_code=500, detail=str(e))


s3_service = S3Service()
