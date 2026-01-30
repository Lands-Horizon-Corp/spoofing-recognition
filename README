# Spoofing Detection API

## Deployment

### Configuration (.env)
add the env file on spoofing_detection_api folder
```env variables

PROJECT_NAME="Spoof Detection API"
IS_INDEVELOPMENT=False
# False = Production mode (hides /docs, strict CORS)
# True = Dev mode (shows /docs, open CORS)

# or * to allow all
CORS_ALLOW_ORIGINS="http://localhost:3000,https://your-frontend.com"

```
Note: cors list should only be comma separated, no space in-between

### download the model and params

download[ model.pt and params.json](https://drive.google.com/drive/folders/1I4ywUHzyxI9t9KITu5LOtlNVnx6Gb8Iu?usp=sharing), place it on spoofing_detection_api/models/

[GDrive link](https://drive.google.com/drive/folders/1I4ywUHzyxI9t9KITu5LOtlNVnx6Gb8Iu?usp=sharing)

### Docker

```cmd
docker compose up --build
```

### Baremetal Deployment

install external libraries list out on the requirements.txt

```cmd
pip install -r requirements.txt
```

run this so spoofing_detection_api can use the utilities on src
```cmd
pip install -e
```

### Fast API
run the following

```cmd
uvicorn app.main:app --app-dir spoofing_detection_api --host 0.0.0.0 --port 8002 --reload
```







# Training
all of the code use for training is on the notebook directory.

dataset source

https://github.com/ZhangYuanhan-AI/CelebA-Spoof
