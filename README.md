# Spoofing Detection API

## Deployment

### for devs

```cmd
docker compose -f docker-compose.dev.yaml up
```

### Configuration (.env)

add the env file on root folder

```env variables

PROJECT_NAME="Spoof Detection API"
APP_ENV=development
IS_LOCAL=True
# False = Production mode (hides /docs, strict CORS)
# True = Dev mode (shows /docs, open CORS)


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
pip install -r requirements-dev.txt
```

run this so spoofing_detection_api can use the utilities on src

```cmd
pip install -e
```

### Robyn

run the following

```cmd
python spoofing_detection_api/app/main.py --worker=1 --processes=1
```

more info for [robyn deployment](https://robyn.tech/documentation/en/example_app/deployment)

# Training

all of the code use for training is on the notebook directory.

dataset source

https://github.com/ZhangYuanhan-AI/CelebA-Spoof

## Results

```
  'test/acc': 0.9210000038146973,
  'test/precision': 0.9040306806564331,
  'test/recall': 0.9419999718666077,
  'test/f1': 0.9226248860359192,
  'test/apcer': 0.058000028133392334,
  'test/bpcer': 0.10000002384185791,
```

# Future Plans

- [ ] try central diff conv
- [ ] try patch based CNN maybe facial features as input (eyes, lips nose, etc) then media pipe running on client side
