from __future__ import annotations

import requests  # type: ignore


async def download_file(file_url: str, file_path: str):
    print(f'Downloading file {file_url} to {file_path}...')
    try:
        response = requests.get(file_url, stream=True)
        response.raise_for_status()

        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print(f'Model downloaded successfully to {file_path}')
    except requests.exceptions.RequestException as e:
        print(f'Error downloading model: {e}')
