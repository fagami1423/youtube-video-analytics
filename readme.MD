# Youtube Video Analysis using Social Media Analytics

Project Structure 
-- config.py 
-- main.py 
-- videoLinks.csv
-- youtube.ipynb

## Steps to follow:
1. Please follow the process https://www.geeksforgeeks.org/youtube-data-api-set-1/ to get the API key

After Getting API key
* Create a config.py file 
```
DEVELOPER_KEY = "Your API KEY"
YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"
```

1. Installation
```
pip install --upgrade google-api-python-client
```

3. Run the script or jupyter notebook file

```
python main.py
```