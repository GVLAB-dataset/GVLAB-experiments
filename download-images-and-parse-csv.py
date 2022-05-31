import csv
import requests



# Extracting CSV rows
import json
import os

print('Starting CSV parse')

csv_rows = []
with open('assets/gvlab_swow_split.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        csv_rows.append(row)

csv_rows = csv_rows[1:]


# Parse CSV data and save to JSON file
data = list(map(lambda row: {"id": row[0], 'cue': row[1], 'candidates': json.loads(row[6]), 'associations': json.loads(row[2])}, csv_rows))
json.dump(data, open( './assets/data.json', 'w+'), indent=4)

print('CSV parse finished')

# Download all images
print('Starting image download')

images = set()
for association in data:
    images = images.union(set(association['candidates']))
    images = images.union(set(association['associations']))


images_directory = os.path.join(os.path.dirname(__file__), 'assets/images')
if not os.path.exists(images_directory):
    os.makedirs(images_directory)

for index, image in enumerate(images):
    image_name = image+'.jpg'
    image_path = os.path.join(images_directory, image_name)
    image_url = 'https://gvlab-bucket.s3.amazonaws.com/{}'.format(image_name)

    image_file = open(image_path, 'wb+')
    image_file.write(requests.get(image_url).content)
    image_file.close()

    if (index+1) % 5 == 0:
        print(str(index+1) + ' images downloaded')

print('Image download finished')
