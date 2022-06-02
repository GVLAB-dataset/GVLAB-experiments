import csv
import json

from config import SWOW_SPLIT_PATH, SWOW_DATA_PATH

################## Loading CSV ####################
csv_rows = []
with open(SWOW_DATA_PATH, 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        csv_rows.append(row)

csv_rows = csv_rows[1:]

data = list(
    map(lambda row: {"id": row[0], 'cue': row[1], 'candidates': json.loads(row[6]), 'associations': json.loads(row[2])},
        csv_rows))

################## Transforming CSV ######################

cache = set()
swow_split_rows = []


def get_row_id(image, cue):
    return image + '-' + cue


def insert_row(image, cue, label):
    id = get_row_id(image, cue)
    if id not in cache:
        swow_split_rows.append({'image': image + '.jpg', 'cue': cue, 'label': label})
        cache.add(id)


for row in data:
    cue = row['cue']
    for image in row['associations']:
        insert_row(image, cue, 1)
    unassociated_images = set(row['candidates']) - set(row['associations'])
    for image in unassociated_images:
        insert_row(image, cue, 0)

json.dump(swow_split_rows, open(SWOW_SPLIT_PATH, 'w+'), indent=4)
