import os
import json
import csv

def read_json_files(directory):
    data = []
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
                data.extend(json.load(file)["response"]["docs"])
    return data

def write_to_csv(data, csv_file):
    with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Section Name', 'Headline'])
        for doc in data:
            writer.writerow([doc.get("section_name", ""), doc["headline"]["main"]])
json_directory = r"C:\Users\PavelAgafonov\corus\pythonProject1\Archive"

json_data = read_json_files(json_directory)
csv_file = "out.csv"
write_to_csv(json_data, csv_file)
