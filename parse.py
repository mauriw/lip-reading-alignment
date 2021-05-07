import numpy as np
from scipy import linalg



def parse_file(file):
    sections = file.split('\n\n')
    res = []
    frames = sections[1].split('\n')[1:]
    timestamps = sections[2].split('\n')[1:]

    for row in frames:
        z = list(map(float, row.split('\t')))
        z[0] = int(z[0])
        z.append(0)
        res.append(z)

    for row in timestamps[:-1]:
        u = list(map(float, row.split()[1:]))
        res[int(u[0] * 25)][-1] = 1
        res[int(u[1] * 25)][-1] = 1
    
    return res



import csv

for i in range(1,57):
    try:
        with open(f"pretrain/0af00UcTOSc/{str(i).zfill(5)}.txt", "r") as f:
            g = f.read()
    except:
        continue

    res = parse_file(g)
    with open(f"{str(i).zfill(5)}.csv", "w") as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',')
        spamwriter.writerow(["frame", "x","y","w","h","transition"])
        for i in res:
            spamwriter.writerow(i)
