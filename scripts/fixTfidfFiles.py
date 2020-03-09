import os

# path to folder with TF-IDF doc files
path = '/home/pfb16181/Music/pubmedindex171819_454072.allDocids.txt.docvector.TF_IDF'

files = []


def check(text):
    for char in text:
        if not (char.isdigit() or char == "." or char == " " or char == "\n"):
            return(False)
    return(True)


# r=root, d=directories, f = files
for r, d, f in os.walk(path):
    for file in f:
        s = os.path.join(r, file)
        files.append(s)


# for file in files:
#    print(file)

for file in files:
    with open(file, "r") as f:
        lines = f.readlines()
    with open(file, "w") as f:
        for i, line in enumerate(lines):
            if i != 1:
                f.write(line)
            else:
                if check(line) == True:
                    continue
                else:
                    f.write(line)

print('ok')
