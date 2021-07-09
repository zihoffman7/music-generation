from music21 import converter, interval, pitch
import os

def mid_to_abc(dir):
    for file in os.listdir(dir):
        print(file)
        # Ensure the file is a midi file
        if not file.split(".")[-1].lower() == "mid" and not file.split(".")[-1].lower() == "midi":
            continue
        path = f"{dir}/{file}"
        fname = path.split("/")[-1].split(".")[0]
        # Load midi file
        print("loading")
        s = converter.parse(path).makeNotation()
        # Attempt to transpose the file to C major
        print("transposing")
        s = s.transpose(interval.Interval(s.analyze("key").tonic, pitch.Pitch("C")))
        # Convert music21 stream object to MusicXML
        print("writing to xml")
        s.write("xml", fp=f"{fname}.xml")
        # Convert MusicXML to abc using xml2abc
        print("converting to abc")
        os.system(f"python xml2abc.py {fname}.xml -o data/abc")
        # Remove the temporary MusicXML file
        os.remove(f"{fname}.xml")
    print("Done!")

def output_to_abc(output, key):
    abc = f"""
X:1
T:Generated Music
C:Test
L:1/4
Q:1/4=120
M:4/4
K:C
{" ".join(output)}"""
    fname = f"output-{len([i for i in os.listdir('output') if i.split('.')[-1] == 'mid']) + 1}"
    with open(f"output/{fname}.abc", "w") as f:
        f.write(abc)
    abc_to_mid(f"output/{fname}.abc", f"output/{fname}.mid", key)

def abc_to_mid(abc_fname, mid_fname, key):
    s = converter.parse(abc_fname)
    s = s.transpose(interval.Interval(s.analyze("key").tonic, pitch.Pitch(key)))
    s.write("mid", fp=mid_fname)
    os.remove(abc_fname)
