from datasets import load_dataset
import subprocess

ds = load_dataset("google/MusicCaps")

# pip install yt-dlp

item = ds["train"][0]

ytid = item["ytid"]
start = item["start_s"]
end = item["end_s"]

url = f"https://www.youtube.com/watch?v={ytid}"



'''
subprocess.run([
    "yt-dlp",
    "-x",
    "--audio-format",
    "wav",
    "-o",
    f"./MusicCaps/{ytid}.wav",
    url
])

'''