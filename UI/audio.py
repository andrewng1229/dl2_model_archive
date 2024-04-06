from pydub import AudioSegment
from pydub.playback import play

def play_video_audio(video_file_path):
    # Load the video file and extract audio
    video_audio = AudioSegment.from_file(video_file_path, format="mp4")

    # Play the audio
    play(video_audio)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python script_name.py <video_file_path>")
        sys.exit(1)

    video_file_path = sys.argv[1]
    play_video_audio(video_file_path)