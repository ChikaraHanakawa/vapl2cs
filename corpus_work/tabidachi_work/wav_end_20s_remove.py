import wave
import os

def remove_last_15_seconds(file_path):
    with wave.open(file_path, 'rb') as wf:
        # Get parameters
        params = wf.getparams()
        framerate = wf.getframerate()
        nframes = wf.getnframes()
        
        # Calculate the number of frames to keep (total frames - frames in 15 seconds)
        frames_to_keep = nframes - (15 * framerate)
        
        if frames_to_keep > 0:
            # Read the frames to keep
            wf.rewind()
            frames = wf.readframes(frames_to_keep)
            
            # Write the frames to a new file
            with wave.open(file_path, 'wb') as wf_out:
                wf_out.setparams(params)
                wf_out.writeframes(frames)
            print(f"Trimmed audio saved to {file_path}")
        else:
            print("Audio file is shorter than 15 seconds, no trimming needed.")

# Example usage
file_path = "307_1_3_zoom.wav"
remove_last_15_seconds(file_path)