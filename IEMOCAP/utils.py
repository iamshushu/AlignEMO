"""IEMOCAP feature extraction + collate functions (text/audio/video batching)."""
import torch
from transformers import AutoProcessor, AutoImageProcessor
from transformers import AutoTokenizer, AutoFeatureExtractor
import librosa
import cv2
import numpy as np
import os

# Encoder selection (override via env var to keep call sites untouched):
#   IEMOCAP_TEXT_MODEL,  IEMOCAP_AUDIO_PROCESSOR
_TEXT_MODEL = os.environ.get("IEMOCAP_TEXT_MODEL", "roberta-large")
_AUDIO_PROCESSOR_MODEL = os.environ.get("IEMOCAP_AUDIO_PROCESSOR", "facebook/data2vec-audio-base-960h")

# WavLM/HuBERT pretraining checkpoints ship only a FeatureExtractor
# (no tokenizer), so AutoProcessor fails. Fall back to AutoFeatureExtractor.
try:
    audio_processor = AutoProcessor.from_pretrained(_AUDIO_PROCESSOR_MODEL)
except (OSError, ValueError, KeyError):
    audio_processor = AutoFeatureExtractor.from_pretrained(_AUDIO_PROCESSOR_MODEL)
video_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")
roberta_tokenizer = AutoTokenizer.from_pretrained(_TEXT_MODEL)
speaker_list = ['<s1>', '<s2>', '<s3>', '<s4>', '<s5>', '<s6>', '<s7>', '<s8>', '<s9>']
speaker_tokens_dict = {'additional_special_tokens': speaker_list}
roberta_tokenizer.add_special_tokens(speaker_tokens_dict)

def encode_right_truncated(text, tokenizer, max_length=511):
    tokenized = tokenizer.tokenize(text)
    truncated = tokenized[-max_length:]    
    ids = tokenizer.convert_tokens_to_ids(truncated)
    return ids + [tokenizer.mask_token_id]


def padding_(ids_list, tokenizer):
    max_len = 0
    for ids in ids_list:
        if len(ids) > max_len:
            max_len = len(ids)
    
    pad_ids = []
    attention_masks = []
    for ids in ids_list:
        pad_len = max_len-len(ids)
        add_ids = [tokenizer.pad_token_id for _ in range(pad_len)]
        pad_ids.append(add_ids+ids)
    return torch.tensor(pad_ids), torch.tensor(attention_masks)

def padding(ids_list, tokenizer):
    max_len = 0
    for ids in ids_list:
        if len(ids) > max_len:
            max_len = len(ids)
    
    pad_ids = []
    attention_masks = []
    for ids in ids_list:
        pad_len = max_len-len(ids)
        add_ids = [tokenizer.pad_token_id for _ in range(pad_len)]
        attention_mask = [ 1 for _ in range(len(ids))]
        add_attention = [ 0 for _ in range(len(add_ids))]
        pad_ids.append(add_ids+ids)
        attention_masks.append(add_attention+attention_mask)
    return torch.tensor(pad_ids), torch.tensor(attention_masks)

def padding_video(batch):
    max_len = 0
    for ids in batch:
        if len(ids) > max_len:
            max_len = len(ids)
    
    pad_ids = []
    for ids in batch:
        pad_len = max_len-len(ids)
        add_ids = [ 0 for _ in range(pad_len)]
        
        pad_ids.append(add_ids+ids.tolist())
    
    return torch.tensor(pad_ids)

def get_audio(processor, path):
    audio, rate = librosa.load(path)

    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")

    return inputs["input_values"][0]
    
def get_video(feature_extractor, path, wav_path, start_time, end_time):
    jpg_folder = wav_path.split('/')[-1].split('.')[0]
    video_folder = path.split('.avi')[0]
    video_folder = os.path.dirname(video_folder)
    jpg_path = os.path.join(video_folder, jpg_folder)
    frames = []
    cropped_images = []
    try:
        files = os.listdir(jpg_path)
        frame_paths = sorted([os.path.join(jpg_path, f) for f in files])
        for frame_path in frame_paths:
            frame = cv2.imread(frame_path)
            frames.append(frame)
        if len(frames) >= 8:
            frames = frames[:8]
            inputs = feature_extractor(frames[:8], return_tensors="pt")        
        else:
            lack = 8 - len(frames)
            extend_frames = [ frames[-1].copy() for _ in range(lack)]
            frames.extend(extend_frames)  
            frames = frames[:8]
            inputs = feature_extractor(frames[:8], return_tensors="pt")

        return inputs["pixel_values"][0]
        
    except Exception as e:
        print(f"get_video error in get_video_bbox /path - {path}, error: {str(e)}")
        return torch.zeros((8, 3, 224, 224))

def get_video_(feature_extractor, path, start_time, end_time):
    video = cv2.VideoCapture(path)
    fps = video.get(cv2.CAP_PROP_FPS)
    time_start = max(float(start_time)-1, 0)
    time_end = float(end_time)+1
    length = fps*(time_end - time_start)
    frames = []
    step = length // 8
    count = 0

    try:
        while(video.isOpened()):
            ret, image = video.read()
            if(ret==False):
                break
            if(int(video.get(1)) >= fps*time_start) and (int(video.get(1)) <= fps*time_end):
                count += 1
                if count % step == 0:
                    frames.append(image)
        video.release()
        if len(frames) >= 8:
            inputs = feature_extractor(frames[:8], return_tensors="pt")
            return inputs["pixel_values"][0]      
        else:
            lack = 8 - len(frames)
            extend_frames = [ frames[-1].copy() for _ in range(lack)]
            frames.extend(extend_frames)  
            inputs = feature_extractor(frames[:8], return_tensors="pt")
            return inputs["pixel_values"][0]

    except:
        print(f"get_video error : path - {path}, start_time - {start_time}, end_time - {end_time}, len : {len(frames)}")


def get_video_bbox(feature_extractor, path, wav_path, start_time, end_time, bbox):
    jpg_folder = wav_path.split('/')[-1].split('.')[0]
    video_folder = path.split('.avi')[0]
    video_folder = os.path.dirname(video_folder)
    jpg_path = os.path.join(video_folder, jpg_folder)
    frames = []
    cropped_images = []
    try:
        files = os.listdir(jpg_path)
        frame_paths = sorted([os.path.join(jpg_path, f) for f in files])
        for frame_path in frame_paths:
            frame = cv2.imread(frame_path)
            frames.append(frame)
        if len(frames) >= 8:
            frames = frames[:8]
            inputs = feature_extractor(frames[:8], return_tensors="pt")        
        else:
            lack = 8 - len(frames)
            extend_frames = [ frames[-1].copy() for _ in range(lack)]
            frames.extend(extend_frames)  
            frames = frames[:8]
            inputs = feature_extractor(frames[:8], return_tensors="pt")

        frame_num = 0
        for i, frame in enumerate(frames):
            frame_key = f'frame_{i:05d}.jpg'
            masked_frame = np.zeros_like(frame)
            masked_frame_ = masked_frame
            if frame_key in bbox.keys():
                bbox_list = bbox[frame_key]
                for (x, y, w, h) in bbox_list:
                    if w > 0 and h > 0:
                        masked_frame[y:y + h, x:x + w] = frame[y:y + h, x:x + w]
            if np.array_equal(masked_frame, masked_frame_):
                cropped_images.append(masked_frame)
            else:
                cropped_images.append(masked_frame)

        if len(cropped_images) >= 8:
            inputs_cropped = feature_extractor(cropped_images[:8], return_tensors="pt")
        else:
            lack = 8 - len(cropped_images)
            extend_frames_cropped = [cropped_images[-1].copy() for _ in range(lack)]
            cropped_images.extend(extend_frames_cropped)
            inputs_cropped = feature_extractor(cropped_images[:8], return_tensors="pt")

        return inputs["pixel_values"][0], inputs_cropped["pixel_values"][0]
        
    except Exception as e:
        print(f"get_video error in get_video_bbox /path - {path}, error: {str(e)}")
        return (torch.zeros((8, 3, 224, 224)), torch.zeros((8, 3, 224, 224)))

def make_batchs(sessions):
    label_list = ['ang', 'exc', 'fru', 'hap', 'neu', 'sad']
    batch_input, batch_audio, batch_video, batch_labels = [], [], [], []
    max_length = 400000
    for session in sessions:
        inputString = ""
        now_speaker = None
        for turn, line in enumerate(session):
            speaker, utt, wav_path, video_path, start_time, end_time, emotion = line
            # text
            inputString += '<s' + str(speaker+1) + '> ' 
            inputString += utt + " "
            now_speaker = speaker
        
        prompt = "Now"+' <s' + str(now_speaker+1) + '> '+"feels"
        concat_string = inputString.strip()
        concat_string += " " + "</s>" + " " + prompt
        
        if wav_path != None and emotion in label_list:
            # audio
            audio_input = get_audio(audio_processor, wav_path) 
            audio_input = audio_input[-max_length:]  
            batch_audio.append(audio_input) 

            # video
            video_input = get_video(video_processor, video_path, wav_path, start_time, end_time)
            batch_video.append(video_input)
            batch_input.append(encode_right_truncated(concat_string, roberta_tokenizer))

            label_ind = label_list.index(emotion)
            batch_labels.append(label_ind) 
    
    if not batch_input or not batch_audio or not batch_video or not batch_labels:
        return None
    batch_input_tokens, batch_attention_masks = padding(batch_input, roberta_tokenizer)
    batch_audio = padding_video(batch_audio)
    batch_video = padding_video(batch_video) 
    batch_labels = torch.tensor(batch_labels)    

    return batch_input_tokens, batch_attention_masks, batch_audio, batch_video, batch_labels

def video_batchs_pickle(sessions):
    label_list = ['ang', 'exc', 'fru', 'hap', 'neu', 'sad']
    batch_input, batch_video, batch_video_2, batch_labels = [], [], [], []
    max_length = 400000
    for session in sessions:
        inputString = ""
        now_speaker = None
        for turn, line in enumerate(session):
            speaker, utt, wav_path, video_path, start_time, end_time, emotion, bbox_list = line
            # text
            inputString += '<s' + str(speaker+1) + '> ' 
            inputString += utt + " "
            now_speaker = speaker
        
        prompt = "Now"+' <s' + str(now_speaker+1) + '> '+"feels"
        concat_string = inputString.strip()
        concat_string += " " + "</s>" + " " + prompt
        
        if wav_path != None and emotion in label_list:
            # video
            video_input, video_input_2 = get_video_bbox(video_processor, video_path, wav_path, start_time, end_time, bbox_list)
            batch_video.append(video_input)
            batch_video_2.append(video_input_2)

            batch_input.append(encode_right_truncated(concat_string, roberta_tokenizer))

            label_ind = label_list.index(emotion)
            batch_labels.append(label_ind) 
    
    if not batch_input or not batch_video or not batch_labels:
        return None
    batch_input_tokens, batch_attention_masks = padding(batch_input, roberta_tokenizer)
    batch_video = padding_video(batch_video) 
    if batch_video_2:
        batch_video_2 = padding_video(batch_video_2)
    batch_labels = torch.tensor(batch_labels)    

    return batch_input_tokens, batch_attention_masks, batch_video, batch_video_2, batch_labels


def make_batchs_pickle(sessions):
    label_list = ['ang', 'exc', 'fru', 'hap', 'neu', 'sad']
    batch_input, batch_audio, batch_video, batch_video_2, batch_labels = [], [], [], [], []
    max_length = 400000
    for session in sessions:
        inputString = ""
        now_speaker = None
        for turn, line in enumerate(session):
            speaker, utt, wav_path, video_path, start_time, end_time, emotion, bbox_list = line
            # text
            inputString += '<s' + str(speaker+1) + '> ' 
            inputString += utt + " "
            now_speaker = speaker
        
        prompt = "Now"+' <s' + str(now_speaker+1) + '> '+"feels"
        concat_string = inputString.strip()
        concat_string += " " + "</s>" + " " + prompt
        
        if wav_path != None and emotion in label_list:
            # audio
            audio_input = get_audio(audio_processor, wav_path) 
            audio_input = audio_input[-max_length:]  
            batch_audio.append(audio_input) 

            # video
            video_input, video_input_2 = get_video_bbox(video_processor, video_path, wav_path, start_time, end_time, bbox_list)
            batch_video.append(video_input)
            batch_video_2.append(video_input_2)

            batch_input.append(encode_right_truncated(concat_string, roberta_tokenizer))

            label_ind = label_list.index(emotion)
            batch_labels.append(label_ind) 
    
    if not batch_input or not batch_audio or not batch_video or not batch_labels:
        return None
    batch_input_tokens, batch_attention_masks = padding(batch_input, roberta_tokenizer)
    batch_audio = padding_video(batch_audio)
    batch_video = padding_video(batch_video) 
    if batch_video_2:
        batch_video_2 = padding_video(batch_video_2)
    batch_labels = torch.tensor(batch_labels)    

    return batch_input_tokens, batch_attention_masks, batch_audio, batch_video, batch_video_2, batch_labels

def teacher_batchs(sessions):
    label_list = ['ang', 'exc', 'fru', 'hap', 'neu', 'sad']

    batch_input, batch_audio, batch_video, batch_labels = [], [], [], []
    for session in sessions:

        inputString = ""
        now_speaker = None
        for turn, line in enumerate(session):
            speaker, utt, wav_path, video_path, start_time, end_time, emotion = line
            
            # text
            inputString += '<s' + str(speaker+1) + '> ' 
            inputString += utt + " "
            now_speaker = speaker

        prompt = "Now"+' <s' + str(now_speaker+1) + '> '+"feels"
        concat_string = inputString.strip()
        concat_string += " " + "</s>" + " " + prompt
        if wav_path != None and emotion in label_list:
            batch_input.append(encode_right_truncated(concat_string, roberta_tokenizer))

            if len(label_list) > 3:
                label_ind = label_list.index(emotion)
            else:
                label_ind = label_list.index(sentiment)
            batch_labels.append(label_ind) 

    if not batch_input or not batch_labels:
        return None
    batch_input_tokens, batch_attention_masks = padding(batch_input, roberta_tokenizer) 
    batch_labels = torch.tensor(batch_labels)   

    return batch_input_tokens, batch_attention_masks, batch_labels 
    
  
def audio_batchs(sessions):
    label_list = ['ang', 'exc', 'fru', 'hap', 'neu', 'sad']

    batch_input, batch_audio, batch_video, batch_labels = [], [], [], []
    max_length = 400000
    for session in sessions:

        inputString = ""
        now_speaker = None
        for turn, line in enumerate(session):
            speaker, utt, wav_path, video_path, start_time, end_time, emotion = line
            
            # text
            inputString += '<s' + str(speaker+1) + '> ' 
            inputString += utt + " "
            now_speaker = speaker
            files=video_path.split("/")[-1][:-4]

        if emotion not in label_list:
            continue

        # audio
        audio_input = get_audio(audio_processor, wav_path)
        audio_input = audio_input[-max_length:]   
        batch_audio.append(audio_input) 

        prompt = "Now"+' <s' + str(now_speaker+1) + '> '+"feels"
        concat_string = inputString.strip()
        concat_string += " " + "</s>" + " " + prompt
        batch_input.append(encode_right_truncated(concat_string, roberta_tokenizer))

        if len(label_list) > 3:
            label_ind = label_list.index(emotion)
        else:
            label_ind = label_list.index(sentiment)
        batch_labels.append(label_ind) 
        
    if not batch_input or not batch_audio or not batch_labels:
        return None

    batch_input_tokens, batch_attention_masks = padding(batch_input, roberta_tokenizer) 
    batch_audio = padding_video(batch_audio)
    batch_labels = torch.tensor(batch_labels)    
    
    return batch_input_tokens, batch_attention_masks, batch_audio, batch_labels