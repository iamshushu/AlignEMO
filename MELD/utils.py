"""MELD data collation utilities: tokenization, audio/video feature extraction, and batch builders."""
import torch
from transformers import AutoProcessor, AutoImageProcessor
from transformers import AutoTokenizer, AutoFeatureExtractor
import librosa
import cv2
import numpy as np
import os

# Encoder selection (override via env var to keep call sites untouched):
#   MELD_TEXT_MODEL,  MELD_AUDIO_PROCESSOR
_TEXT_MODEL = os.environ.get("MELD_TEXT_MODEL", "roberta-large")
_AUDIO_PROCESSOR_MODEL = os.environ.get("MELD_AUDIO_PROCESSOR", "facebook/data2vec-audio-base-960h")

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
# Whisper processor kept for student_audio_whisper.py variant
whisper_processor = AutoProcessor.from_pretrained("openai/whisper-large-v3")

def encode_right_truncated(text, tokenizer, max_length=511):
    tokenized = tokenizer.tokenize(text)
    truncated = tokenized[-max_length:]    
    ids = tokenizer.convert_tokens_to_ids(truncated)
    
    return ids + [tokenizer.mask_token_id]

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
    wav_path = path.replace('MELD.Raw', 'MELD.Raw2')
    wav_path = wav_path.replace('.mp4', '.wav')
    try:
        if os.path.exists(wav_path):
            audio, rate = librosa.load(wav_path, sr=16000)
            inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
        else:
            print(f"File not found: {wav_path}. Skipping this sample.")
            return None
            
    except Exception as e:
        print(f"Error processing audio file {wav_path}: {e}")
        return None

    return inputs["input_values"][0]

def get_audio_whisper(processor, path):
    wav_path = path.replace('MELD.Raw', 'MELD.Raw2')
    wav_path = wav_path.replace('.mp4', '.wav')

    try:
        if os.path.exists(wav_path):
            audio, rate = librosa.load(wav_path, sr=16000)
            inputs = processor(
                audio=audio,
                sampling_rate=16000,
                return_tensors="pt"
            )
            return inputs["input_features"][0]
        else:
            print(f"File not found: {wav_path}. Returning None.")
            return None
    except Exception as e:
        print(f"Error processing whisper audio file {wav_path}: {e}")
        return None

def make_batchs_whisper(sessions):
    label_list = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
    batch_input, batch_audio, batch_video, batch_labels = [], [], [], []

    for session in sessions:
        inputString = ""
        now_speaker = None

        for turn, line in enumerate(session):
            speaker, utt, video_path, emotion = line

            inputString += '<s' + str(speaker+1) + '> '
            inputString += utt + " "
            now_speaker = speaker

        audio_input = get_audio_whisper(whisper_processor, video_path)
        if audio_input is not None:
            batch_audio.append(audio_input)

            video_input = get_video(video_processor, video_path)
            batch_video.append(video_input)

            prompt = "Now" + ' <s' + str(now_speaker+1) + '> ' + "feels"
            concat_string = inputString.strip()
            concat_string += " " + "</s>" + " " + prompt
            batch_input.append(encode_right_truncated(concat_string, roberta_tokenizer))

            label_ind = label_list.index(emotion)
            batch_labels.append(label_ind)

    batch_input_tokens, batch_attention_masks = padding(batch_input, roberta_tokenizer)
    batch_audio = torch.stack(batch_audio)
    batch_video = torch.stack(batch_video)
    batch_labels = torch.tensor(batch_labels)

    return batch_input_tokens, batch_attention_masks, batch_audio, batch_video, batch_labels

def make_batchs_pickle_whisper(sessions):
    label_list = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
    batch_input, batch_audio, batch_video, batch_video_2, batch_labels = [], [], [], [], []

    for session in sessions:
        inputString = ""
        now_speaker = None

        for turn, line in enumerate(session):
            speaker, utt, video_path, emotion, bbox_list = line

            inputString += '<s' + str(speaker+1) + '> '
            inputString += utt + " "
            now_speaker = speaker

        audio_input = get_audio_whisper(whisper_processor, video_path)
        if audio_input is not None:
            batch_audio.append(audio_input)

            video_input, video_input_2 = get_video_bbox(video_processor, video_path, bbox_list)
            batch_video.append(video_input)
            batch_video_2.append(video_input_2)

            prompt = "Now" + ' <s' + str(now_speaker+1) + '> ' + "feels"
            concat_string = inputString.strip()
            concat_string += " " + "</s>" + " " + prompt
            batch_input.append(encode_right_truncated(concat_string, roberta_tokenizer))

            label_ind = label_list.index(emotion)
            batch_labels.append(label_ind)

    batch_input_tokens, batch_attention_masks = padding(batch_input, roberta_tokenizer)
    batch_audio = torch.stack(batch_audio)
    batch_video = torch.stack(batch_video)
    batch_video_2 = torch.stack(batch_video_2)
    batch_labels = torch.tensor(batch_labels)

    return batch_input_tokens, batch_attention_masks, batch_audio, batch_video, batch_video_2, batch_labels


def get_video(feature_extractor, path):
    video_folder = path.split('.mp4')[0]

    jpg_path = video_folder
    frames = []
    
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
        return None

  
def get_video_bbox(feature_extractor, path, bbox):
    video_folder_ = path.split('.mp4')[0]
    jpg_path = video_folder_
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
        # for frame in frames:
        #     frame_key = 'frame_{:05d}.jpg'.format(frame_num)
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
        return None

  
def get_reverse_video_bbox(feature_extractor, path, bbox):
    video_folder_ = path.split('.mp4')[0]
    jpg_path = video_folder_
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
        # for frame in frames:
        #     frame_key = 'frame_{:05d}.jpg'.format(frame_num)
        for i, frame in enumerate(frames):
            frame_key = f'frame_{i:05d}.jpg'
            masked_frame = frame.copy()

            if frame_key in bbox:
                for (x, y, w, h) in bbox[frame_key]:
                    if w > 0 and h > 0:
                        # 좌표 보정(프레임 범위 내로 클램프)
                        H, W = frame.shape[:2]
                        x1, y1 = int(max(0, x)), int(max(0, y))
                        x2, y2 = int(min(W, x + w)), int(min(H, y + h))
                        if x2 > x1 and y2 > y1:
                            # BGR 3채널일 때 검정 채우기
                            cv2.rectangle(masked_frame, (x1, y1), (x2, y2), (0, 0, 0), thickness=-1)
            cropped_images.append(masked_frame)
        if len(cropped_images) == 0:
            cropped_images = [f.copy() for f in frames] 
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
        return None


def reverse_video_batchs_pickle(sessions):
    label_list = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
    batch_input, batch_video, batch_video_2, batch_labels = [], [], [], []
    max_length = 400000
    for session in sessions:
        inputString = ""
        now_speaker = None
        for turn, line in enumerate(session):
            speaker, utt, video_path , emotion, bbox_list = line
            
            # text
            inputString += '<s' + str(speaker+1) + '> ' 
            inputString += utt + " "
            now_speaker = speaker

        # video
        try:
            video_input, video_input_2 = get_reverse_video_bbox(video_processor, video_path, bbox_list)
        except:
            print(f"Error processing video: {video_path}")
            continue
        batch_video.append(video_input)
        batch_video_2.append(video_input_2)

        prompt = "Now"+' <s' + str(now_speaker+1) + '> '+"feels"
        concat_string = inputString.strip()
        concat_string += " " + "</s>" + " " + prompt
        batch_input.append(encode_right_truncated(concat_string, roberta_tokenizer))


        label_ind = label_list.index(emotion)
        batch_labels.append(label_ind) 

    batch_input_tokens, batch_attention_masks = padding(batch_input, roberta_tokenizer)
    batch_video = torch.stack(batch_video) 
    batch_video_2 = torch.stack(batch_video_2)
    batch_labels = torch.tensor(batch_labels)    
    
    return batch_input_tokens, batch_attention_masks, batch_video, batch_video_2, batch_labels

def video_batchs_pickle(sessions):
    label_list = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
    batch_input, batch_video, batch_video_2, batch_labels = [], [], [], []
    max_length = 400000
    for session in sessions:
        inputString = ""
        now_speaker = None
        for turn, line in enumerate(session):
            speaker, utt, video_path , emotion, bbox_list = line
            
            # text
            inputString += '<s' + str(speaker+1) + '> ' 
            inputString += utt + " "
            now_speaker = speaker

        # video
        try:
            video_pair = get_video_bbox(video_processor, video_path, bbox_list)
            if video_pair is None:
                continue
            video_input, video_input_2 = video_pair
        except Exception:
            print(f"Error processing video: {video_path}")
            continue
        batch_video.append(video_input)
        batch_video_2.append(video_input_2)

        prompt = "Now"+' <s' + str(now_speaker+1) + '> '+"feels"
        concat_string = inputString.strip()
        concat_string += " " + "</s>" + " " + prompt
        batch_input.append(encode_right_truncated(concat_string, roberta_tokenizer))


        label_ind = label_list.index(emotion)
        batch_labels.append(label_ind) 

    batch_input_tokens, batch_attention_masks = padding(batch_input, roberta_tokenizer)
    batch_video = torch.stack(batch_video) 
    batch_video_2 = torch.stack(batch_video_2)
    batch_labels = torch.tensor(batch_labels)    
    
    return batch_input_tokens, batch_attention_masks, batch_video, batch_video_2, batch_labels


def make_batchs_pickle(sessions):
    label_list = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
    batch_input, batch_audio, batch_video, batch_labels = [], [], [], []
    batch_video_2 = []
    max_length = 400000
    for session in sessions:
        inputString = ""
        now_speaker = None
        for turn, line in enumerate(session):
            speaker, utt, video_path , emotion, bbox_list = line
            # text
            inputString += '<s' + str(speaker+1) + '> '
            inputString += utt + " "
            now_speaker = speaker

        
        audio_input = get_audio(audio_processor, video_path)
        if audio_input != None:
            audio_input = audio_input[-max_length:] 

            # video
            video_pair = get_video_bbox(video_processor, video_path, bbox_list)
            if video_pair is None:
                continue
            video_input, video_input_2 = video_pair
            batch_audio.append(audio_input)
            batch_video_2.append(video_input_2)        
            batch_video.append(video_input)

            prompt = "Now"+' <s' + str(now_speaker+1) + '> '+"feels"
            concat_string = inputString.strip()
            concat_string += " " + "</s>" + " " + prompt
            batch_input.append(encode_right_truncated(concat_string, roberta_tokenizer))
            
            label_ind = label_list.index(emotion)
            batch_labels.append(label_ind) 

    if not batch_input:
        return None
    batch_input_tokens, batch_attention_masks = padding(batch_input, roberta_tokenizer)
    batch_audio = padding_video(batch_audio)
    batch_video = torch.stack(batch_video) 
    batch_video_2 = torch.stack(batch_video_2)
    batch_labels = torch.tensor(batch_labels)    
    
    return batch_input_tokens, batch_attention_masks, batch_audio, batch_video, batch_video_2, batch_labels


def make_batchs(sessions):
    label_list = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
    batch_input, batch_audio, batch_video, batch_labels = [], [], [], []
    max_length = 400000
    for session in sessions:

        inputString = ""
        now_speaker = None
        for turn, line in enumerate(session):
            speaker, utt, video_path , emotion = line
            
            # text
            inputString += '<s' + str(speaker+1) + '> ' 
            inputString += utt + " "
            now_speaker = speaker
            files=video_path.split("/")[-1][:-4]

        # audio
        audio_input = get_audio(audio_processor, video_path)
        if audio_input != None:
            audio_input = audio_input[-max_length:] 

            # video
            video_input = get_video(video_processor, video_path)
            if video_input is None:
                continue
            batch_audio.append(audio_input)
            batch_video.append(video_input)

            prompt = "Now"+' <s' + str(now_speaker+1) + '> '+"feels"
            concat_string = inputString.strip()
            concat_string += " " + "</s>" + " " + prompt
            batch_input.append(encode_right_truncated(concat_string, roberta_tokenizer))

            if len(label_list) > 3:
                label_ind = label_list.index(emotion)
            else:
                label_ind = label_list.index(sentiment)
            batch_labels.append(label_ind) 

    batch_input_tokens, batch_attention_masks = padding(batch_input, roberta_tokenizer)
    batch_audio = padding_video(batch_audio)
    batch_video = torch.stack(batch_video) 
    batch_labels = torch.tensor(batch_labels)    
    
    return batch_input_tokens, batch_attention_masks, batch_audio, batch_video, batch_labels

def make_batchs_pickle_whisper_meld(sessions):
    label_list = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
    batch_input, batch_audio, batch_video, batch_video_2, batch_labels = [], [], [], [], []

    for session in sessions:
        if session is None:
            continue

        inputString = ""
        now_speaker = None
        video_path = None
        emotion = None
        bbox_list = None

        for turn, line in enumerate(session):
            # expected: speaker, utt, video_path, emotion, bbox_list
            speaker, utt, video_path, emotion, bbox_list = line

            inputString += '<s' + str(speaker + 1) + '> '
            inputString += utt + " "
            now_speaker = speaker

        audio_input = get_audio_whisper(whisper_processor, video_path)
        if audio_input is None:
            continue

        video_input, video_input_2 = get_video_bbox(video_processor, video_path, bbox_list)

        prompt = "Now" + ' <s' + str(now_speaker + 1) + '> ' + "feels"
        concat_string = inputString.strip() + " </s> " + prompt

        batch_input.append(encode_right_truncated(concat_string, roberta_tokenizer))
        batch_audio.append(audio_input)
        batch_video.append(video_input)
        batch_video_2.append(video_input_2)
        batch_labels.append(label_list.index(emotion))

    batch_input_tokens, batch_attention_masks = padding(batch_input, roberta_tokenizer)
    batch_audio = torch.stack(batch_audio, dim=0)
    batch_video = torch.stack(batch_video, dim=0)
    batch_video_2 = torch.stack(batch_video_2, dim=0)
    batch_labels = torch.tensor(batch_labels)

    return batch_input_tokens, batch_attention_masks, batch_audio, batch_video, batch_video_2, batch_labels
