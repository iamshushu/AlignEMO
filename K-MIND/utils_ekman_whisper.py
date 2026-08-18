import torch
from transformers import RobertaTokenizer, RobertaModel, AutoProcessor, AutoImageProcessor
from transformers import AutoModel, AutoTokenizer, pipeline
import librosa
import cv2
import numpy as np
import av
import os

import hashlib
from pathlib import Path
from transformers import WhisperProcessor

# =========================
# Whisper cache config
# =========================
WHISPER_MODEL_NAME = "openai/whisper-large-v3"
WHISPER_CACHE_ROOT = "./dataset/KMIND/whisper_cache_large_v3"

# The K-MIND CSVs store absolute wav paths from the machine that produced them.
# If yours differ, set both env vars to rewrite the prefix on the fly, e.g.
#   export KMIND_WAV_OLD_PREFIX=/original/path/to/KMIND/
#   export KMIND_WAV_NEW_PREFIX=/your/path/to/KMIND/
# Leaving them unset is a no-op (paths are used exactly as the CSV stores them).
WAV_PATH_OLD_PREFIX = os.environ.get("KMIND_WAV_OLD_PREFIX", "")
WAV_PATH_NEW_PREFIX = os.environ.get("KMIND_WAV_NEW_PREFIX", "")


def remap_wav_path(path):
    """Rewrite a CSV-stored wav path onto this machine's layout (no-op if unset)."""
    if WAV_PATH_OLD_PREFIX and path:
        return path.replace(WAV_PATH_OLD_PREFIX, WAV_PATH_NEW_PREFIX)
    return path

# 캐시 파일 저장 시 half로 저장하면 디스크 절약됨
WHISPER_SAVE_HALF = True

# 캐시가 없을 때 학습 중 즉석 생성할지 여부
# 속도 때문에 보통 False 추천
WHISPER_ONLINE_FALLBACK = False

_whisper_processor = None


def get_whisper_processor():
    global _whisper_processor
    if _whisper_processor is None:
        _whisper_processor = WhisperProcessor.from_pretrained(WHISPER_MODEL_NAME)
    return _whisper_processor


def resolve_wav_path(wav_path, old_prefix=WAV_PATH_OLD_PREFIX, new_prefix=WAV_PATH_NEW_PREFIX):
    if wav_path is None:
        return None

    path = str(wav_path)

    if old_prefix is not None and new_prefix is not None and path.startswith(old_prefix):
        path = path.replace(old_prefix, new_prefix, 1)

    return path


def wav_to_feature_path(wav_path, cache_root=WHISPER_CACHE_ROOT):
    """
    wav_path -> deterministic cache file path
    """
    resolved = resolve_wav_path(wav_path)
    if resolved is None:
        return None

    resolved_abs = os.path.abspath(resolved)
    digest = hashlib.md5(resolved_abs.encode("utf-8")).hexdigest()
    stem = Path(resolved_abs).stem

    # 파일 너무 많아지는 것 방지용 서브디렉토리
    subdir = digest[:2]
    feat_name = f"{stem}_{digest[:16]}.pt"
    return os.path.join(cache_root, subdir, feat_name)


def build_whisper_feature_from_wav(wav_path, processor=None):
    """
    wav -> Whisper input_features
    return: Tensor [mel_bins, time]
    """
    resolved = resolve_wav_path(wav_path)
    if resolved is None or not os.path.isfile(resolved):
        return None

    if processor is None:
        processor = get_whisper_processor()

    try:
        audio, _ = librosa.load(resolved, sr=16000, mono=True)
    except Exception as e:
        print(f"[WARN] audio load failed: {resolved} ({e})")
        return None

    if audio is None or len(audio) == 0:
        print(f"[WARN] empty audio: {resolved}")
        return None

    # Whisper는 30초 window 기준. 짧은 발화여도 여기서 고정 길이로 맞춰두면 안전함.
    max_samples = 30 * 16000
    if len(audio) > max_samples:
        audio = audio[:max_samples]
    else:
        pad_len = max_samples - len(audio)
        audio = np.pad(audio, (0, pad_len))

    try:
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
        feat = inputs["input_features"][0].contiguous()   # [mel_bins, time]
        return feat
    except Exception as e:
        print(f"[WARN] whisper processor failed: {resolved} ({e})")
        return None


def save_whisper_feature(wav_path, cache_root=WHISPER_CACHE_ROOT, overwrite=False):
    """
    wav_path에 대응하는 Whisper feature를 .pt로 저장
    return: ("saved"|"skipped"|"failed", feat_path)
    """
    feat_path = wav_to_feature_path(wav_path, cache_root=cache_root)
    if feat_path is None:
        return "failed", None

    if (not overwrite) and os.path.isfile(feat_path):
        return "skipped", feat_path

    feat = build_whisper_feature_from_wav(wav_path)
    if feat is None:
        return "failed", feat_path

    os.makedirs(os.path.dirname(feat_path), exist_ok=True)

    if WHISPER_SAVE_HALF:
        feat = feat.half()
    else:
        feat = feat.float()

    torch.save(feat, feat_path)
    return "saved", feat_path


def load_cached_whisper_feature(wav_path, cache_root=WHISPER_CACHE_ROOT):
    feat_path = wav_to_feature_path(wav_path, cache_root=cache_root)
    if feat_path is None or not os.path.isfile(feat_path):
        return None

    try:
        feat = torch.load(feat_path, map_location="cpu")
        if isinstance(feat, dict):
            # 혹시 dict로 저장했을 경우 대비
            if "input_features" in feat:
                feat = feat["input_features"]
            else:
                return None
        feat = feat.float()
        return feat
    except Exception as e:
        print(f"[WARN] cached feature load failed: {feat_path} ({e})")
        return None


def audio_batchs_whisper(sessions):
    """
    preprocessing.py 출력 형식:
    [speaker, utt, wav_path, video_path, start_time, end_time, emotion]
    """
    batch_input_ids, batch_attn = [], []
    batch_audio, batch_labels = [], []

    for session in sessions:
        inputString, now_speaker = "", None
        last_emotion, last_wav = None, None

        for line in session:
            speaker, utt, wav_path, video_path, start_time, end_time, emotion = line
            inputString += f"<s{speaker+1}> {utt} "
            now_speaker = speaker
            last_emotion = emotion
            last_wav = wav_path

        if last_wav is None:
            continue

        # 1) 캐시 우선 로드
        audio_input = load_cached_whisper_feature(last_wav)

        # 2) 없으면 선택적으로 즉석 생성
        if audio_input is None and WHISPER_ONLINE_FALLBACK:
            audio_input = build_whisper_feature_from_wav(last_wav)

        if audio_input is None:
            # 캐시 안 만들었거나 경로 문제면 여기서 skip
            continue

        if audio_input.ndim != 2:
            print(f"[WARN] invalid whisper feature shape: {tuple(audio_input.shape)}")
            continue

        batch_audio.append(audio_input)

        prompt = f"지금 <s{now_speaker+1}> 의 감정은"
        concat_string = (inputString.strip() + " </s> " + prompt).strip()

        enc = roberta_tokenizer(
            concat_string,
            truncation=True,
            max_length=512,
            padding="max_length",
            return_tensors="pt",
        )
        batch_input_ids.append(enc["input_ids"].squeeze(0))
        batch_attn.append(enc["attention_mask"].squeeze(0))

        emo = (last_emotion or "").strip()
        label = label2id.get(emo, -1)
        batch_labels.append(label)

    if not (batch_input_ids and batch_audio and batch_labels):
        return None

    batch_input_tokens = torch.stack(batch_input_ids, dim=0)
    batch_attention_masks = torch.stack(batch_attn, dim=0)
    batch_audio = torch.stack(batch_audio, dim=0)   # [B, mel_bins, time]
    batch_labels = torch.tensor(batch_labels, dtype=torch.long)

    return batch_input_tokens, batch_attention_masks, batch_audio, batch_labels

audio_processor = AutoProcessor.from_pretrained("facebook/data2vec-audio-base-960h")
video_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")
# roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
roberta_tokenizer = AutoTokenizer.from_pretrained("klue/roberta-base")
speaker_list = ['<s1>', '<s2>', '<s3>', '<s4>', '<s5>', '<s6>', '<s7>', '<s8>', '<s9>']
speaker_tokens_dict = {'additional_special_tokens': speaker_list}
roberta_tokenizer.add_special_tokens(speaker_tokens_dict)

LABELS = ["기쁨","슬픔","화남","놀람","무서움","혐오","중립"]
label2id = {lab: i for i, lab in enumerate(LABELS)}

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


def padding(ids_list, tokenizer, max_length=512):
    clipped = []
    for ids in ids_list:
        if len(ids) > max_length:
            ids = ids[-max_length:]        
        clipped.append(ids)

    max_len = max(len(ids) for ids in clipped)
    pad_ids, attention_masks = [], []
    for ids in clipped:
        pad_len = max_len - len(ids)
        add_ids = [tokenizer.pad_token_id] * pad_len
        pad_ids.append(add_ids + ids)
        attention_mask = [0]*pad_len + [1]*len(ids)
        attention_masks.append(attention_mask)

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
    if not path or not os.path.isfile(path):
        return None
    try:
        audio, rate = librosa.load(path, sr=16000, mono=True)
    except Exception as e:
        print(f"[WARN] audio load failed: {path} ({e})")
        return None

    try:
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
        return inputs["input_values"][0]
    except Exception as e:
        print(f"[WARN] audio processor failed: {path} ({e})")
        return None


def get_video(feature_extractor, path, wav_path, start_time, end_time):
    jpg_path = path
    if not jpg_path or not os.path.isdir(jpg_path):   # ← isdir
        return torch.zeros((8, 3, 224, 224))
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
        # print(f"get_video error in get_video_bbox /path - {path}, error: {str(e)}")
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



def get_video_bbox_reverse(feature_extractor, path, wav_path, start_time, end_time, bbox):
    jpg_path = path
    if not jpg_path or not os.path.isdir(jpg_path):   # ← isdir
        return (torch.zeros((8, 3, 224, 224)), torch.zeros((8, 3, 224, 224)))
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
        return (torch.zeros((8, 3, 224, 224)), torch.zeros((8, 3, 224, 224)))

def get_video_bbox(feature_extractor, path, wav_path, start_time, end_time, bbox):
    jpg_path = path
    if not jpg_path or not os.path.isdir(jpg_path):   # ← isdir
        return (torch.zeros((8, 3, 224, 224)), torch.zeros((8, 3, 224, 224)))
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
            # frame_key = 'frame_{:05d}.jpg'.format(frame_num)
        for i, frame in enumerate(frames):
            frame_key = f'frame_{i:05d}.jpg'
            # print(frame_key)
            masked_frame = np.zeros_like(frame)
            masked_frame_ = masked_frame
            if frame_key in bbox.keys():
                bbox_list = bbox[frame_key]
                for (x, y, w, h) in bbox_list:
                    if w > 0 and h > 0:
                        masked_frame[y:y + h, x:x + w] = frame[y:y + h, x:x + w]
            if np.array_equal(masked_frame, masked_frame_):
                cropped_images.append(frame)
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

def get_video_bbox_bc(feature_extractor, path, wav_path, start_time, end_time, bbox):
    jpg_path = path
    # 폴더가 없으면 0 텐서 리턴
    if not jpg_path or not os.path.isdir(jpg_path):
        return (torch.zeros((8, 3, 224, 224)), torch.zeros((8, 3, 224, 224)))

    frames = []
    masked_images = []          # bbox 부분만 남긴 이미지
    reverse_masked_images = []  # bbox 부분만 지운 이미지

    try:
        files = os.listdir(jpg_path)
        frame_paths = sorted([os.path.join(jpg_path, f) for f in files])

        # 모든 프레임 읽기
        for frame_path in frame_paths:
            frame = cv2.imread(frame_path)
            if frame is None:
                continue
            frames.append(frame)

        # 프레임이 하나도 없으면 0 텐서 리턴
        if len(frames) == 0:
            return (torch.zeros((8, 3, 224, 224)), torch.zeros((8, 3, 224, 224)))

        # 길이 8로 맞추기 (pad with last frame)
        if len(frames) >= 8:
            frames = frames[:8]
        else:
            lack = 8 - len(frames)
            extend_frames = [frames[-1].copy() for _ in range(lack)]
            frames.extend(extend_frames)
            frames = frames[:8]

        # 각 프레임마다 masked / reverse-masked 만들기
        for i, frame in enumerate(frames):
            frame_key = f'frame_{i:05d}.jpg'

            # bbox 영역만 남긴 이미지 (나머지는 0)
            masked_frame = np.zeros_like(frame)

            # bbox 영역만 까맣게 지운 이미지 (나머지는 원본)
            reverse_masked_frame = frame.copy()

            if frame_key in bbox.keys():
                bbox_list = bbox[frame_key]
                for (x, y, w, h) in bbox_list:
                    if w > 0 and h > 0:
                        # bbox 부분만 살리기
                        masked_frame[y:y + h, x:x + w] = frame[y:y + h, x:x + w]
                        # reverse 쪽은 bbox 부분만 지우기
                        reverse_masked_frame[y:y + h, x:x + w] = 0

            # bbox가 없으면 masked_frame은 올블랙, reverse_masked는 원본 그대로
            masked_images.append(masked_frame)
            reverse_masked_images.append(reverse_masked_frame)

        # 길이 8로 맞추기 (혹시 모를 경우 대비)
        if len(masked_images) < 8:
            lack = 8 - len(masked_images)
            extend_frames_masked = [masked_images[-1].copy() for _ in range(lack)]
            masked_images.extend(extend_frames_masked)

        if len(reverse_masked_images) < 8:
            lack = 8 - len(reverse_masked_images)
            extend_frames_reverse = [reverse_masked_images[-1].copy() for _ in range(lack)]
            reverse_masked_images.extend(extend_frames_reverse)

        # feature_extractor 적용
        inputs_masked = feature_extractor(masked_images[:8], return_tensors="pt")
        inputs_reverse = feature_extractor(reverse_masked_images[:8], return_tensors="pt")

        # (8, 3, 224, 224) 형태로 리턴
        # reverse / masked
        return inputs_reverse["pixel_values"][0], inputs_masked["pixel_values"][0]

    except Exception as e:
        print(f"get_video error in get_video_bbox_bc /path - {path}, error: {str(e)}")
        return (torch.zeros((8, 3, 224, 224)), torch.zeros((8, 3, 224, 224)))

def get_video_bbox_abc(feature_extractor, path, wav_path, start_time, end_time, bbox):
    jpg_path = path
    # 폴더가 없으면 0 텐서 리턴
    if not jpg_path or not os.path.isdir(jpg_path):
        return (torch.zeros((8, 3, 224, 224)), torch.zeros((8, 3, 224, 224)), torch.zeros((8, 3, 224, 224)))

    frames = []
    masked_images = []          # bbox 부분만 남긴 이미지
    reverse_masked_images = []  # bbox 부분만 지운 이미지

    try:
        files = os.listdir(jpg_path)
        frame_paths = sorted([os.path.join(jpg_path, f) for f in files])

        # 모든 프레임 읽기
        for frame_path in frame_paths:
            frame = cv2.imread(frame_path)
            if frame is None:
                continue
            frames.append(frame)

        # 프레임이 하나도 없으면 0 텐서 리턴
        if len(frames) == 0:
            return (torch.zeros((8, 3, 224, 224)), torch.zeros((8, 3, 224, 224)),  torch.zeros((8, 3, 224, 224)))

        # 길이 8로 맞추기 (pad with last frame)
        if len(frames) >= 8:
            frames = frames[:8]
            inputs = feature_extractor(frames[:8], return_tensors="pt")        
        else:
            lack = 8 - len(frames)
            extend_frames = [frames[-1].copy() for _ in range(lack)]
            frames.extend(extend_frames)
            frames = frames[:8]
            inputs = feature_extractor(frames[:8], return_tensors="pt") 

        # 각 프레임마다 masked / reverse-masked 만들기
        for i, frame in enumerate(frames):
            frame_key = f'frame_{i:05d}.jpg'

            # bbox 영역만 남긴 이미지 (나머지는 0)
            masked_frame = np.zeros_like(frame)

            # bbox 영역만 까맣게 지운 이미지 (나머지는 원본)
            reverse_masked_frame = frame.copy()

            if frame_key in bbox.keys():
                bbox_list = bbox[frame_key]
                for (x, y, w, h) in bbox_list:
                    if w > 0 and h > 0:
                        # bbox 부분만 살리기
                        masked_frame[y:y + h, x:x + w] = frame[y:y + h, x:x + w]
                        # reverse 쪽은 bbox 부분만 지우기
                        reverse_masked_frame[y:y + h, x:x + w] = 0

            # bbox가 없으면 masked_frame은 올블랙, reverse_masked는 원본 그대로
            masked_images.append(masked_frame)
            reverse_masked_images.append(reverse_masked_frame)

        # 길이 8로 맞추기 (혹시 모를 경우 대비)
        if len(masked_images) < 8:
            lack = 8 - len(masked_images)
            extend_frames_masked = [masked_images[-1].copy() for _ in range(lack)]
            masked_images.extend(extend_frames_masked)

        if len(reverse_masked_images) < 8:
            lack = 8 - len(reverse_masked_images)
            extend_frames_reverse = [reverse_masked_images[-1].copy() for _ in range(lack)]
            reverse_masked_images.extend(extend_frames_reverse)

        # feature_extractor 적용
        inputs_masked = feature_extractor(masked_images[:8], return_tensors="pt")
        inputs_reverse = feature_extractor(reverse_masked_images[:8], return_tensors="pt")

        # (8, 3, 224, 224) 형태로 리턴
        # reverse / masked
        return inputs_reverse["pixel_values"][0], inputs_masked["pixel_values"][0], inputs["pixel_values"][0]

    except Exception as e:
        print(f"get_video error in get_video_bbox_bc /path - {path}, error: {str(e)}")
        return (torch.zeros((8, 3, 224, 224)), torch.zeros((8, 3, 224, 224)), torch.zeros((8, 3, 224, 224)))




def make_batchs_pickle_audio_teacher_video_student(sessions):
    batch_input_ids, batch_attn = [], []
    batch_audio, batch_video, batch_video_2, batch_labels = [], [], [], []
    max_length = 400000

    for session in sessions:
        inputString, now_speaker = "", None
        last_emotion, last_wav = None, None
        for turn, line in enumerate(session):
            speaker, utt, wav_path, video_path, start_time, end_time, emotion, bbox_list = line
            inputString += f"<s{speaker+1}> {utt} "
            now_speaker = speaker
            last_emotion = emotion
            last_wav = wav_path

        last_wav = remap_wav_path(last_wav)

       
        # ---- 오디오 ----
        if last_wav is None:
            continue
        audio_input = get_audio(audio_processor, last_wav)
        if audio_input is None:
            continue
        audio_input = audio_input[-max_length:]
        batch_audio.append(audio_input)

        # ---- 비디오 ----
        video_input, video_input_2 = get_video_bbox(
            video_processor, video_path, last_wav, start_time, end_time, bbox_list
        )
        batch_video.append(video_input)
        batch_video_2.append(video_input_2)

       
        # ---- 라벨 ----
        emo = (last_emotion or "").strip()
        label = label2id.get(emo, -1)  # 모르는 값은 -1(ignore)
        batch_labels.append(label)

    batch_audio          = padding_video(batch_audio)                      # 기존 헬퍼 활용
    batch_video          = padding_video(batch_video)
    batch_video_2        = padding_video(batch_video_2) if batch_video_2 else None
    batch_labels         = torch.tensor(batch_labels, dtype=torch.long)    # long 보장

    return batch_audio, batch_video, batch_video_2, batch_labels



def make_batchs_pickle_audio_teacher_text_student(sessions):
    batch_input_ids, batch_attn = [], []
    batch_audio, batch_labels = [], []
    max_length = 400000

    for session in sessions:
        inputString, now_speaker = "", None
        last_emotion, last_wav = None, None
        for turn, line in enumerate(session):
            speaker, utt, wav_path, video_path, start_time, end_time, emotion, bbox_list = line
            inputString += f"<s{speaker+1}> {utt} "
            now_speaker = speaker
            last_emotion = emotion
            last_wav = wav_path

        last_wav = remap_wav_path(last_wav)

        prompt = f"지금 <s{now_speaker+1}> 의 감정은"
        concat_string = (inputString.strip() + " </s> " + prompt).strip()

        # ---- 오디오 ----
        if last_wav is None:
            continue
        audio_input = get_audio(audio_processor, last_wav)
        if audio_input is None:
            continue
        audio_input = audio_input[-max_length:]
        batch_audio.append(audio_input)


        # ---- 텍스트 (무조건 512로 자르고 패딩) ----
        enc = roberta_tokenizer(
            concat_string,
            truncation=True,
            max_length=512,
            padding="max_length",
            return_tensors="pt",
        )
        batch_input_ids.append(enc["input_ids"].squeeze(0))       # [512]
        batch_attn.append(enc["attention_mask"].squeeze(0))       # [512]

        # ---- 라벨 ----
        emo = (last_emotion or "").strip()
        label = label2id.get(emo, -1)  # 모르는 값은 -1(ignore)
        batch_labels.append(label)


    batch_input_tokens   = torch.stack(batch_input_ids, dim=0)            # [B,512]
    batch_attention_masks= torch.stack(batch_attn,      dim=0)            # [B,512]
    batch_audio          = padding_video(batch_audio)                      # 기존 헬퍼 활용
    batch_labels         = torch.tensor(batch_labels, dtype=torch.long)    # long 보장

    with torch.no_grad():
        max_len = (batch_input_tokens != 0).sum(dim=1).max().item()
        assert max_len <= 512, f"max_len={max_len} > 512 (truncation 누락)"
        mn, mx = int(batch_labels.min().item()), int(batch_labels.max().item())
        assert mn >= -1 and mx < len(LABELS), f"label range [{mn},{mx}] out of [0,{len(LABELS)-1}]"

    return batch_input_tokens, batch_attention_masks, batch_audio, batch_labels

def make_batchs(sessions):
    import torch
    batch_input_ids, batch_attn = [], []
    batch_audio, batch_video, batch_labels = [], [], []
    max_length = 400000

    for session in sessions:
        inputString, now_speaker = "", None
        last_wav, last_video_path, last_start, last_end, last_emotion = None, None, None, None, None

        for turn, line in enumerate(session):
            # (speaker, utt, wav_path, video_path, start_time, end_time, emotion)
            # print(line)
            speaker, utt, wav_path, video_path, start_time, end_time, emotion = line
            inputString += f"<s{speaker+1}> {utt} "
            now_speaker = speaker
            # 세션의 최종 turn 기준으로 사용 (기존 코드와 동일 정책)
            last_wav, last_video_path = wav_path, video_path
            last_start, last_end = start_time, end_time
            last_emotion = emotion

        # 오디오 없으면 스킵(정책 유지)
        audio_input = get_audio(audio_processor, last_wav)
        if audio_input is None:
            continue
        audio_input = audio_input[-max_length:]
        batch_audio.append(audio_input)

        # 비디오(기존 헬퍼 사용)
        video_input = get_video(video_processor, last_video_path, last_wav, last_start, last_end)
        batch_video.append(video_input)

        # ---- 텍스트: 무조건 512로 자르고 패딩 ----
        prompt = f"지금 <s{now_speaker+1}> 의 감정은"
        concat_string = (inputString.strip() + " </s> " + prompt).strip()
        enc = roberta_tokenizer(
            concat_string,
            truncation=True,          # ★ 필수
            max_length=512,           # ★ 필수
            padding="max_length",     # 배치 길이 동일화
            return_tensors="pt",
        )
        batch_input_ids.append(enc["input_ids"].squeeze(0))     # [512]
        batch_attn.append(enc["attention_mask"].squeeze(0))     # [512]

        # ---- 라벨: 7클래스 고정, 미매핑은 -1(ignore) ----
        emo = (last_emotion or "").strip()
        label = label2id.get(emo, -1)
        batch_labels.append(label)

    # 빈 배치면 상위 루프에서 continue하도록 None 반환(기존 정책 유지)
    if not (batch_input_ids and batch_audio and batch_labels):
        return None

    batch_input_tokens    = torch.stack(batch_input_ids, dim=0)            # [B,512]
    batch_attention_masks = torch.stack(batch_attn,      dim=0)            # [B,512]
    batch_audio           = padding_video(batch_audio)                      # 기존 헬퍼
    batch_video           = padding_video(batch_video)
    batch_labels          = torch.tensor(batch_labels, dtype=torch.long)    # long 필수

    # 안전장치(디버깅용): 길이/라벨 범위 확인
    with torch.no_grad():
        max_len = (batch_input_tokens != 0).sum(dim=1).max().item()
        assert max_len <= 512, f"max_len={max_len} > 512 (truncation 누락)"
        mn, mx = int(batch_labels.min().item()), int(batch_labels.max().item())
        assert mn >= -1 and mx < len(LABELS), f"label range [{mn},{mx}] out of [0,{len(LABELS)-1}]"

    return batch_input_tokens, batch_attention_masks, batch_audio, batch_video, batch_labels

def video_batchs_pickle(sessions):
    batch_input_ids, batch_attn, batch_video, batch_video_2, batch_labels = [], [], [], [], []

    for session in sessions:
        inputString = ""
        now_speaker = None
        last_emotion, last_wav = None, None

        for turn, line in enumerate(session):
            speaker, utt, wav_path, video_path, start_time, end_time, emotion, bbox_list = line
            inputString += f"<s{speaker+1}> {utt} "
            now_speaker = speaker
            last_emotion = emotion
            last_wav = wav_path

        prompt = f"지금 <s{now_speaker+1}> 의 감정은"
        concat_string = (inputString.strip() + " </s> " + prompt).strip()

        # 비디오 특징 추출(기존 로직 유지)
        if last_wav is not None:
            video_input, video_input_2 = get_video_bbox(
                video_processor, video_path, last_wav, start_time, end_time, bbox_list
            )
            batch_video.append(video_input)
            batch_video_2.append(video_input_2)

            # ★ 핵심: 토크나이저에서 무조건 512로 자르고 패딩
            enc = roberta_tokenizer(
                concat_string,
                truncation=True,
                max_length=512,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].squeeze(0)        # [512]
            attn_mask = enc["attention_mask"].squeeze(0)   # [512]

            # 라벨 매핑: 모르는 값은 -1 (ignore)
            emo = (last_emotion or "").strip()
            label = label2id.get(emo, -1)

            batch_input_ids.append(input_ids)
            batch_attn.append(attn_mask)
            batch_labels.append(label)

    import torch
    if len(batch_input_ids) == 0 or len(batch_video) == 0 or len(batch_labels) == 0:
        return None

    batch_input_tokens = torch.stack(batch_input_ids, dim=0)  # [B, 512]
    batch_attention_masks = torch.stack(batch_attn, dim=0)    # [B, 512]

    # 비디오 패딩(기존 함수 사용)
    batch_video = padding_video(batch_video)
    batch_video_2 = padding_video(batch_video_2) if len(batch_video_2) > 0 else None

    batch_labels = torch.tensor(batch_labels, dtype=torch.long)

    # 안전장치(디버깅용): 길이/라벨 범위 확인
    with torch.no_grad():
        max_len = (batch_input_tokens != 0).sum(dim=1).max().item()
        assert max_len <= 512, f"max_len={max_len} > 512 (truncation 누락)"
        mn, mx = int(batch_labels.min().item()), int(batch_labels.max().item())
        assert mn >= -1 and mx < len(LABELS), f"label range [{mn},{mx}] out of [0,{len(LABELS)-1}]"

    return batch_input_tokens, batch_attention_masks, batch_video, batch_video_2, batch_labels


def video_reverse_batchs_pickle_bc(sessions):
    batch_input_ids, batch_attn, batch_video, batch_video_2, batch_labels = [], [], [], [], []

    for session in sessions:
        inputString = ""
        now_speaker = None
        last_emotion, last_wav = None, None

        for turn, line in enumerate(session):
            speaker, utt, wav_path, video_path, start_time, end_time, emotion, bbox_list = line
            inputString += f"<s{speaker+1}> {utt} "
            now_speaker = speaker
            last_emotion = emotion
            last_wav = wav_path

        prompt = f"지금 <s{now_speaker+1}> 의 감정은"
        concat_string = (inputString.strip() + " </s> " + prompt).strip()

        # 비디오 특징 추출(기존 로직 유지)
        if last_wav is not None:
            video_input, video_input_2 = get_video_bbox_bc(
                video_processor, video_path, last_wav, start_time, end_time, bbox_list
            )
            batch_video.append(video_input)
            batch_video_2.append(video_input_2)

            # ★ 핵심: 토크나이저에서 무조건 512로 자르고 패딩
            enc = roberta_tokenizer(
                concat_string,
                truncation=True,
                max_length=512,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].squeeze(0)        # [512]
            attn_mask = enc["attention_mask"].squeeze(0)   # [512]

            # 라벨 매핑: 모르는 값은 -1 (ignore)
            emo = (last_emotion or "").strip()
            label = label2id.get(emo, -1)

            batch_input_ids.append(input_ids)
            batch_attn.append(attn_mask)
            batch_labels.append(label)

    import torch
    if len(batch_input_ids) == 0 or len(batch_video) == 0 or len(batch_labels) == 0:
        return None

    batch_input_tokens = torch.stack(batch_input_ids, dim=0)  # [B, 512]
    batch_attention_masks = torch.stack(batch_attn, dim=0)    # [B, 512]

    # 비디오 패딩(기존 함수 사용)
    batch_video = padding_video(batch_video)
    batch_video_2 = padding_video(batch_video_2) if len(batch_video_2) > 0 else None

    batch_labels = torch.tensor(batch_labels, dtype=torch.long)

    # 안전장치(디버깅용): 길이/라벨 범위 확인
    with torch.no_grad():
        max_len = (batch_input_tokens != 0).sum(dim=1).max().item()
        assert max_len <= 512, f"max_len={max_len} > 512 (truncation 누락)"
        mn, mx = int(batch_labels.min().item()), int(batch_labels.max().item())
        assert mn >= -1 and mx < len(LABELS), f"label range [{mn},{mx}] out of [0,{len(LABELS)-1}]"

    return batch_input_tokens, batch_attention_masks, batch_video, batch_video_2, batch_labels

def video_reverse_batchs_pickle_abc(sessions):
    batch_input_ids, batch_attn, batch_video, batch_video_2, batch_labels = [], [], [], [], []
    batch_video_3 = []
    for session in sessions:
        inputString = ""
        now_speaker = None
        last_emotion, last_wav = None, None

        for turn, line in enumerate(session):
            speaker, utt, wav_path, video_path, start_time, end_time, emotion, bbox_list = line
            inputString += f"<s{speaker+1}> {utt} "
            now_speaker = speaker
            last_emotion = emotion
            last_wav = wav_path

        prompt = f"지금 <s{now_speaker+1}> 의 감정은"
        concat_string = (inputString.strip() + " </s> " + prompt).strip()

        # 비디오 특징 추출(기존 로직 유지)
        if last_wav is not None:
            video_input, video_input_2, video_input_3 = get_video_bbox_abc(
                video_processor, video_path, last_wav, start_time, end_time, bbox_list
            )
            batch_video.append(video_input)
            batch_video_2.append(video_input_2)
            batch_video_3.append(video_input_3)

            # ★ 핵심: 토크나이저에서 무조건 512로 자르고 패딩
            enc = roberta_tokenizer(
                concat_string,
                truncation=True,
                max_length=512,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].squeeze(0)        # [512]
            attn_mask = enc["attention_mask"].squeeze(0)   # [512]

            # 라벨 매핑: 모르는 값은 -1 (ignore)
            emo = (last_emotion or "").strip()
            label = label2id.get(emo, -1)

            batch_input_ids.append(input_ids)
            batch_attn.append(attn_mask)
            batch_labels.append(label)

    import torch
    if len(batch_input_ids) == 0 or len(batch_video) == 0 or len(batch_labels) == 0:
        return None

    batch_input_tokens = torch.stack(batch_input_ids, dim=0)  # [B, 512]
    batch_attention_masks = torch.stack(batch_attn, dim=0)    # [B, 512]

    # 비디오 패딩(기존 함수 사용)
    batch_video = padding_video(batch_video)
    batch_video_2 = padding_video(batch_video_2) if len(batch_video_2) > 0 else None
    batch_video_3 = padding_video(batch_video_3)

    batch_labels = torch.tensor(batch_labels, dtype=torch.long)

    # 안전장치(디버깅용): 길이/라벨 범위 확인
    with torch.no_grad():
        max_len = (batch_input_tokens != 0).sum(dim=1).max().item()
        assert max_len <= 512, f"max_len={max_len} > 512 (truncation 누락)"
        mn, mx = int(batch_labels.min().item()), int(batch_labels.max().item())
        assert mn >= -1 and mx < len(LABELS), f"label range [{mn},{mx}] out of [0,{len(LABELS)-1}]"

    return batch_input_tokens, batch_attention_masks, batch_video, batch_video_2, batch_video_3, batch_labels


def make_batchs_pickle(sessions):
    batch_input_ids, batch_attn = [], []
    batch_audio, batch_video, batch_video_2, batch_labels = [], [], [], []
    max_length = 400000

    for session in sessions:
        inputString, now_speaker = "", None
        last_emotion, last_wav = None, None
        for turn, line in enumerate(session):
            speaker, utt, wav_path, video_path, start_time, end_time, emotion, bbox_list = line
            inputString += f"<s{speaker+1}> {utt} "
            now_speaker = speaker
            last_emotion = emotion
            last_wav = wav_path

        last_wav = remap_wav_path(last_wav)

        prompt = f"지금 <s{now_speaker+1}> 의 감정은"
        concat_string = (inputString.strip() + " </s> " + prompt).strip()

        # ---- 오디오 ----
        if last_wav is None:
            continue
        audio_input = get_audio(audio_processor, last_wav)
        if audio_input is None:
            continue
        audio_input = audio_input[-max_length:]
        batch_audio.append(audio_input)

        # ---- 비디오 ----
        video_input, video_input_2 = get_video_bbox(
            video_processor, video_path, last_wav, start_time, end_time, bbox_list
        )
        batch_video.append(video_input)
        batch_video_2.append(video_input_2)

        # ---- 텍스트 (무조건 512로 자르고 패딩) ----
        enc = roberta_tokenizer(
            concat_string,
            truncation=True,
            max_length=512,
            padding="max_length",
            return_tensors="pt",
        )
        batch_input_ids.append(enc["input_ids"].squeeze(0))       # [512]
        batch_attn.append(enc["attention_mask"].squeeze(0))       # [512]

        # ---- 라벨 ----
        emo = (last_emotion or "").strip()
        label = label2id.get(emo, -1)  # 모르는 값은 -1(ignore)
        batch_labels.append(label)

    import torch
    if not (batch_input_ids and batch_audio and batch_video and batch_labels):
        return None

    batch_input_tokens   = torch.stack(batch_input_ids, dim=0)            # [B,512]
    batch_attention_masks= torch.stack(batch_attn,      dim=0)            # [B,512]
    batch_audio          = padding_video(batch_audio)                      # 기존 헬퍼 활용
    batch_video          = padding_video(batch_video)
    batch_video_2        = padding_video(batch_video_2) if batch_video_2 else None
    batch_labels         = torch.tensor(batch_labels, dtype=torch.long)    # long 보장

    with torch.no_grad():
        max_len = (batch_input_tokens != 0).sum(dim=1).max().item()
        assert max_len <= 512, f"max_len={max_len} > 512 (truncation 누락)"
        mn, mx = int(batch_labels.min().item()), int(batch_labels.max().item())
        assert mn >= -1 and mx < len(LABELS), f"label range [{mn},{mx}] out of [0,{len(LABELS)-1}]"

    return batch_input_tokens, batch_attention_masks, batch_audio, batch_video, batch_video_2, batch_labels

def make_batchs_pickle_bc(sessions):
    batch_input_ids, batch_attn = [], []
    batch_audio, batch_video, batch_video_2, batch_labels = [], [], [], []
    max_length = 400000

    for session in sessions:
        inputString, now_speaker = "", None
        last_emotion, last_wav = None, None
        # session: [(speaker, utt, wav_path, video_path, start_time, end_time, emotion, bbox_list), ...]
        for turn, line in enumerate(session):
            speaker, utt, wav_path, video_path, start_time, end_time, emotion, bbox_list = line
            inputString += f"<s{speaker+1}> {utt} "
            now_speaker = speaker
            last_emotion = emotion
            last_wav = wav_path

        prompt = f"지금 <s{now_speaker+1}> 의 감정은"
        concat_string = (inputString.strip() + " </s> " + prompt).strip()

        if last_wav is None:
            # 오디오가 없으면 이 샘플 스킵(필요 시 정책 변경)
            continue

        # ---- 오디오 ----
        audio_input = get_audio(audio_processor, last_wav)
        if audio_input is None:
            return None  # 기존 정책 유지
        audio_input = audio_input[-max_length:]
        batch_audio.append(audio_input)

        # ---- 비디오 ----
        video_input, video_input_2 = get_video_bbox_bc(
            video_processor, video_path, last_wav, start_time, end_time, bbox_list
        )
        batch_video.append(video_input)
        batch_video_2.append(video_input_2)

        # ---- 텍스트 (무조건 512로 자르고 패딩) ----
        enc = roberta_tokenizer(
            concat_string,
            truncation=True,
            max_length=512,
            padding="max_length",
            return_tensors="pt",
        )
        batch_input_ids.append(enc["input_ids"].squeeze(0))       # [512]
        batch_attn.append(enc["attention_mask"].squeeze(0))       # [512]

        # ---- 라벨 ----
        emo = (last_emotion or "").strip()
        label = label2id.get(emo, -1)  # 모르는 값은 -1(ignore)
        batch_labels.append(label)

    import torch
    if not (batch_input_ids and batch_audio and batch_video and batch_labels):
        return None

    batch_input_tokens   = torch.stack(batch_input_ids, dim=0)            # [B,512]
    batch_attention_masks= torch.stack(batch_attn,      dim=0)            # [B,512]
    batch_audio          = padding_video(batch_audio)                      # 기존 헬퍼 활용
    batch_video          = padding_video(batch_video)
    batch_video_2        = padding_video(batch_video_2) if batch_video_2 else None
    batch_labels         = torch.tensor(batch_labels, dtype=torch.long)    # long 보장

    # 안전장치(디버깅용)
    with torch.no_grad():
        max_len = (batch_input_tokens != 0).sum(dim=1).max().item()
        assert max_len <= 512, f"max_len={max_len} > 512 (truncation 누락)"
        mn, mx = int(batch_labels.min().item()), int(batch_labels.max().item())
        assert mn >= -1 and mx < len(LABELS), f"label range [{mn},{mx}] out of [0,{len(LABELS)-1}]"

    return batch_input_tokens, batch_attention_masks, batch_audio, batch_video, batch_video_2, batch_labels

def make_batchs_pickle_abc(sessions):
    batch_input_ids, batch_attn = [], []
    batch_audio, batch_video, batch_video_2, batch_labels = [], [], [], []
    batch_video_3 = []
    max_length = 400000

    for session in sessions:
        inputString, now_speaker = "", None
        last_emotion, last_wav = None, None
        # session: [(speaker, utt, wav_path, video_path, start_time, end_time, emotion, bbox_list), ...]
        for turn, line in enumerate(session):
            speaker, utt, wav_path, video_path, start_time, end_time, emotion, bbox_list = line
            inputString += f"<s{speaker+1}> {utt} "
            now_speaker = speaker
            last_emotion = emotion
            last_wav = wav_path

        prompt = f"지금 <s{now_speaker+1}> 의 감정은"
        concat_string = (inputString.strip() + " </s> " + prompt).strip()

        if last_wav is None:
            # 오디오가 없으면 이 샘플 스킵(필요 시 정책 변경)
            continue

        # ---- 오디오 ----
        audio_input = get_audio(audio_processor, last_wav)
        if audio_input is None:
            return None  # 기존 정책 유지
        audio_input = audio_input[-max_length:]
        batch_audio.append(audio_input)

        # ---- 비디오 ----
        video_input, video_input_2, video_input_3 = get_video_bbox_abc(
            video_processor, video_path, last_wav, start_time, end_time, bbox_list
        )
        batch_video.append(video_input)
        batch_video_2.append(video_input_2)
        batch_video_3.append(video_input_3)

        # ---- 텍스트 (무조건 512로 자르고 패딩) ----
        enc = roberta_tokenizer(
            concat_string,
            truncation=True,
            max_length=512,
            padding="max_length",
            return_tensors="pt",
        )
        batch_input_ids.append(enc["input_ids"].squeeze(0))       # [512]
        batch_attn.append(enc["attention_mask"].squeeze(0))       # [512]

        # ---- 라벨 ----
        emo = (last_emotion or "").strip()
        label = label2id.get(emo, -1)  # 모르는 값은 -1(ignore)
        batch_labels.append(label)

    import torch
    if not (batch_input_ids and batch_audio and batch_video and batch_labels):
        return None

    batch_input_tokens   = torch.stack(batch_input_ids, dim=0)            # [B,512]
    batch_attention_masks= torch.stack(batch_attn,      dim=0)            # [B,512]
    batch_audio          = padding_video(batch_audio)                      # 기존 헬퍼 활용
    batch_video          = padding_video(batch_video)
    batch_video_2        = padding_video(batch_video_2) if batch_video_2 else None
    batch_video_3        = padding_video(batch_video_3) 
    batch_labels         = torch.tensor(batch_labels, dtype=torch.long)    # long 보장

    # 안전장치(디버깅용)
    with torch.no_grad():
        max_len = (batch_input_tokens != 0).sum(dim=1).max().item()
        assert max_len <= 512, f"max_len={max_len} > 512 (truncation 누락)"
        mn, mx = int(batch_labels.min().item()), int(batch_labels.max().item())
        assert mn >= -1 and mx < len(LABELS), f"label range [{mn},{mx}] out of [0,{len(LABELS)-1}]"

    return batch_input_tokens, batch_attention_masks, batch_audio, batch_video, batch_video_2, batch_video_3, batch_labels


def teacher_batchs(sessions):
    """
    sessions: DataLoader가 넘겨주는 batch 샘플들
              각 session은 여러 turn의 (speaker, utt, wav_path, video_path, start_time, end_time, emotion) 튜플 리스트라고 가정
    반환: (input_ids, attention_mask, labels)
    - 입력은 무조건 512로 truncate + max_length pad
    - 라벨은 [0..6] 또는 -1(ignore)
    """
    batch_input_ids = []
    batch_attn = []
    batch_labels = []

    for session in sessions:
        inputString = ""
        now_speaker = None
        last_emotion = None
        last_wav = None

        for turn, line in enumerate(session):
            speaker, utt, wav_path, video_path, start_time, end_time, emotion = line
            inputString += f"<s{speaker+1}> {utt} "
            now_speaker = speaker
            last_emotion = emotion
            last_wav = wav_path

        # 프롬프트 구성
        prompt = f"지금 <s{now_speaker+1}> 의 감정은"
        concat_string = (inputString.strip() + " </s> " + prompt).strip()

        # 텍스트가 없거나 wav가 비어있으면 스킵 (필요없다면 조건 제거)
        if last_wav is None:
            continue

        # ---- 핵심: 토크나이즈 시 무조건 512로 자르고 패딩 ----
        enc = roberta_tokenizer(
            concat_string,
            truncation=True,        # ★ 필수
            max_length=512,         # ★ 필수
            padding="max_length",   # ★ 배치간 길이 맞춤
            return_tensors="pt"
        )
        input_ids = enc["input_ids"].squeeze(0)        # [512]
        attn_mask = enc["attention_mask"].squeeze(0)   # [512]

        # 라벨 매핑: 모르는 값은 -1(ignore)
        emo = (last_emotion or "").strip()
        label = label2id.get(emo, -1)

        batch_input_ids.append(input_ids)
        batch_attn.append(attn_mask)
        batch_labels.append(label)

    import torch
    if len(batch_input_ids) == 0:
        # DataLoader가 None을 받으면 터질 수 있으니, 빈 배치는 건너뛰도록 상위 루프에서 continue 처리 권장
        return None

    batch_input_ids = torch.stack(batch_input_ids, dim=0)  # [B,512]
    batch_attn      = torch.stack(batch_attn,      dim=0)  # [B,512]
    batch_labels    = torch.tensor(batch_labels, dtype=torch.long)  # long 필수

    # 안전장치(디버깅용): 길이/라벨 범위 확인
    with torch.no_grad():
        max_len = (batch_input_ids != 0).sum(dim=1).max().item()
        assert max_len <= 512, f"max_len={max_len} > 512 (truncation 미적용)"
        mn, mx = int(batch_labels.min().item()), int(batch_labels.max().item())
        assert mn >= -1 and mx < len(LABELS), f"label range [{mn},{mx}] out of [0,{len(LABELS)-1}]"

    return batch_input_ids, batch_attn, batch_labels
    


def teacher_batchs_no_prompt(sessions):
    """
    sessions: DataLoader가 넘겨주는 batch 샘플들
              각 session은 여러 turn의 (speaker, utt, wav_path, video_path, start_time, end_time, emotion) 튜플 리스트라고 가정
    반환: (input_ids, attention_mask, labels)
    - 입력은 무조건 512로 truncate + max_length pad
    - 라벨은 [0..6] 또는 -1(ignore)
    """
    batch_input_ids = []
    batch_attn = []
    batch_labels = []

    for session in sessions:
        inputString = ""
        now_speaker = None
        last_emotion = None
        last_wav = None

        for turn, line in enumerate(session):
            speaker, utt, wav_path, video_path, start_time, end_time, emotion = line
            inputString += f"<s{speaker+1}> {utt} "
            now_speaker = speaker
            last_emotion = emotion
            last_wav = wav_path

        # 프롬프트 구성
        # prompt = f"지금 <s{now_speaker+1}> 의 감정은"
        # concat_string = (inputString.strip() + " </s> " + prompt).strip()
        concat_string = inputString.strip()

        # 텍스트가 없거나 wav가 비어있으면 스킵 (필요없다면 조건 제거)
        if last_wav is None:
            continue

        # ---- 핵심: 토크나이즈 시 무조건 512로 자르고 패딩 ----
        enc = roberta_tokenizer(
            concat_string,
            truncation=True,        # ★ 필수
            max_length=512,         # ★ 필수
            padding="max_length",   # ★ 배치간 길이 맞춤
            return_tensors="pt"
        )
        input_ids = enc["input_ids"].squeeze(0)        # [512]
        attn_mask = enc["attention_mask"].squeeze(0)   # [512]

        # 라벨 매핑: 모르는 값은 -1(ignore)
        emo = (last_emotion or "").strip()
        label = label2id.get(emo, -1)

        batch_input_ids.append(input_ids)
        batch_attn.append(attn_mask)
        batch_labels.append(label)

    import torch
    if len(batch_input_ids) == 0:
        # DataLoader가 None을 받으면 터질 수 있으니, 빈 배치는 건너뛰도록 상위 루프에서 continue 처리 권장
        return None

    batch_input_ids = torch.stack(batch_input_ids, dim=0)  # [B,512]
    batch_attn      = torch.stack(batch_attn,      dim=0)  # [B,512]
    batch_labels    = torch.tensor(batch_labels, dtype=torch.long)  # long 필수

    # 안전장치(디버깅용): 길이/라벨 범위 확인
    with torch.no_grad():
        max_len = (batch_input_ids != 0).sum(dim=1).max().item()
        assert max_len <= 512, f"max_len={max_len} > 512 (truncation 미적용)"
        mn, mx = int(batch_labels.min().item()), int(batch_labels.max().item())
        assert mn >= -1 and mx < len(LABELS), f"label range [{mn},{mx}] out of [0,{len(LABELS)-1}]"

    return batch_input_ids, batch_attn, batch_labels

def audio_batchs(sessions):
    batch_input_ids, batch_attn = [], []
    batch_audio, batch_labels = [], []
    max_length = 400000

    for session in sessions:
        inputString, now_speaker = "", None
        last_emotion, last_wav = None, None

        for turn, line in enumerate(session):
            # session 원소: (speaker, utt, wav_path, video_path, start_time, end_time, emotion)
            speaker, utt, wav_path, video_path, start_time, end_time, emotion = line
            inputString += f"<s{speaker+1}> {utt} "
            now_speaker = speaker
            last_emotion = emotion
            last_wav = wav_path

        # 오디오 없으면 스킵(정책에 맞게 조정 가능)
        if last_wav is None:
            continue

        # ---- 오디오 ----
        audio_input = get_audio(audio_processor, last_wav)
        if audio_input is None:
            continue
        audio_input = audio_input[-max_length:]
        batch_audio.append(audio_input)

        # ---- 텍스트 (무조건 512로 truncate + pad) ----
        prompt = f"지금 <s{now_speaker+1}> 의 감정은"
        concat_string = (inputString.strip() + " </s> " + prompt).strip()

        enc = roberta_tokenizer(
            concat_string,
            truncation=True,          # ★ 필수: 512 초과 컷
            max_length=512,           # ★ 필수: 모델 한계
            padding="max_length",     # 배치 길이 동일화
            return_tensors="pt",
        )
        batch_input_ids.append(enc["input_ids"].squeeze(0))     # [512]
        batch_attn.append(enc["attention_mask"].squeeze(0))     # [512]

        # ---- 라벨 ----
        emo = (last_emotion or "").strip()
        label = label2id.get(emo, -1)  # 매핑 실패 시 -1(ignore)
        batch_labels.append(label)

    import torch
    if not (batch_input_ids and batch_audio and batch_labels):
        return None

    batch_input_tokens    = torch.stack(batch_input_ids, dim=0)           # [B, 512]
    batch_attention_masks = torch.stack(batch_attn,      dim=0)           # [B, 512]
    batch_audio           = padding_video(batch_audio)                     # 기존 헬퍼 사용
    batch_labels          = torch.tensor(batch_labels, dtype=torch.long)   # long 보장

    # 안전장치(디버깅용): 길이/라벨 범위 즉시 확인
    with torch.no_grad():
        max_len = (batch_input_tokens != 0).sum(dim=1).max().item()
        assert max_len <= 512, f"max_len={max_len} > 512 (truncation 누락)"
        mn, mx = int(batch_labels.min().item()), int(batch_labels.max().item())
        assert mn >= -1 and mx < len(LABELS), f"label range [{mn},{mx}] out of [0,{len(LABELS)-1}]"

    return batch_input_tokens, batch_attention_masks, batch_audio, batch_labels