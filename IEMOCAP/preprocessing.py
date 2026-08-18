"""IEMOCAP CSV preprocessing into per-session dialogue samples (with face-bbox pickle)."""
import os
import csv
import pickle

def split(session):
    # CONTEXT_LEN env var caps the preceding-utterance context fed to the text
    # encoder (context-length ablation, CMC revision):
    #   CONTEXT_LEN=k (>=0) -> keep target + last k preceding utterances
    #   unset / "full" / "-1" -> keep all preceding utterances (default behaviour)
    _cl = os.environ.get("CONTEXT_LEN", "full")
    max_ctx = None if _cl in ("", "full", "-1") else int(_cl)
    final_data = []
    split_session = []
    for line in session:
        split_session.append(line)
        if max_ctx is None:
            final_data.append(split_session[:])
        else:
            final_data.append(split_session[-(max_ctx + 1):])
    return final_data

def preprocessing(data_path):
  f = open(data_path, 'r')
  rdr = csv.reader(f)
        
  session_dataset = []
  session = []
  speaker_set = []

  pre_sess = 'start'
  for i, line in enumerate(rdr):
    if i == 0:
      header  = line
      utt_idx = header.index('Utterance')
      speaker_idx = header.index('Speaker')
      emo_idx = header.index('Emotion')
      sess_idx = header.index('Dialogue_ID')
      wav_idx = header.index('Wav_Path')
      video_idx = header.index('Video_Path')
      start_idx = header.index('Start_Time')
      end_idx = header.index('End_Time')

    else:
      utt = line[utt_idx]
      speaker = line[speaker_idx]
      if speaker in speaker_set:
        uniq_speaker = speaker_set.index(speaker)
      else:
        speaker_set.append(speaker)
        uniq_speaker = speaker_set.index(speaker)
      emotion = line[emo_idx]
      sess = line[sess_idx]
      if sess == '':
        sess_speaker = line[speaker_idx]
        sess = sess_speaker[:-2]
      video_path = line[video_idx]
      wav_path = line[wav_idx]
      start_time = line[start_idx]
      end_time = line[end_idx]
      if pre_sess == 'start' or sess == pre_sess:
        session.append([uniq_speaker, utt, wav_path, video_path, start_time, end_time, emotion])

      else:
        session_dataset += split(session)
        session = [[uniq_speaker, utt, wav_path, video_path, start_time, end_time, emotion]]
        speaker_set = []
      pre_sess = sess   
  session_dataset += split(session)           
  f.close()
  return session_dataset


def preprocessing_pickle(data_path, pickle_path):
  f = open(data_path, 'r')
  rdr = csv.reader(f)
  with open(pickle_path, 'rb') as pk:
    bbox = pickle.load(pk)      
  session_dataset = []
  session = []
  speaker_set = []

  pre_sess = 'start'
  for i, line in enumerate(rdr):
    if i == 0:
      header  = line
      utt_idx = header.index('Utterance')
      speaker_idx = header.index('Speaker')
      emo_idx = header.index('Emotion')
      sess_idx = header.index('Dialogue_ID')
      wav_idx = header.index('Wav_Path')
      video_idx = header.index('Video_Path')
      start_idx = header.index('Start_Time')
      end_idx = header.index('End_Time')
    else:
      utt = line[utt_idx]
      speaker = line[speaker_idx]
      emotion = line[emo_idx]
      sess = line[sess_idx]
      if sess == '':
        sess = speaker[:-2]

      # Flush on every dialogue boundary, including when the first target in
      # the new dialogue has label -1.  The old -1 branch appended to the old
      # dialogue instead of resetting, duplicating/cross-contaminating samples.
      if pre_sess != 'start' and sess != pre_sess:
        session_dataset += split(session)
        session = []
        speaker_set = []

      if speaker not in speaker_set:
        speaker_set.append(speaker)
      uniq_speaker = speaker_set.index(speaker)

      if emotion != '-1':
        video_path = line[video_idx]
        wav_path = line[wav_idx]
        start_time = line[start_idx]
        end_time = line[end_idx]
        key = wav_path.split('/')[-1].split('.')[0]
        bbox_list = bbox[key]
        item = [uniq_speaker, utt, wav_path, video_path, start_time, end_time, emotion, bbox_list]
      else:
        item = [uniq_speaker, utt, None, None, 0, 0, emotion, None]
      session.append(item)
      pre_sess = sess
  session_dataset += split(session)           
  f.close()
  return session_dataset
