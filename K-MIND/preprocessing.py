"""K-MIND CSV preprocessing: build per-dialogue session lists (optionally with face bbox pickles)."""
import csv
import pickle

def split(session):
    final_data = []
    split_session = []
    for line in session:
        split_session.append(line)
        final_data.append(split_session[:])    
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
      utt_idx = header.index('\ufeffUtterance')
      speaker_idx = header.index('Speaker')
      emo_idx = header.index('Affective_State')
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
      utt_idx = header.index('\ufeffUtterance')
      speaker_idx = header.index('Speaker')
      emo_idx = header.index('Affective_State')
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
      
      if emotion != '기타':
        video_path = line[video_idx]
        wav_path = line[wav_idx]
        start_time = line[start_idx]
        end_time = line[end_idx]
        # key = wav_path.split('/')[-1].split('.')[0]
        key = video_path
        if key in bbox.keys():
          bbox_list = bbox[key]
        else:
          bbox_list = None
        if pre_sess == 'start' or sess == pre_sess:
          session.append([uniq_speaker, utt, wav_path, video_path, start_time, end_time, emotion, bbox_list])
        else:
          session_dataset += split(session)
          session = [[uniq_speaker, utt, wav_path, video_path, start_time, end_time, emotion, bbox_list]]

      else:
        if pre_sess == 'start' or sess == pre_sess:
          session.append([uniq_speaker, utt, None, None, 0, 0, emotion, None])
        else:
          session_dataset += split(session)
          session.append([uniq_speaker, utt, None, None, 0, 0, emotion, None]) 

      pre_sess = sess   
  session_dataset += split(session)           
  f.close()
  return session_dataset