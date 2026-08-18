"""K-MIND torch Dataset wrapper with the Ekman 7-way label list."""
from torch.utils.data import Dataset


class kimind_dataset(Dataset):
    def __init__(self, data):
        # self.emoList = ["중립","행복","슬픔","분노","놀람","경멸","공포","혐오","기타"]
        # self.emoList = ["기대","기쁨","놀람","농담","머뭇거림","멸시","무서움","수치","슬픔","염려/걱정","중립","지루함","혐오","화남","흥미"]
        # self.emoList = ["기쁨","놀람","무서움","슬픔","중립","혐오","화남"]
        self.emoList = ["기쁨", "슬픔", "화남", "놀람", "무서움", "혐오", "중립"]
        self.session_dataset = data

    def __len__(self): 
        return len(self.session_dataset)
    
    def __getitem__(self, idx): 
        return self.session_dataset[idx]



# 감정 (7개)
# Neutral / Sad / Anger(Angry) / Disgust / Surprise(Surprised) / Fear / Contempt
# 태도 
# Interest / Joy / Concern / Expectation / Hesitation / Joke / Shame / Boredom