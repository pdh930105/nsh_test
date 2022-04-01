import logging
import pytz

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count 
    

def make_logger(name=None, file_path="./result.log", clean_logger=False):


    #1 logger instance를 만든다.
    
    logger = logging.getLogger(name)

    #2 logger의 level을 가장 낮은 수준인 DEBUG로 설정해둔다.
    logger.setLevel(logging.DEBUG)

    #3 formatter 지정
    formatter = logging.Formatter("[%(asctime)s | %(name)s] - %(message)s", "%Y-%m-%d %H:%M:%S")
    
    #4 handler instance 생성
    console = logging.StreamHandler()
    if clean_logger:
        file_handler = logging.FileHandler(filename=file_path, mode="w")
    else:
        file_handler = logging.FileHandler(filename=file_path)
    
    #5 handler 별로 다른 level 설정
    console.setLevel(logging.INFO)
    file_handler.setLevel(logging.DEBUG)

    #6 handler 출력 format 지정
    console.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    #7 logger에 handler 추가
    logger.addHandler(console)
    logger.addHandler(file_handler)

    return logger