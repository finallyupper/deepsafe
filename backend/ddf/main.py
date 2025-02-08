import torch
import os
import cv2
from ddf.DualDefense_gan_fs import DualDefense 
import warnings
warnings.filterwarnings('ignore')
from PIL import Image
from torchvision.transforms import transforms
import numpy as np 
from ddf.engine.utils import blend_image 

_DDF_MODEL_MAPPINGS = {
    "byeon_cha": "/home/yoojinoh/Others/ckpts/ckpt_best_img_lam1_al2.pt",
    "win_chuu": "/home/yoojinoh/Others/ckpts/winchuu_ckpt_best_img_lam1_al2.pt",
}

_TEST_CONFIGS = {
    "message_size": {'byeon_cha': 4, 'win_chuu': 15},
    "height": 160,
    "width": 160
}

#NOTE(Yoojin): 임시 유저-메세지 데이터
USER_WATERMARK_IDS = {
    "byeon": [0., 1., 0., 1.],
    "cha": [1., 1., 0., 0.],
    "win": [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    "chu": [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]
} 

def assign_messages(users, device='cpu'):
    num_users = len(users)
    max_users = 16
    assert num_users <= max_users, f"The number of maximum users is {max_users}."
    unique_messages = torch.eye(max_users, MESSAGE_SIZE, dtype=torch.float).to(device)
    user_messages = {user: unique_messages[i].tolist() for i, user in enumerate(users)}
    #TODO(Yoojin): Save info to DB
    return user_messages    

def find_message2user(message: list) -> str:
    message_array = np.array(message, dtype=np.float64)

    best_match = None
    min_distance = float('inf')  

    for user, user_vector in USER_WATERMARK_IDS.items():
        user_array = np.array(user_vector, dtype=np.float64)

        if np.array_equal(user_array, message_array): 
            return user
        distance = np.linalg.norm(user_array - message_array)

        if distance < min_distance:  
            min_distance = distance
            best_match = user

    return best_match 

    
def encode_image(model, image, message):
    """insert watermark into original image"""
    if len(image.shape) == 3:
        image = image.unsqueeze(0) 
    return model.encode(image, message)

def save_image(image, save_path):
    cv2.imwrite(save_path, image)
    print(f'[INFO] Saved image to {save_path}')

def decode_image(model, encoded_image):
    return model.decode(encoded_image) 

def get_transform(bgr_image, height, width):
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((height, width))]
    )
    return transform(bgr_image)

def load_image(image_path):
    image = Image.fromarray(cv2.imread(image_path)) #BGR image
    return image

def load_model(model_type, mode = 'eval', device=None):# Modified
    model = DualDefense(_TEST_CONFIGS['message_size'][model_type], in_channels=3, model_type=model_type, device=device) 
    model.encoder.load_state_dict(torch.load(_DDF_MODEL_MAPPINGS[model_type])['encoder'], strict=False)
    model.decoder.load_state_dict(torch.load(_DDF_MODEL_MAPPINGS[model_type])['decoder'], strict=False) 
    
    if mode == 'eval':
        model.encoder.eval()
        model.decoder.eval()
    return model 

def apply_faceswap(model_type, swapped_image_path, src_path, tgt_path, src_user, encoded=True):
    """
    Apply dual defense and save the faceswapped image (swap face a with face b)

    Args:
        model_type (str) : type of dual defense model to apply. (byeon_cha, win_chu)
        swapped_image_path (str) : path to save the face swapped image
        src_path (str): path of (encoded) image a
        tgt_path (str): path of (encoded) image b
        encoded (bool): if the given image is encoded or not
    """
    device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
    assert model_type in _DDF_MODEL_MAPPINGS.keys() 
    model = load_model(model_type, mode='eval', device=device)

    original_src_image, src_image, src_coord = crop_image(src_path) # load_image(src_path) 
    original_tgt_image, tgt_image, tgt_coord = crop_image(tgt_path) # load_image(tgt_path)

    src_image = Image.fromarray(src_image)
    tgt_image = Image.fromarray(tgt_image)

    src_image = get_transform(src_image, _TEST_CONFIGS['height'], _TEST_CONFIGS['width']).to(device).unsqueeze(0)
    tgt_image = get_transform(tgt_image, _TEST_CONFIGS['height'], _TEST_CONFIGS['width']).to(device).unsqueeze(0) 

    with torch.no_grad():
        src_image = src_image.to(device) 
        tgt_image = tgt_image.to(device) 

        # Apply faceswap 
        assert src_user in ['cha', 'byeon', 'win', 'chu'], f"There is no user named {src_user}"
        if src_user in ['cha', 'chu']:     
            _, _, _image_a_deepfake = model.deepfake1(src_image, 'A') 
            _, _, _image_b_deepfake = model.deepfake1(tgt_image, 'B')
        else: # ['byeon', 'win']
            _, _, _image_a_deepfake = model.deepfake1(src_image, 'B') 
            _, _, _image_b_deepfake = model.deepfake1(tgt_image, 'A')

        image_a_deepfake =(_image_a_deepfake[0]  * 255).permute(1, 2, 0).detach().cpu().numpy()
        image_b_deepfake =(_image_b_deepfake[0]  * 255).permute(1, 2, 0).detach().cpu().numpy()

        save_image(image_a_deepfake, os.path.join(swapped_image_path, 'output_A2B.png'))
        save_image(image_b_deepfake, os.path.join(swapped_image_path, 'output_B2A.png'))

        if encoded:
            _pred_a_message = model.decode(_image_a_deepfake)        
            _pred_b_message = model.decode(_image_b_deepfake) 

            pred_a_message = list(map(int, _pred_a_message[0])) 
            pred_b_message = list(map(int, _pred_b_message[0])) 
        
            usera = find_message2user(pred_a_message)
            userb = find_message2user(pred_b_message)

            print(f'>> Someone tried to make deepfake with user {usera} and user {userb}')
            src_output_path = os.path.join(swapped_image_path, 'output_A2B.png')
            return [
                {"source_image_url": src_output_path,
                 "user_prediction": f"Someone tried to make deepfake with user {usera}"}
                 ]

        # restore_original(original_src_image, _image_a_deepfake[0], src_coord, os.path.join(swapped_image_path, 'output_A2B.png')) 
        # restore_original(original_tgt_image, _image_b_deepfake[0], tgt_coord, os.path.join(swapped_image_path, 'output_B2A.png'))  

# def crop_image(image_path):
#     image_path = os.path.join('ddf', image_path) # ADD(YOOJIN): Path error in backend
#     cascade = cv2.CascadeClassifier('ddf/data/haarcascade_frontalface_alt.xml') 
#     image = cv2.imread(image_path)
#     results = cascade.detectMultiScale(image)

#     if len(results) == 0:
#         raise ValueError("No face detected in the image.")
    
#     crop_coord = results[0] 
#     (x, y, w, h) = crop_coord
#     cropped_face = image[y:y+h, x:x+w]
#     original_image = image.copy()
#     print("Sucessfully Cropped image")
#     return original_image, cropped_face, crop_coord

def restore_original(original_image, encoded_face, coord, result_path):
    (x, y, w, h) = coord 

    encoded_face = (encoded_face * 255).permute(1, 2, 0).detach().cpu().numpy()
    encoded_face = np.clip(encoded_face, 0, 255)

    resized_encoded_face = cv2.resize(encoded_face, (w, h))
    
    result_image = original_image.astype(np.float32)
    result_image[y:y+h, x:x+w] = resized_encoded_face
    save_image(result_image, result_path) 
    print("Restored to original image")

def crop_and_encode_image(model_type, image_path, message, device, alpha=1.0):
    """
    Inputs original image, and save image with encoded face
    args:
        model_type (str): ['byeon_cha', 'win_chuu'] 
        image_path (str): path of the image to encode
        message (list) : [1., 0., 1] ->ex. torch.randint(0, 2, (1, 4), dtype=torch.float).to(device).detach()
        device : cpu or gpu
        alpha (float) : the amount to blend encoded image with original image 
    """    
    original_image, cropped_face, (x, y, w, h) = crop_image(image_path)
    transformed_cropped_face = get_transform(cropped_face, 160, 160).to(device)

    model = load_model(model_type, 'eval', device)
    message = torch.FloatTensor(message).to(device).detach()
    encoded_face = encode_image(model, transformed_cropped_face, message)[0]
    
    #Modified(Yoojin)
    encoded_face = blend_image(encoded_face, transformed_cropped_face).to(device)

    print(f'Successfully encoded image with message {message}')
    result_path = os.path.join(os.path.dirname(image_path), 'encoded_' + os.path.basename(image_path))
    restore_original(original_image, encoded_face, (x, y, w, h), result_path) 

from mtcnn import MTCNN
import cv2

def crop_image(image_path):
    detector = MTCNN()
    print(f"[DEBUG] Image path = {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Error loading image from path: {image_path}")

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    detections = detector.detect_faces(rgb_image)

    if not detections:
        raise ValueError("No face detected in the image.")

    # 첫 번째 얼굴 검출 결과 사용
    x, y, w, h = detections[0]['box']
    cropped_face = image[y:y+h, x:x+w]

    return image, cropped_face, (x, y, w, h)


# if __name__ == "__main__":
#     device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
#     #TODO(Yoojin): 개별 user를 위한 watermark information으로 수정
#     crop_and_encode_image('byeon_cha', 
#                           '/home/yoojinoh/Others/byeon_total5.jpg', 
#                           USER_WATERMARK_IDS['byeon'], 
#                           device) 
#     crop_and_encode_image('byeon_cha', 
#                           '/home/yoojinoh/Others/cha_total2.jpg', 
#                           USER_WATERMARK_IDS['cha'], 
#                           device) 
#     apply_faceswap(model_type="byeon_cha",  swapped_image_path="/home/yoojinoh/Others/", 
#                    src_path="/home/yoojinoh/Others/encoded_byeon_total5.jpg", 
#                    tgt_path="/home/yoojinoh/Others/encoded_cha_total2.jpg")
