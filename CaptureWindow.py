import win32gui
import time
import mss
import numpy as np
import cv2
import segmentation_models as sm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from Autoencoder import Encoder
import tensorflow as tf
from skimage.draw import ellipse
import pyvjoy

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)


MAX_VJOY = 32768
j = pyvjoy.VJoyDevice(1)
j.reset_buttons()
j.reset_povs()
j.reset()
control_di = np.zeros(1)
print(np.shape(control_di))

class process_mask():
    def __init__(self):
        self.imsize = 11
        self.ratio_mainaxis = 4
        self.rotations = [90]
        self.kernel = np.zeros((self.imsize, self.imsize, len(self.rotations)), dtype=np.uint8)
        self.kernel_square = np.ones((3, 3), dtype=np.uint8)
        for i in range(0, len(self.rotations)):
          rr, cc = ellipse(self.imsize/2, self.imsize/2, self.imsize/2, self.imsize/(2*self.ratio_mainaxis), rotation=np.deg2rad(self.rotations[i]))
          self.kernel[rr, cc, i] = 1

    def process_mask(self, m):
        s = np.shape(m)
        m = cv2.dilate(m, self.kernel_square, iterations=10, borderValue=1)
        m = cv2.erode(m, self.kernel_square, iterations=10, borderValue=0)
        m = cv2.erode(m, self.kernel_square, iterations=10, borderValue=0)
        m = cv2.dilate(m, self.kernel_square, iterations=10, borderValue=1)
        m = cv2.erode(m, self.kernel[:, :, 0], iterations=5, borderValue=1)
        return m

def reset_gamecontrolls():
    j.data.wAxisXRot = int(MAX_VJOY/2)
    j.data.wAxisYRot = int(MAX_VJOY/2)
    j.data.wAxisY = int(MAX_VJOY/2)
    j.data.wAxisX = int(MAX_VJOY/2)
    j.data.wAxisZ = 0
    j.data.wAxisZRot = 0
    j.update()
    return

def update_gamecontrolls(ai,di):
    global control_di
    control_di[0] = di
    di = np.mean(control_di, axis=0)
    di = int(((di) * (MAX_VJOY/2))+MAX_VJOY/2)
    di = np.clip(di, 0, MAX_VJOY)
    j.data.wAxisX = di
    j.update()
    control_di = np.roll(control_di, -1)

device = torch.device('cpu')
class PolicyFunction(nn.Module):
    def __init__(self, input_shape, action_dimension, hidden_units, log_std_min_max=20):
        super(PolicyFunction, self).__init__()

        self.log_std_min_max = log_std_min_max
        self.lay1 = nn.Linear(input_shape, hidden_units)
        self.lay2 = nn.Linear(hidden_units, hidden_units)
        self.mean = nn.Linear(hidden_units, action_dimension)
        self.std = nn.Linear(hidden_units, action_dimension)

    def forward(self, state):
        x = F.relu(self.lay1(state))
        x = F.relu(self.lay2(x))
        log_std = torch.clamp(self.std(x), -self.log_std_min_max,
                              self.log_std_min_max)
        return self.mean(x), log_std

    def get_action_log(self, state, EPS=1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        z = Normal(0, 1).sample().to(device)
        action = torch.tanh(mean + std * z)
        log_prob = Normal(mean, std).log_prob(mean + std * z)
        log_prob -= torch.log(1 - action.pow(2) + EPS)
        log_prob = torch.sum(log_prob, dim=1).unsqueeze(1)
        return action, log_prob

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        mean, log_std = self.forward(state)
        z = Normal(0, 1).sample().to(device)
        action = torch.tanh(mean + log_std.exp() * z).cpu()
        return action.squeeze().detach().numpy()


model = sm.Unet('resnet50', classes=2)
model.compile(
    optimizer='sgd',
    loss=sm.losses.bce_jaccard_loss,
    metrics=[sm.metrics.iou_score],)
model.load_weights("Weights/43-0.1618-0.7353_v2.h5")
hwnd = win32gui.FindWindow(None, "Grand Theft Auto V")
rect = win32gui.GetWindowRect(hwnd)
x = rect[0]
y = rect[1]
w = rect[2] - x
h = rect[3] - y
enc = Encoder()
m = torch.load('Weights/policy_network.pt', map_location=torch.device('cpu'))
print(x)
print(y)
print(w)
print(h)
past_actions = np.zeros([2, 2])
reset_gamecontrolls()
mask = process_mask()

with mss.mss() as sct:
    while "Screen capturing":
        last_time = time.time()
        img = np.array(sct.grab(rect))
        orig_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.expand_dims(cv2.resize(orig_img, (320, 160)), axis=0)
        tmp = model.predict(img)
        img = np.squeeze(img)
        tmp = np.squeeze(tmp)
        tmp[:, :, 0] = mask.process_mask(tmp[:, :, 0])
        tmp_2 = cv2.resize(abs(255 - tmp*255)[:, :, 0], (320, 176))
        action_space = enc.Encode_img((tmp_2).T)
        action_space = np.append(action_space, past_actions.flatten())
        past_actions = np.roll(past_actions, 1)
        action = m.get_action(action_space)
        past_actions[0, 0] = action[0]
        past_actions[0, 1] = action[1]
        update_gamecontrolls(action[0], action[1])
        img = [(img[:, :, 0]), (img[:, :, 1] + np.squeeze(tmp[:, :, 0] * 255)),
               (img[:, :, 2])]
        img = np.clip(np.transpose(img, (1, 2, 0)), 0, 255)
        img = cv2.cvtColor(img.astype("uint8"), cv2.COLOR_RGB2BGR)
        cv2.imshow("OpenCV/Numpy normal", np.squeeze(img))
        print("fps: {}".format(1 / (time.time() - last_time)))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break
