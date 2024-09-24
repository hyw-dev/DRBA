# for high quality output
import subprocess
from queue import Queue
import cv2
import _thread
from tqdm import tqdm
import torch
import numpy as np
import time
from models.model_pg104.softsplat import softsplat as warp
from models.model_pg104.GMFSS import Model
from models.rife_422_lite.IFNet_HDv3 import IFNet
import warnings

warnings.filterwarnings("ignore")

input = r'E:\01.mkv'  # input video path
output = r'D:\tmp\output.mkv'  # output video path
scale = 1.0  # flow scale
times = 5  # Must be an integer multiple
global_size = (1920, 1080)  # frame output resolution
hwaccel = True  # Use hardware acceleration video encoder


def generate_frame_renderer(input_path, output_path):
    video_capture = cv2.VideoCapture(input_path)
    read_fps = video_capture.get(cv2.CAP_PROP_FPS)
    encoder = 'libx264'
    preset = 'medium'
    if hwaccel:
        encoder = 'h264_nvenc'
        preset = 'p7'
    ffmpeg_cmd = [
        'ffmpeg', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-r', f'{read_fps * times}',
        '-s', f'{global_size[0]}x{global_size[1]}',
        '-i', 'pipe:0', '-i', input_path,
        '-map', '0:v', '-map', '1:a',
        '-c:v', encoder, "-movflags", "+faststart", "-pix_fmt", "yuv420p", "-qp", "16", '-preset', preset,
        '-c:a', 'aac', '-b:a', '320k', f'{output_path}'
    ]

    return subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)


ffmpeg_writer = generate_frame_renderer(input, output)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True


def convert(param):
    return {
        k.replace("module.", ""): v
        for k, v in param.items()
        if "module." in k
    }


ifnet = IFNet().cuda().eval()
ifnet.load_state_dict(convert(torch.load(r'weights\train_log_rife_422_lite\rife.pkl', map_location='cpu')), False)
model = Model()
model.load_model(r'weights\train_log_pg104', -1)
model.device()
model.eval()


def to_tensor(img):
    return torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float().cuda() / 255.


def load_image(img, _scale):
    h, w, _ = img.shape
    while h * _scale % 64 != 0:
        h += 1
    while w * _scale % 64 != 0:
        w += 1
    img = cv2.resize(img, (w, h))
    img = to_tensor(img)
    return img


def put(things):
    write_buffer.put(things)


def get():
    return read_buffer.get()


def build_read_buffer(r_buffer, v):
    ret, __x = v.read()
    while ret:
        r_buffer.put(cv2.resize(__x, global_size))
        ret, __x = v.read()
    r_buffer.put(None)


def clear_write_buffer(w_buffer):
    global ffmpeg_writer
    while True:
        item = w_buffer.get()
        if item is None:
            break
        result = cv2.resize(item, global_size)
        ffmpeg_writer.stdin.write(np.ascontiguousarray(result[:, :, ::-1]))
    ffmpeg_writer.stdin.close()
    ffmpeg_writer.wait()


@torch.autocast(device_type="cuda")
def make_inference(_I0, _I1, _I2, _scale):
    # Flow distance calculator
    def distance_calculator(_x):
        u, v = _x[:, 0:1], _x[:, 1:]
        return torch.sqrt(u ** 2 + v ** 2)

    reuse_i1i0 = model.reuse(_I1, _I0, scale)
    reuse_i1i2 = model.reuse(_I1, _I2, scale)

    flow10, metric10 = reuse_i1i0[0], reuse_i1i0[2]
    flow12, metric12 = reuse_i1i2[0], reuse_i1i0[2]

    # Compute the distance using the optical flow and distance calculator
    d10 = distance_calculator(flow10) + 1e-4
    d12 = distance_calculator(flow12) + 1e-4

    # Calculate the distance ratio map
    drm10 = d10 / (d10 + d12)
    drm12 = d12 / (d10 + d12)

    # The distance ratio map (drm) is initially aligned with I1.
    # To align it with I0 and I2, we need to warp the drm maps.
    # Note: To reverse the direction of the drm map, use 1 - drm and then warp it.
    drm01 = warp(1 - drm10, flow10, metric10, strMode='soft')
    drm21 = warp(1 - drm12, flow12, metric12, strMode='soft')

    # Create a mask with all ones to identify the holes in the warped drm maps
    ones_mask = torch.ones_like(drm01, device=drm01.device)

    # Warp the ones mask
    warped_ones_mask01 = warp(ones_mask, flow10, metric10, strMode='soft')
    warped_ones_mask21 = warp(ones_mask, flow12, metric12, strMode='soft')

    # Identify holes in warped drm map
    holes01 = warped_ones_mask01 < 0.999
    holes21 = warped_ones_mask21 < 0.999

    # Fill the holes in the warped drm maps with the inverse of the original drm maps
    drm01[holes01] = (1 - drm10)[holes01]
    drm21[holes21] = (1 - drm12)[holes21]

    def calc_drm_rife(_t):
        # For RIFE, drm should be aligned with the time corresponding to the intermediate frame.
        drm01r = warp(1 - drm10, flow10 * ((1 - drm10) * 2) * _t, metric10, strMode='soft')
        drm21r = warp(1 - drm12, flow12 * ((1 - drm12) * 2) * _t, metric12, strMode='soft')

        warped_ones_mask01r = warp(ones_mask, flow10 * ((1 - drm01r) * 2) * _t, metric10, strMode='soft')
        warped_ones_mask21r = warp(ones_mask, flow12 * ((1 - drm21r) * 2) * _t, metric12, strMode='soft')

        holes01r = warped_ones_mask01r < 0.999
        holes21r = warped_ones_mask21r < 0.999

        drm01r[holes01r] = (1 - drm10)[holes01r]
        drm21r[holes21r] = (1 - drm12)[holes21r]

        return drm01r, drm21r

    f_I0, f_I1, f_I2 = map(lambda x: torch.nn.functional.interpolate(x, scale_factor=0.5, mode='bilinear',
                                                                     align_corners=False), [_I0, _I1, _I2])

    output1, output2 = list(), list()
    _output = list()
    if times % 2:
        for i in range((times - 1) // 2):
            t = (i + 1) / times
            # Adjust timestep parameters for interpolation between frames I0, I1, and I2
            # The drm values range from [0, 1], so scale the timestep values for interpolation between I0 and I1 by a factor of 2

            drm01r, drm21r = calc_drm_rife(t)
            I10 = ifnet(torch.cat((f_I1, f_I0), 1), timestep=t * (2 * drm01r),
                        scale_list=[8 / scale, 4 / scale, 2 / scale, 1 / scale])
            I12 = ifnet(torch.cat((f_I1, f_I2), 1), timestep=t * (2 * drm21r),
                        scale_list=[8 / scale, 4 / scale, 2 / scale, 1 / scale])

            output1.append(model.inference_t2(_I1, _I0, reuse_i1i0, timestep0=t * (2 * (1 - drm10)),
                                              timestep1=1 - t * (2 * drm01), rife=I10))
            output2.append(model.inference_t2(_I1, _I2, reuse_i1i2, timestep0=t * (2 * (1 - drm12)),
                                              timestep1=1 - t * (2 * drm21), rife=I12))
        _output = list(reversed(output1)) + [_I1] + output2
    else:
        for i in range(times // 2):
            t = (i + 0.5) / times

            drm01r, drm21r = calc_drm_rife(t)
            I10 = ifnet(torch.cat((f_I1, f_I0), 1), timestep=(t * 2 * drm01r),
                        scale_list=[8 / scale, 4 / scale, 2 / scale, 1 / scale])
            I12 = ifnet(torch.cat((f_I1, f_I2), 1), timestep=(t * 2 * drm21r),
                        scale_list=[8 / scale, 4 / scale, 2 / scale, 1 / scale])

            output1.append(model.inference_t2(_I1, _I0, reuse_i1i0, timestep0=(t * 2) * (1 - drm10),
                                              timestep1=1 - (t * 2) * drm01, rife=I10))
            output2.append(model.inference_t2(_I1, _I2, reuse_i1i2, timestep0=(t * 2) * (1 - drm12),
                                              timestep1=1 - (t * 2) * drm21, rife=I12))
        _output = list(reversed(output1)) + output2

    _output = map(lambda x: (x[0].cpu().float().numpy().transpose(1, 2, 0) * 255.).astype(np.uint8), _output)

    return _output


video_capture = cv2.VideoCapture(input)
total_frames_count = video_capture.get(7)
pbar = tqdm(total=total_frames_count)
read_buffer = Queue(maxsize=100)
write_buffer = Queue(maxsize=-1)
_thread.start_new_thread(build_read_buffer, (read_buffer, video_capture))
_thread.start_new_thread(clear_write_buffer, (write_buffer,))

# start inference
i0, i1 = get(), get()
I0, I1 = load_image(i0, scale), load_image(i1, scale)

# head
output = make_inference(I0, I0, I1, scale)
for x in output:
    put(x)
pbar.update(1)

while True:
    i2 = get()
    if i2 is None:
        break
    I2 = load_image(i2, scale)

    output = make_inference(I0, I1, I2, scale)
    for x in output:
        put(x)

    i0, i1 = i1, i2
    I0, I1 = I1, I2
    pbar.update(1)

# tail(At the end, i0 and i1 have moved to the positions of index -2 and -1 frames.)
output = make_inference(I0, I1, I1, scale)
for x in output:
    put(x)
pbar.update(1)

# wait for output
while not write_buffer.empty():
    time.sleep(1)
pbar.close()
