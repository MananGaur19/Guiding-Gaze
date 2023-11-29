import argparse
from asyncio.windows_events import NULL
import time
from pathlib import Path
from scipy.interpolate import RectBivariateSpline
from PIL import Image
import matplotlib.pyplot as plt 
from playsound import playsound

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy
from tensorflow.keras.models import Model

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
	scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
""" Uncomment for facial and scene recognision -- warning will run slow on CPU
import numpy as np
#from keras.engine import Model
from keras.layers import Input
from keras_vggface.vggface import VGGFace
from keras.preprocessing import image
from keras_vggface import utils
prev=NULL
path="C:/Users/tejes/Downloads/github_upload_final/github_upload_final/yolov7/sounds/"
mo=Model()
#mo=mo.load_weights('model.h5')
vggface = VGGFace(model='vgg16')
vgg_features = VGGFace(include_top=False, input_shape=(224, 224, 3), pooling='max')
modelF = VGGFace()
"""
def depth_to_distance(depth_value,depth_scale):
	return 1.0 / (depth_value*depth_scale)

def apply_ema_filter(current_depth):
	global previous_depth
	filtered_depth = alpha * current_depth + (1 - alpha) * previous_depth
	previous_depth = filtered_depth  
	return filtered_depth

alpha = 0.2
previous_depth = 0.0
depth_scale = 1.0

midas = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')
midas.to('cpu')
midas.eval()
transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
transform = transforms.small_transform 

def detect(save_img=False):
	k = 0
	source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
	save_img = not opt.nosave and not source.endswith('.txt')  
	webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
		('rtsp://', 'rtmp://', 'http://', 'https://'))

	save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  
	(save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  

	set_logging()
	frame_bounding_box_data = []
	device = select_device(opt.device)
	half = device.type != 'cpu'  

	
	model = attempt_load(weights, map_location=device)  
	stride = int(model.stride.max())  
	imgsz = check_img_size(imgsz, s=stride)  

	if trace:
		model = TracedModel(model, device, opt.img_size)

	if half:
		model.half()  

	
	classify = False
	if classify:
		modelc = load_classifier(name='resnet101', n=2)  
		modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

	
	vid_path, vid_writer = None, None
	if webcam:
		view_img = check_imshow()
		cudnn.benchmark = True  
		dataset = LoadStreams(source, img_size=imgsz, stride=stride)
	else:
		dataset = LoadImages(source, img_size=imgsz, stride=stride)

	
	names = model.module.names if hasattr(model, 'module') else model.names
	colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

	
	if device.type != 'cpu':
		model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  
	old_img_w = old_img_h = imgsz
	old_img_b = 1

	t0 = time.time()
	for path, img, im0s, vid_cap in dataset:
		numerical_values_2d = []
		img = torch.from_numpy(img).to(device)
		img = img.half() if half else img.float()  
		img /= 255.0  
		if img.ndimension() == 3:
			img = img.unsqueeze(0)

		
		if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
			old_img_b = img.shape[0]
			old_img_h = img.shape[2]
			old_img_w = img.shape[3]
			for i in range(3):
				model(img, augment=opt.augment)[0]

		
		t1 = time_synchronized()
		with torch.no_grad():   
			pred = model(img, augment=opt.augment)[0]
		t2 = time_synchronized()

		
		pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
		t3 = time_synchronized()

		
		if classify:
			pred = apply_classifier(pred, modelc, img, im0s)

		
		for i, det in enumerate(pred):  
			if webcam:  
				p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
			else:
				p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

			p = Path(p)  # to Path
			save_path = str(save_dir / p.name)  # img.jpg
			txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
			gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
			if len(det):
				# Rescale boxes from img_size to im0 size
				det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

				# Print results
				for c in det[:, -1].unique():
					n = (det[:, -1] == c).sum()  # detections per class
					s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
			
			img = numpy.array(im0)
			imgbatch = transform(img).to(device)
			with torch.no_grad(): 
				prediction = midas(imgbatch)
				prediction = torch.nn.functional.interpolate(
					prediction.unsqueeze(1),
					size = img.shape[:2], 
					mode='bicubic', 
					align_corners=False
				).squeeze()
				output = prediction.cpu().numpy()
			output_norm = cv2.normalize(output, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
			plt.imshow(output_norm)
			h,w = output_norm.shape
			x_grid = numpy.arange(w)
			y_grid = numpy.arange(h)
			spline = RectBivariateSpline(y_grid, x_grid, output_norm)
			depth_scale = 1
				
			class_ids = [0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 12, 13, 15, 16, 19, 56, 57, 59, 60, 61, 62, 71, 72, 75]
			# Write results
			for *xyxy, conf, cls in reversed(det):
				if int(cls) in class_ids:
					if conf >= 0.5:
					# Store bounding box data
						bbox_data = {'class': int(cls),
							'bbox': list(xyxy),
							'confidence': conf.item()}
						frame_bounding_box_data.append(bbox_data)
						x1 = (bbox_data['bbox'][0]+ bbox_data['bbox'][2])/2
						y1 = (bbox_data['bbox'][1]+ bbox_data['bbox'][3])/2
						x_min = bbox_data['bbox'][0]
						x_max = bbox_data['bbox'][2]
						x_center = (x_min+x_max)/2
						depth_mid_filt = spline(y1,x1) #y, x
						depth_midas = depth_to_distance(depth_mid_filt, depth_scale)
						depth_mid_filt = (apply_ema_filter(depth_midas)/10)[0][0]

						numerical_values_2d.append({'class': bbox_data['class'], 'x_center': x1, 'y_center': y1, 'confidence': bbox_data['confidence'], 'depth': depth_midas[0][0], 'x_min': x_min, 'x_max': x_max, 'x_center': x_center})

						if save_txt:  
							xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  
							line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  
							with open(txt_path + '.txt', 'a') as f:
								f.write(('%g ' * len(line)).rstrip() % line + '\n')
						if save_img or view_img:  
							label = f'{names[int(cls)]} {conf:.2f}'
							plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)
		
					

		

			if k%10 == 0:
					d = 2.3
					l = 0
					c = 0
					r = 0
					numerical_values_2d.sort(key=lambda x: x['depth'])
					top_objects = [obj for obj in numerical_values_2d if obj['depth'] < d][:5]
					print("top objects: ", top_objects)
					for obj in top_objects:
						x_min = obj['x_min']
						x_max = obj['x_max']
						x_center = obj['x_center']
						if 2*im0.shape[1]/3 <= x_min or 2*im0.shape[1]/3 <= x_max:
							r += 1
						if im0.shape[1]/3 <= x_min < 2*im0.shape[1]/3 or im0.shape[1]/3 <= x_max <2*im0.shape[1]/3:
							c += 1
						if 0 <= x_min < im0.shape[1]/3 or 0 <= x_max < im0.shape[1]/3:
							l += 1
						if im0.shape[1]/3 <= x_center < 2*im0.shape[1]/3:
							c+=1
					print(f"Left: {l}, Center: {c}, Right: {r}")
					print(numerical_values_2d)
					
					if c == 0:
						gray = cv2.cvtColor(im0, cv2.COLOR_BGR2GRAY)
						blurred = cv2.GaussianBlur(gray, (7, 7), 0)
						adaptive_threshold = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
						contours, _ = cv2.findContours(adaptive_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
						min_contour_area = 1000 
						contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
						total_area = im0.shape[0] * im0.shape[1]
						change_area = sum(cv2.contourArea(cnt) for cnt in contours)
						percentage = (change_area / total_area) * 100
						print(percentage)
						threshold_percentage = 97
						if percentage > threshold_percentage:
							playsound("C:/Users/tejes/Downloads/github_upload_final/github_upload_final/yolov7/sounds/w.mp3")
							print(f"Detected changes occupy more than {threshold_percentage}% of the screen. Possible wall detected.")
						else:
							playsound("C:/Users/tejes/Downloads/github_upload_final/github_upload_final/yolov7/sounds/c.mp3")
					elif l == 0:
						playsound("C:/Users/tejes/Downloads/github_upload_final/github_upload_final/yolov7/sounds/l.mp3")
					elif r == 0:
						playsound("C:/Users/tejes/Downloads/github_upload_final/github_upload_final/yolov7/sounds/r.mp3")
					else:

							playsound("C:/Users/tejes/Downloads/github_upload_final/github_upload_final/yolov7/sounds/s.mp3")
					
				  
			k = k+1 
			print("length: ", len(numerical_values_2d))
			print("k: ", k)
			print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')
			""" Uncomment for Scene and Facial -- warning will run slow on cpu
			if k%100==0:
				im0=numpy.array(im0)
				i=im0.reshape(-1,im0.shape[1],im0.shape[0],3)
				a=3
				#a=mo.predict(i)
				#a=numpy.argmax(a)
				if a!=prev or prev==NULL:
					if a==0:
						playsound(path+"Balcony.mp3")
					elif a==1:
						playsound(path+"Bathroom.mp3")
					elif a==2:
						playsound(path+"Bedroom.mp3")
					elif a==3:
						playsound(path+"Bridge.mp3")
					elif a==4:
						playsound(path+"garage.mp3")
					elif a==6:
						playsound(path+"Kitchen.mp3")
					elif a==7:
						playsound(path+"Lake.mp3")
					elif a==8:
						playsound(path+"market.mp3")
					elif a==5:
						playsound(path+"road.mp3")
					elif a==9:
						playsound(path+"site.mp3")
					elif a==11:
						playsound(path+"train.mp3")
					elif a==10:
						playsound(path+"Street1.mp3")
					prev=a
				x = image.img_to_array(i)
				x = np.expand_dims(x, axis=0)
				x = utils.preprocess_input(x, version=1) # or version=2
				preds = modelF.predict(x)
				print('Predicted:', utils.decode_predictions(preds))
				"""		
				
			# Stream results
			if view_img:
				cv2.imshow(str(p), im0)
				cv2.waitKey(1)  # 1 millisecond

			# Save results (image with detections)
			if save_img:
				if dataset.mode == 'image':
					cv2.imwrite(save_path, im0)
					print(f" The image with the result is saved in: {save_path}")
				else:  # 'video' or 'stream'
					if vid_path != save_path:  # new video
						vid_path = save_path
						if isinstance(vid_writer, cv2.VideoWriter):
							vid_writer.release()  # release previous video writer
						if vid_cap:  # video
							fps = vid_cap.get(cv2.CAP_PROP_FPS)
							w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
							h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
						else:  # stream
							fps, w, h = 30, im0.shape[1], im0.shape[0]
							save_path += '.mp4'
						vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
					vid_writer.write(im0)

	
	if save_txt or save_img:
		s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
		#print(f"Results saved to {save_dir}{s}")

	print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
	parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
	parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
	parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
	parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
	parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
	parser.add_argument('--view-img', action='store_true', help='display results')
	parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
	parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
	parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
	parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
	parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
	parser.add_argument('--augment', action='store_true', help='augmented inference')
	parser.add_argument('--update', action='store_true', help='update all models')
	parser.add_argument('--project', default='runs/detect', help='save results to project/name')
	parser.add_argument('--name', default='exp', help='save results to project/name')
	parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
	parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
	opt = parser.parse_args()
	print(opt)
	#check_requirements(exclude=('pycocotools', 'thop'))

	with torch.no_grad():
		if opt.update:  # update all models (to fix SourceChangeWarning)
			for opt.weights in ['yolov7.pt']:
				detect()
				strip_optimizer(opt.weights)
		else:
			detect()
