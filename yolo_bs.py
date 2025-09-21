# yolo_deploy_min.py
import os, sys, argparse, glob
import cv2
from ultralytics import YOLO

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='/Users/xyz/Documents/yolo.zip/my_model/train_results/weights/best.pt',
                    help='path to best.pt (default set to your path)')
    ap.add_argument('--source', required=True,
                    help='image | folder | /path/video.mp4 | usb0 | picamera0')
    ap.add_argument('--conf', type=float, default=0.5, help='confidence threshold')
    ap.add_argument('--resolution', default=None, help='WxH display size, e.g. 640x480')
    ap.add_argument('--save_video', default='', help='optional output video path, e.g. out.avi')
    return ap.parse_args()

def main():
    args = parse_args()

    if not os.path.exists(args.model):
        print('ERROR: model path not found:', args.model)
        sys.exit(1)

    model = YOLO(args.model, task='detect')
    labels = model.names

    img_exts = {'.jpg','.jpeg','.png','.bmp','.JPG','.JPEG','.PNG','.BMP'}
    vid_exts = {'.avi','.mov','.mp4','.mkv','.wmv'}
    src = args.source

    if os.path.isdir(src):
        source_type = 'folder'
        img_list = [f for f in sorted(glob.glob(os.path.join(src, '*')))
                    if os.path.splitext(f)[1] in img_exts]
    elif os.path.isfile(src):
        ext = os.path.splitext(src)[1]
        if ext in img_exts:
            source_type = 'image'
            img_list = [src]
        elif ext in vid_exts:
            source_type = 'video'
        else:
            print('Unsupported file extension:', ext); sys.exit(1)
    elif src.startswith('usb'):
        source_type = 'usb'
        usb_idx = int(src[3:])
    elif src.startswith('picamera'):
        source_type = 'picamera'
    else:
        print('Invalid source:', src); sys.exit(1)

    resize = args.resolution is not None
    if resize:
        resW, resH = map(int, args.resolution.split('x'))

    writer = None  # will lazily open if --save_video provided

    def draw(frame, r):
        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        clses = r.boxes.cls.cpu().numpy()
        for (x1,y1,x2,y2), c, k in zip(boxes, confs, clses):
            if c < args.conf:
                continue
            name = labels[int(k)]
            p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
            color = (0,255,0)
            cv2.rectangle(frame, p1, p2, color, 2)
            cv2.putText(frame, f'{name} {int(c*100)}%', (p1[0], max(0, p1[1]-10]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    def infer_and_show(frame):
        nonlocal writer
        if resize:
            frame = cv2.resize(frame, (resW, resH))
        r = model(frame, verbose=False)[0]
        draw(frame, r)
        cv2.imshow('YOLO Deploy', frame)

        if args.save_video:
            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(args.save_video, fourcc, 30, (w, h))
            writer.write(frame)

    if source_type in ('image', 'folder'):
        for path in img_list:
            img = cv2.imread(path)
            if img is None: continue
            infer_and_show(img)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break

    elif source_type == 'video':
        cap = cv2.VideoCapture(src)
        if resize:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, resW)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resH)
        while True:
            ok, frame = cap.read()
            if not ok: break
            infer_and_show(frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()

    elif source_type == 'usb':
        cap = cv2.VideoCapture(usb_idx)
        if resize:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, resW)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resH)
        while True:
            ok, frame = cap.read()
            if not ok: break
            infer_and_show(frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()

    elif source_type == 'picamera':
        from picamera2 import Picamera2
        size = (resW, resH) if resize else (640, 480)
        picam = Picamera2()
        picam.configure(picam.create_video_configuration(main={"format": 'RGB888', "size": size}))
        picam.start()
        try:
            while True:
                frame = picam.capture_array()
                infer_and_show(frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            picam.stop()

    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
