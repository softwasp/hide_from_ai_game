import cv2
import time, threading, pygame
import numpy as np
CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4
STOP_THR = 0
def play_video(situatiakrch):
    pygame.init()
    pygame.mixer.init()


    if situatiakrch == "2":
        pygame.mixer.music.load("b.mp3")
        video_path = "b.mp4"
    elif situatiakrch == "1":
        pygame.mixer.music.load("a.mp3")
        video_path = "a.mp4"
    pygame.mixer.music.play()
    cap = cv2.VideoCapture(video_path)
    while True:
        if situatiakrch == "1":
            global STOP_THR
            if STOP_THR == 1:
                break
        else:
            pass
        ret, frame = cap.read()
        if not ret:
            # Video has ended, go back to the beginning of the video
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        cv2.imshow("game", frame)
        if situatiakrch == "2":
            pass
            #time.sleep(5)
            # Press 'q' key to exit the loop
        if cv2.waitKey(1) == ord("q"):
            pygame.mixer.music.stop()
            break


    cap.release()
    cv2.destroyAllWindows()
    pygame.mixer.music.stop()
net = cv2.dnn.readNetFromDarknet("Resources/yolov4-tiny.cfg", "Resources/yolov4-tiny.weights")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
def apply_yolo_object_detection(image_to_process, idco):
    height, width, _ = image_to_process.shape
    blob = cv2.dnn.blobFromImage(image_to_process, scalefactor=1/255, size=(416, 416), mean=(0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(output_layers)
    class_ids = []
    confidences = []
    boxes = []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > CONFIDENCE_THRESHOLD and class_id in classes_to_look_for:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = center_x - w // 2
                y = center_y - h // 2
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

    for index in indices:
        idco = idco + 1
        box = boxes[index]
        x, y, w, h = box
        label = classes[class_ids[index]]
        color = (255, 255, 0)
        cv2.rectangle(image_to_process, (x, y), (x + w, y + h), color, thickness=2)
        cv2.putText(
            image_to_process,
            f"{label}:{idco}: {str(int(confidences[index]*100))}",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=color,
            thickness=2,
        )
        print(label)
        if label == "person":
            # Создаем и запускаем новый поток
            global STOP_THR
            STOP_THR = 1
            thread3 = threading.Thread(target=play_video, args=("2",))
            thread3.start()
            #time.sleep(1000)
            #play_video("b.mp4")

        #try:
            #SV.say("Human")
            #SV.runAndWait()
            #time.sleep(0.25)
            #SV.say(str(int(int(y)/34))+" meters")
            #SV.runAndWait()
            #print(str(int(int(y)/34)))
            #time.sleep(5)

        #except:
        #    pass

    objects_count = len(indices)
    cv2.putText(
        image_to_process,
        f"Objects found: {objects_count}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1,
        color=(255, 255, 255),
        thickness=2,
    )
    return image_to_process
def start_image_object_detection(idco):
    try:
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)

            image = apply_yolo_object_detection(frame, idco)
            #play_video("a.mp4")
            cv2.imshow("Image", image)
            if cv2.waitKey(1) == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()
    except KeyboardInterrupt:
        pass
if __name__ == "__main__":
    with open("Resources/coco.names.txt") as file:
        classes = [line.strip() for line in file.readlines()]
    #look_for = "person".split(",")
    #classes_to_look_for = [classes.index(item.strip()) for item in look_for]
    classes_to_look_for = list(range(len(classes)))
    idco = 0
    thread = threading.Thread(target=play_video, args=("1", ))
    thread2 = threading.Thread(target=start_image_object_detection, args=(idco, ))
    thread.start()
    thread2.start()
    thread.join()
    thread2.join()
