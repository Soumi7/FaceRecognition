import cv2

def draw_boundary(img,classifier, scaleFactor, minNeighbors, color, text, clf):
	gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	features = classifier.detectMultiScale(gray_img,scaleFactor, minNeighbors)
	coords = []
	for (x,y,w,h) in features:
		cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
		id, _ =clf.predict(gray_img[y:y+h,x:x+w])
		if id==1:
			cv2.putText(img,"soumi", (x, (y-4)), cv2.FONT_HERSHEY_SIMPLEX,0.8,color,1,cv2.LINE_AA)
		if id==2:
			cv2.putText(img,"durga", (x, (y-4)), cv2.FONT_HERSHEY_SIMPLEX,0.8,color,1,cv2.LINE_AA)
		
		coords = [x, y, w, h]
	return coords, img

def recognize(img, clf, facecascade):
	color={"blue":(255,0,0),"green":(0,255,0),"red":(0,0,255),"white":(255,255,255)}
	coords,img = draw_boundary(img, faceCascade,1.1,10,color["blue"],'Face',clf)
	return img



faceCascade= cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

clf= cv2.face.LBPHFaceRecognizer_create()
clf.read('classifier.yml')

img_id=0

video_capture=cv2.VideoCapture(0)
while True:
    _,img = video_capture.read()
    #img= detect(img, faceCascade, img_id)
    img= recognize(img, clf, faceCascade)
    cv2.imshow("face detection",img)
    img_id+=1
    if cv2.waitKey(1) & 0xFF==ord('q'):
    	break

video_capture.release()
cv2.destroyAllWindows()