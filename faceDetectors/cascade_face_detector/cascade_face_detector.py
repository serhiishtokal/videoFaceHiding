import cv2


def get_faces(image, face_cascade, scale_factor=1.3, min_neighbors=3):
    gray_image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray_image,
        scale_factor,
        min_neighbors,
        # minSize=(30,30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    return faces
