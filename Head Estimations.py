import cv2
import mediapipe as mp
import numpy as np
import time
def match_images(runtime_image, file_image_path):
    # Load the file image
    file_image = cv2.imread(file_image_path)

    # Convert both images to grayscale for simplicity
    runtime_gray = cv2.cvtColor(runtime_image, cv2.COLOR_BGR2GRAY)
    file_gray = cv2.cvtColor(file_image, cv2.COLOR_BGR2GRAY)

    # Compare the two images
    if np.array_equal(runtime_gray, file_gray):
        print("Images match")
    else:
        print("Images not match")
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
drawing_specs = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

cap = cv2.VideoCapture(0)

# Initialize a set to store unique movement conditions for all frames
all_frames_conditions = set()

while cap.isOpened():
    success, image = cap.read()

    start = time.time()

    # Flip the image horizontally for a letter selfie-view display
    # Also convert the color space from BGR to RGB
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

    # To improve performance
    image.flags.writeable = False

    # Get the result
    results = face_mesh.process(image)

    # To improve performance
    image.flags.writeable = True

    # Convert the color specs from RGB to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    img_h, img_w, img_c = image.shape
    face_3d = []
    face_2d = []

    # Initialize the set to store unique movement conditions for the current frame
    movement_conditions = set()

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                    if idx == 1:
                        nose_2d = (lm.x * img_w, lm.y * img_h)
                        nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

                    x, y = int(lm.x * img_w), int(lm.y * img_h)

                    # Get the 2D coordinates
                    face_2d.append([x, y])

                    # Get the 3D Coordinates
                    face_3d.append([x, y, lm.z])

            # Convert it into Numpy array
            face_2d = np.array(face_2d, dtype=np.float64)
            # Convert it into Numpy array
            face_3d = np.array(face_3d, dtype=np.float64)

            # the camera matrix
            focal_length = 1 * img_w
            cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                   [0, focal_length, img_w / 2],
                                   [0, 0, 1]])

            # the distance matrix
            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            # Solve Pnp
            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

            # Get rotational matrix
            rmat, jac = cv2.Rodrigues(rot_vec)

            # Get angles
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

            # Get the y rotation degree
            x = angles[0] * 360
            y = angles[1] * 360
            z = angles[2] * 360

            # Add the condition for head movement to the set for the current frame
            if y < -10:
                movement_conditions.add("Looking Left")
            elif y > 10:
                movement_conditions.add("Looking Right")
            elif x > -10:
                movement_conditions.add("Looking Down")
            elif x > 10:
                movement_conditions.add("Looking Up")
            else:
                movement_conditions.add("Forward")

            # Display the node direction
            nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

            p1 = (int(nose_2d[0]), int(nose_2d[1]))
            p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))

            cv2.line(image, p1, p2, (255, 0, 0), 3)

            # Add text on the image for the current frame
            cv2.putText(image, "Current Frame", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(image, "x: " + str(np.round(x, 2)), (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(image, "y: " + str(np.round(y, 2)), (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(image, "z: " + str(np.round(z, 2)), (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        end = time.time()
        totalTime = end - start

        if totalTime > 0:
            fps = 1 / totalTime
            print("FPS:", fps)
        else:
            print("Unable to calculate FPS. totalTime is zero.")

        cv2.putText(image, f'FPS: {int(fps)}', (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=drawing_specs,
            connection_drawing_spec=drawing_specs)

        # Add the unique movement conditions for the current frame to the set for all frames
        all_frames_conditions.add(frozenset(movement_conditions))

    # Display the unique movement conditions for all frames on the image
    frame_conditions_text = "Frame Conditions: " + ", ".join(str(cond) for cond in movement_conditions)
    cv2.putText(image, frame_conditions_text, (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow('Head Pose Estimation', image)

    if cv2.waitKey(5) & 0xFF == 27:
        runtime_image = image.copy()
        break
file_image_path = 'picture.png'
# Print the unique movement conditions for all frames
print("Unique movement conditions for all frames:")
for frame_conditions in all_frames_conditions:
    print(frame_conditions)

cap.release()
if len(all_frames_conditions) >= 3:
    print("Head estimation passed")
    match_images(runtime_image,file_image_path)
else:
    print("Head estimation failed")