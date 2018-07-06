import dlib
import cv2
import numpy as np
from imutils import face_utils

predictor_path = "./shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(predictor_path)
detector = dlib.get_frontal_face_detector()

# 使用する3Dモデルに合わせて変更する必要がある
model_points = np.array([
    (0.0, 0.0, 0.0),  # 鼻先
    (0.0, -330.0, -65.0),  # 顎
    (-225.0, 170.0, -135.0),  # 左目左端
    (225.0, 170.0, -135.0),  # 右目右端
    (-150.0, -150.0, -125.0),  # 口の左端
    (150.0, -150.0, -125.0)  # 口の右端
])


def face_shape_detecter_dlib(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    result = []
    points = [30, 8, 45, 36, 54, 48]  # 鼻先、顎、左目左端、右目右端、口の左端、口の右端の番号
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (127, 127, 0), (127, 0, 127), (0, 127, 127)]

    dets, scores, idx = detector.run(img_rgb, 0)
    if len(dets) > 0:
        for i, rect in enumerate(dets):
            shape = predictor(img_rgb, rect)
            shape = face_utils.shape_to_np(shape)
            clone = img.copy()
            # clone = np.zeros_like(img)

            for (x, y) in shape:
                cv2.circle(clone, (x, y), 2, (127, 127, 127), -1)

            c = 0
            for i in points:
                result.append(shape[i])
                cv2.putText(clone, str(i) + ": " + str(shape[i][0]) + ", " + str(shape[i][1]), (10, (c + 1) * 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors[c], 2)
                cv2.circle(clone, (shape[i][0], shape[i][1]), 3, colors[c], -1)
                c += 1
            # print(result)
        return result, clone
    else:
        return result, img


def solvePnP(model_points, img_points, camera_matrix):
    dist_coeffs = np.zeros((4, 1), np.double)  # レンズの歪みが無いと仮定
    success, rotation_vector, translation_vecter = cv2.solvePnP(model_points, img_points, camera_matrix, dist_coeffs)
    return rotation_vector, translation_vecter


def rodrigues(rotation_vector):
    rotation_matrix, jacobian = cv2.Rodrigues(rotation_vector)
    return rotation_matrix


def projectionMat(rotation_matrix):
    projMat = np.array(
        [[rotation_matrix[0][0], rotation_matrix[0][1], rotation_matrix[0][2], 0],
         [rotation_matrix[1][0], rotation_matrix[1][1], rotation_matrix[1][2], 0],
         [rotation_matrix[2][0], rotation_matrix[2][1], rotation_matrix[2][2], 0]]
    )
    return projMat


def decomposeProjectionMatrix(projMat):
    a = cv2.decomposeProjectionMatrix(projMat)
    return a[6]  # pitch, yaw, roll


def main():
    cap = cv2.VideoCapture(0)
    H = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    W = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    size = (H, W)

    # カメラ校正を適当に行う
    focal_length = size[1]
    center = (size[1] / 2, size[0] / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], np.double
    )

    while True:
        ret, frame = cap.read()
        face_list, dframe = face_shape_detecter_dlib(frame)

        if face_list is not None:
            try:
                img_points = np.array([
                    face_list[0],  # 鼻先
                    face_list[1],  # 顎
                    face_list[2],  # 左目左端
                    face_list[3],  # 右目右端
                    face_list[4],  # 口の左端
                    face_list[5]  # 口の右端
                ], np.double)
                # print(img_points)

                rotation_vector, translation_vector = solvePnP(model_points, img_points, camera_matrix)
                # print("Rotation Vector: \n {}".format(rotation_vector))
                # print("Translation Vecter \n {}".format(translation_vector))

                rotation_matrix = rodrigues(rotation_vector)
                projMat = projectionMat(rotation_matrix)
                pitch, yaw, roll = decomposeProjectionMatrix(projMat)
                print("\n--------------------\npitch: {}\nyaw  : {}\nroll : {}\n".format(pitch, yaw, roll))

                cv2.putText(dframe, "pitch : " + str(pitch), (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (127, 127, 127),
                            2)
                cv2.putText(dframe, "yaw : " + str(yaw), (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (127, 127, 127), 2)
                cv2.putText(dframe, "roll : " + str(roll), (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (127, 127, 127), 2)

            except:
                print("例外")  # トラッキングが外れた感じですな

        try:
            cv2.imshow("img", dframe)
        except TypeError:
            cv2.imshow("img", frame)

        c = cv2.waitKey(1)
        if c == 27:  # ESCキー
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
