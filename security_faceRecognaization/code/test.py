import cv2
import numpy as np
import face_recognition
import os
import matplotlib.pyplot as plt

def find_encodings(images):
    encode_list = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encode_list.append(encode)
    return encode_list

def calculate_eer(roc_curve, thresholds):
    eer = None
    eer_threshold = None
    min_difference = float('inf')

    for fmr, fnmr in roc_curve:
        difference = abs(fmr - fnmr)
        if difference < min_difference:
            min_difference = difference
            eer = (fmr + fnmr) / 2
            eer_threshold = thresholds[roc_curve.index((fmr, fnmr))]

    return eer, eer_threshold

def main(thresholds):
    training_path = 'persons/training'
    test_directory = 'persons/test'

    images = []
    class_names = []
    for cl in os.listdir(training_path):
        img = cv2.imread(os.path.join(training_path, cl))
        images.append(img)
        class_names.append(os.path.splitext(cl)[0])

    encode_list_known = find_encodings(images)

    print('Encoding Complete.')

    test_paths = [os.path.join(test_directory, img) for img in os.listdir(test_directory)]

    roc_curve = []

    for threshold in thresholds:
        fnmr = 0
        fmr = 0
        count_i_equals_j = 0
        count_i_not_equals_j = 0
        face_distances_list = []

        for image_path in test_paths:
            img = cv2.imread(image_path)
            img_s = cv2.resize(img, (0, 0), None, 0.25, 0.25)
            img_s = cv2.cvtColor(img_s, cv2.COLOR_BGR2RGB)

            face_current_frame = face_recognition.face_locations(img_s)
            encode_current_frame = face_recognition.face_encodings(img_s, face_current_frame)

            face_dis_list_current = []

            for encode_face, face_loc in zip(encode_current_frame, face_current_frame):
                matches = face_recognition.compare_faces(encode_list_known, encode_face, tolerance=0.6)
                face_dis = face_recognition.face_distance(encode_list_known, encode_face)

                if matches:
                    match_index = np.argmin(face_dis)
                    name = class_names[match_index].upper()
                else:
                    name = "unknown"

                print("Name:", name)  # Debug output

                face_dis_list_current.append(face_dis.tolist())

            face_distances_list.extend(face_dis_list_current)

        face_distances_array = np.array(face_distances_list)

        for i in range(len(face_distances_array)):
            for j in range(len(face_distances_array[i])):
                if i == j:
                    count_i_equals_j += 1
                    if face_distances_array[i][j] > threshold:
                        fnmr += 1
                else:
                    count_i_not_equals_j += 1
                    if face_distances_array[i][j] <= threshold:
                        fmr += 1

        fnmr_rate = fnmr / count_i_equals_j
        fmr_rate = fmr / count_i_not_equals_j

        roc_curve.append((fmr_rate, fnmr_rate))

    print("ROC Curve Points:", roc_curve)  # Debug output

    fmr_rates, fnmr_rates = zip(*roc_curve)
    plt.plot(fmr_rates, fnmr_rates)
    plt.xlabel('False Match Rate (FMR)')
    plt.ylabel('False Non-Match Rate (FNMR)')
    plt.title('ROC Curve')
    plt.grid(True)
    plt.show()

    eer, eer_threshold = calculate_eer(roc_curve, thresholds)

    print("Equal Error Rate (EER):", eer)
    print("EER Threshold:", eer_threshold)

if __name__ == "__main__":
    thresholds = [ 0.76926607, 0.6560450775876246,0.53701562, 0.1]  # Define the thresholds you want to test
    main(thresholds)
