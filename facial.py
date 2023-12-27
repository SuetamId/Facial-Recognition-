import cv2
import face_recognition

# Carregando a imagem de referência
reference_image_path = "./img/mateus.jpg"
reference_image = face_recognition.load_image_file(reference_image_path)
reference_encoding = face_recognition.face_encodings(reference_image)[0]

# Inicializando a webcam
webcam = cv2.VideoCapture(0)

while webcam.isOpened():
    validacao, frame = webcam.read()  # Lê a imagem da webcam
    if not validacao:
        break

    # Encontrando todas as faces no quadro
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Comparando com a imagem de referência
        matches = face_recognition.compare_faces([reference_encoding], face_encoding)
        name = "Desconhecido"

        if True in matches:
            name = "Mateus"

        # Desenhando um retângulo ao redor da face e exibindo o nome
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    # Exibindo o quadro resultante
    cv2.imshow('Reconhecimento Facial', frame)

    if cv2.waitKey(5) == 27:  # ESC
        break

webcam.release()
cv2.destroyAllWindows()