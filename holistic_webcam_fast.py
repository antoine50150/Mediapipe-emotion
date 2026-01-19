import os
import time
import cv2
import mediapipe as mp

# Réduit le bruit des logs
os.environ["GLOG_minloglevel"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

CAM_INDEX = 0

# Capture (souvent ignorée si la webcam impose ses modes, mais MJPG aide)
CAP_W, CAP_H = 640, 360
CAP_FPS = 30

# Downscale uniquement pour l'inférence (affichage garde la taille d'origine)
INFER_SCALE = 0.75  # 1.0 (max qualité), 0.75 (sweet spot), 0.6 (plus rapide)

# Inférer 1 frame sur N (énorme gain sans trop perdre en qualité)
PROCESS_EVERY_N = 2  # 1=chaque frame, 2=1/2, 3=1/3

# Holistic params
MODEL_COMPLEXITY = 1   # 0=plus rapide, 1=bon, 2=plus lent
REFINE_FACE = False    # True ralentit beaucoup
SMOOTH_LANDMARKS = True
MIN_DET = 0.5
MIN_TRACK = 0.5

# Face : contours uniquement (plus léger)
DRAW_FACE_CONTOURS = True


def main():
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_H)
    cap.set(cv2.CAP_PROP_FPS, CAP_FPS)

    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir la webcam (index={CAM_INDEX}).")

    real_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    real_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    real_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[CAP] real={real_w}x{real_h}@{real_fps:.1f} | infer_scale={INFER_SCALE} | every_n={PROCESS_EVERY_N}")

    prev_t = time.perf_counter()
    fps_smooth = 0.0
    frame_idx = 0

    # Derniers résultats réutilisés entre deux inférences
    last_results = None
    enable_face = True

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=MODEL_COMPLEXITY,
        smooth_landmarks=SMOOTH_LANDMARKS,
        refine_face_landmarks=REFINE_FACE,
        enable_segmentation=False,
        min_detection_confidence=MIN_DET,
        min_tracking_confidence=MIN_TRACK,
    ) as holistic:

        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            frame_bgr = cv2.flip(frame_bgr, 1)
            h0, w0 = frame_bgr.shape[:2]

            # Inference seulement 1 frame sur N
            do_infer = (frame_idx % PROCESS_EVERY_N == 0)

            if do_infer:
                # Downscale pour l'inférence
                if INFER_SCALE != 1.0:
                    infer_w = int(w0 * INFER_SCALE)
                    infer_h = int(h0 * INFER_SCALE)
                    infer_bgr = cv2.resize(frame_bgr, (infer_w, infer_h), interpolation=cv2.INTER_AREA)
                else:
                    infer_bgr = frame_bgr

                infer_rgb = cv2.cvtColor(infer_bgr, cv2.COLOR_BGR2RGB)
                infer_rgb.flags.writeable = False
                last_results = holistic.process(infer_rgb)
                infer_rgb.flags.writeable = True

            results = last_results  # réutilise entre 2 inférences

            # Dessin
            if results is not None:
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        frame_bgr,
                        results.pose_landmarks,
                        mp_holistic.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style(),
                    )

                if results.left_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame_bgr,
                        results.left_hand_landmarks,
                        mp_holistic.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style(),
                    )

                if results.right_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame_bgr,
                        results.right_hand_landmarks,
                        mp_holistic.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style(),
                    )

                if enable_face and results.face_landmarks:
                    if DRAW_FACE_CONTOURS:
                        mp_drawing.draw_landmarks(
                            frame_bgr,
                            results.face_landmarks,
                            mp_holistic.FACEMESH_CONTOURS,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_styles.get_default_face_mesh_contours_style(),
                        )
                    else:
                        mp_drawing.draw_landmarks(
                            frame_bgr,
                            results.face_landmarks,
                            mp_holistic.FACEMESH_TESSELATION,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_styles.get_default_face_mesh_tesselation_style(),
                        )

            # FPS (boucle)
            now = time.perf_counter()
            dt = now - prev_t
            prev_t = now
            inst_fps = 1.0 / max(dt, 1e-6)
            fps_smooth = inst_fps if fps_smooth == 0.0 else (0.9 * fps_smooth + 0.1 * inst_fps)

            cv2.putText(
                frame_bgr,
                f"FPS:{fps_smooth:.1f}  {w0}x{h0}  scale:{INFER_SCALE}  N:{PROCESS_EVERY_N}  face:{int(enable_face)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame_bgr,
                "ESC quit | f face | 1 N=1 | 2 N=2 | 3 N=3 | q scale=1.0 | w scale=0.75 | e scale=0.6",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("Holistic fast", frame_bgr)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            if key == ord("f"):
                enable_face = not enable_face

            # Changer N à la volée
            if key == ord("1"):
                globals()["PROCESS_EVERY_N"] = 1
            if key == ord("2"):
                globals()["PROCESS_EVERY_N"] = 2
            if key == ord("3"):
                globals()["PROCESS_EVERY_N"] = 3

            # Changer scale à la volée
            if key == ord("q"):
                globals()["INFER_SCALE"] = 1.0
            if key == ord("w"):
                globals()["INFER_SCALE"] = 0.75
            if key == ord("e"):
                globals()["INFER_SCALE"] = 0.6

            frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
