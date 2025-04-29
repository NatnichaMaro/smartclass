import cv2
from ultralytics import YOLO
from datetime import datetime
from zoneinfo import ZoneInfo
import firebase_admin
from firebase_admin import credentials, firestore, storage
import subprocess
import os

# ===== Config =====
timestamp = datetime.now(ZoneInfo("Asia/Bangkok")).strftime("%Y-%m-%d_%H%M%S")
VIDEO_PATH = "latest_video.mp4"
MODEL_PATH = "yolov8n.pt"
OUTPUT_FILENAME = f"{timestamp}.mp4"
OUTPUT_PATH = OUTPUT_FILENAME
CONF_THRESHOLD = 0.5
PROCESS_EVERY = 1
FIREBASE_CRED_PATH = "smart-class-e9661-firebase-adminsdk-fbsvc-f287bdab03.json"
FIREBASE_BUCKET = "smart-class-e9661.firebasestorage.app"

# ===== Firebase Init =====
print("Initializing Firebase...")
cred = credentials.Certificate(FIREBASE_CRED_PATH)
firebase_admin.initialize_app(cred, {'storageBucket': FIREBASE_BUCKET})
db = firestore.client()
bucket = storage.bucket()
print("Firebase initialized.")

# ===== Reset total_count to 0 every midnight =====
now = datetime.now(ZoneInfo("Asia/Bangkok"))
current_time = now.time()

if current_time.hour == 0 and current_time.minute == 0:
    video_date = now.strftime("%Y-%m-%d")
    print("⏰ Midnight reached! Resetting total_count in people_counter to 0.")
    people_counter_ref = db.collection("people_counter")
    for doc in people_counter_ref.stream():
        people_counter_ref.document(doc.id).update({"total_count": 0})
    print("✅ total_count in people_counter has been reset to 0 for today.")

# ===== Find New Video File =====
print("Looking for new video file...")
used_video_names = set(doc.to_dict().get("video_name") for doc in db.collection("people_counter").stream())
blobs = list(bucket.list_blobs(prefix="videos/"))
new_blob = None
for blob in sorted(blobs, key=lambda b: b.updated):
    if blob.name.endswith(".mp4") and blob.name not in used_video_names:
        new_blob = blob
        break

if not new_blob:
    print("❌ No new video found to process.")
    exit()

print(f"✅ New video found: {new_blob.name}")
new_blob.download_to_filename(VIDEO_PATH)

# ===== Load Model =====
print("Loading YOLOv8 model...")
model = YOLO(MODEL_PATH)
print("Model loaded.")

# ===== Video Setup =====
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")
ret, frame = cap.read()
if not ret:
    raise RuntimeError("Cannot read video frames")
frame_h, frame_w = frame.shape[:2]
fps = cap.get(cv2.CAP_PROP_FPS) or 30
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (frame_w, frame_h))
print(f"Writing output to {OUTPUT_PATH}: {frame_w}x{frame_h} @ {fps:.1f} FPS")

# ===== Line Position =====
LINE_START = (0, frame_h - 50)
LINE_END   = (frame_w - 1550, frame_h - 300)
line_y = (LINE_START[1] + LINE_END[1]) / 2

prev_positions = {}
crossed_ids = {}
in_count = 0
out_count = 0
frame_idx = 0
total_count = 0
initial_people = 0
prev_positions = {}
possible_exits = {}
counted_ids_in = set()
counted_ids_out = set()
last_seen = {}
max_disappear_frame = 30
track_status = {}

print(f"Counting line: {LINE_START} -> {LINE_END}, line_y={line_y}")
print("Starting processing...")

# ===== Load previous total count (if any) =====
latest_doc = db.collection("people_counter").order_by("timestamp", direction=firestore.Query.DESCENDING).limit(1).stream()
last_total = 0
for doc in latest_doc:
    data = doc.to_dict()
    last_total = data.get("total_count", 0)

detected_first = False

# ===== Frame Loop =====
while True:
    ret, frame = cap.read()
    if not ret:
        break
  
    frame_idx += 1
    if frame_idx % PROCESS_EVERY != 0:
        continue
    results = model.track(frame, persist=True, classes=[0], conf=CONF_THRESHOLD, verbose=False)[0]
    boxes = results.boxes.xyxy.cpu().numpy()
    

    if results.boxes.id is None:
        out.write(frame)
        continue

    ids = results.boxes.id.cpu().numpy()
    
    print(f"\nFrame {frame_idx}: Detected IDs {ids.tolist()}")

    # if not detected_first and last_total == 0:
    #     initial_people = len(ids)
    #     detected_first = True
    #     print(f"🎯 Initial people detected: {initial_people}")
    # if not initial_people_counted and frame_idx == 1:
    #     predict_results = model.predict(frame, conf=0.2, classes=[0])[0]
    #     people_boxes = predict_results.boxes
    #     initial_people = sum(1 for box in people_boxes if int(box.cls) == 0)
    #     total_count += initial_people
    #     print(f"👀 คนในห้องตั้งแต่ต้น: {initial_people}")
    #     initial_people_counted = True


    # ตรวจว่าไม่มี ID เลย
    if len(ids) == 0:
        out.write(frame)
        continue

    current_ids = set()
    for box, tid in zip(boxes, ids):
        x1, y1, x2, y2 = map(int, box)
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        if tid not in prev_positions:
            prev_positions[tid] = (cx, cy)
            track_status[tid] = None
        else:
            px, py = prev_positions[tid]

            if py >= line_y > cy and track_status[tid] != "in":
                in_count += 1
                total_count += 1
                track_status[tid] = "in"
                print(f"✅ ID {tid} เข้า (in_count={in_count})")

            elif py < line_y <= cy and track_status[tid] == "in":
                out_count += 1
                total_count -= 1
                track_status[tid] = "out"
                print(f"⬅️ ID {tid} ออก (out_count={out_count})")

            prev_positions[tid] = (cx, cy)

        last_seen[tid] = frame_idx

        # ตรวจหา track ที่หายไปเกินระยะเวลา
        disappeared_ids = [
            tid for tid, last_frame in last_seen.items()
            if frame_idx - last_frame > max_disappear_frame
        ]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 87, 212), 2)
        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

        for tid, (cx, cy) in list(possible_exits.items()):
            if tid not in current_ids and cy > line_y and tid not in counted_ids_out:
                out_count += 1
                counted_ids_out.add(tid)
                possible_exits.pop(tid)

    total_count = initial_people + in_count - out_count + last_total

    # cv2.line(frame, LINE_START, LINE_END, (250, 192, 23), 6)
    # cv2.putText(frame, f"In: {in_count}", (1600, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    # cv2.putText(frame, f"Out: {out_count}", (1600, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f"Total_count: {total_count}", (1600, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    out.write(frame)

cap.release()
out.release()
print(f"Finished. Total In={in_count}, Out={out_count}, Total_count={total_count}")

# ===== Convert to H.264 MP4 =====
converted_filename = f"Room901_{OUTPUT_FILENAME}"
subprocess.run([
    "ffmpeg", "-i", OUTPUT_FILENAME, "-c:v", "libx264",
    "-preset", "fast", "-movflags", "+faststart", "-y", converted_filename
])

# ===== Upload to Firebase =====
time_now = datetime.now(ZoneInfo("Asia/Bangkok"))
blob = bucket.blob(f"counter_videos/{converted_filename}")
blob.upload_from_filename(converted_filename, content_type='video/mp4')
blob.make_public()
video_url = blob.public_url
print(f"Uploaded to Firebase: {video_url}")

# ===== Save Metadata to Firestore =====
db.collection("people_counter").document(timestamp).set({
    "timestamp": time_now.isoformat(),
    "in": in_count,
    "out": out_count,
    "total_count": total_count,
    "video_name": new_blob.name,
    "video_url": video_url
})
print("Uploaded counts to Firestore.")

# ===== Cleanup =====
for f in [VIDEO_PATH, OUTPUT_FILENAME, converted_filename]:
    if os.path.exists(f):
        os.remove(f)
        print(f"Removed file: {f}")
