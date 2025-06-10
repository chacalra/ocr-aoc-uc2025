import cv2, numpy as np, os, glob

# === CONFIGURACIÓN ===
EAST_MODEL   = "models/frozen_east_text_detection.pb"
INPUT_GLOB   = "inputs/*.*"
OUT_DIR      = "outputs_fine_tuned"

# Umbrales de detección
CONF_THRESH  = 0.35   # confianza mínima para EAST
NMS_THRESH   = 0.25   # umbral NMS para fusionar rectángulos

# Ratios morfológicos (ajustados)
h_ratio      = 0.25   # ancho dilatación horizontal
v_ratio      = 0.05   # alto dilatación vertical
close_ratio  = 0.04   # kernel para cierre
open_ratio   = 0.02   # kernel para opening final

os.makedirs(OUT_DIR, exist_ok=True)
net = cv2.dnn.readNet(EAST_MODEL)

def detect_and_fuse(path):
    img = cv2.imread(path)
    H, W = img.shape[:2]

    # 1) Blob EAST
    newW, newH = (W//32)*32, (H//32)*32
    blob = cv2.dnn.blobFromImage(img, 1.0, (newW,newH),
                                 (123.68,116.78,103.94),
                                 swapRB=True, crop=False)
    net.setInput(blob)
    scores, geometry = net.forward([
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"
    ])

    # 2) Decodificar cajas
    rects, confidences = [], []
    rows, cols = scores.shape[2:4]
    for y in range(rows):
        sd = scores[0,0,y]
        x0,x1,x2,x3 = [geometry[0,i,y] for i in range(4)]
        angles = geometry[0,4,y]
        for x in range(cols):
            if sd[x] < CONF_THRESH: continue
            offX,offY = x*4.0, y*4.0
            angle = angles[x]
            cos,sin = np.cos(angle), np.sin(angle)
            h = x0[x]+x2[x]; w = x1[x]+x3[x]
            endX = int(offX + cos*x1[x] + sin*x2[x])
            endY = int(offY - sin*x1[x] + cos*x2[x])
            startX, startY = int(endX-w), int(endY-h)
            rects.append((startX,startY,endX,endY))
            confidences.append(float(sd[x]))

    # 3) NMS
    idxs = cv2.dnn.NMSBoxes(rects, confidences, CONF_THRESH, NMS_THRESH)

    # 4) Máscara combinada
    mask = np.zeros((H,W), dtype=np.uint8)
    if len(idxs)>0:
        for i in idxs.flatten():
            sx,sy,ex,ey = rects[i]
            cv2.rectangle(mask,
                          (max(0,sx),max(0,sy)),
                          (min(W,ex),min(H,ey)),
                          255, -1)

    # 5) Morfología ajustada
    k_h     = cv2.getStructuringElement(cv2.MORPH_RECT, (int(W*h_ratio), 1))
    mask_h  = cv2.dilate(mask, k_h, iterations=1)

    k_v     = cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(H*v_ratio)))
    mask_v  = cv2.dilate(mask_h, k_v, iterations=1)

    k_close = cv2.getStructuringElement(
                  cv2.MORPH_RECT,
                  (int(W*close_ratio), int(H*close_ratio)))
    mask_c  = cv2.morphologyEx(mask_v, cv2.MORPH_CLOSE, k_close)

    k_open  = cv2.getStructuringElement(
                  cv2.MORPH_RECT,
                  (int(W*open_ratio), int(H*open_ratio)))
    fused   = cv2.morphologyEx(mask_c, cv2.MORPH_OPEN, k_open)

    # 6) Contornos y recortes
    contours,_ = cv2.findContours(fused, cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_SIMPLE)
    crops = []
    for i,cnt in enumerate(contours):
        x,y,w,h = cv2.boundingRect(cnt)
        crop = img[y:y+h, x:x+w]
        fn = os.path.join(OUT_DIR,
             f"{os.path.basename(path).rsplit('.',1)[0]}_bub_{i}.png")
        cv2.imwrite(fn, crop)
        crops.append(fn)
    return crops

if __name__=="__main__":
    for f in glob.glob(INPUT_GLOB):
        print("Procesando:", f)
        res = detect_and_fuse(f)
        print(" → Guardados:", res)
    print("Proceso con ajuste fino completado. Revisa", OUT_DIR)
