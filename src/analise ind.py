
from ultralytics import YOLO
import cv2
import time
import logging

# Carrega o modelo
model = YOLO(r"C:\Users\yan.fernandes\Downloads\weights.pt")

# Inicia a câmera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    logging.error("Erro ao abrir a câmera.")
    print("Erro ao abrir a câmera.")
    exit()


# Parâmetros de validação
CLASSE_DE_ESTRIBO = "Estribo"

# Área esperada do Estribo (baseada na imagem que você mandou)
area_x1, area_y1 = 220, 540
area_x2, area_y2 = 500, 580

# Tolerâncias
offset_pos = 75  # Para afrouxar as regras de posição
tolerancia_rotacao = 1.50  # Para permitir mais variação no ângulo

# Aspect ratio esperado baseado na bounding box ideal
altura_esperada = area_y2 - area_y1
largura_esperada = area_x2 - area_x1
aspect_ratio_ideal = largura_esperada / altura_esperada

def dentro_tolerancia(valor, esperado, margem):
    return (esperado - margem) <= valor <= (esperado + margem)

def verificar_ESTRIBO(x1, y1, x2, y2):
    largura = x2 - x1
    altura = y2 - y1 + 0.01  # evitar divisão por zero
    aspect_ratio = largura / altura

    erros = []

    if not dentro_tolerancia(x1, area_x1, offset_pos):
        erros.append("Posição X fora do esperado")
    if not dentro_tolerancia(y1, area_y1, offset_pos):
        erros.append("Posição Y fora do esperado")
    if abs(aspect_ratio - aspect_ratio_ideal) > tolerancia_rotacao:
        erros.append("Angulação suspeita (Estribo torto)")

    return erros

print("Sistema de verificação de Estribo INICIADO.")
print("Pressione p para sair.")
print(model.names)

# Limite de confiança para as classes
CONFIDENCE_THRESHOLD_ESTRIBO = 0.85  # Limite de confiança para "Estribo"
CONFIDENCE_THRESHOLD_TELA = 0.90  # Limite de confiança para "Tela"

# Contadores para Tela e Estribo
tela_detectada = 0
estribo_detectado = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Erro na leitura da câmera.")
        break

    results = model(frame)[0]

    # Contagem das detecções
    tela_detectada = 0
    estribo_detectado = 0

    # Desenha a caixa roxa para a área de anomalia (sempre visível)
    cv2.rectangle(frame, (area_x1, area_y1), (area_x2, area_y2), (255, 0, 255), 2)  # Cor roxa
    cv2.putText(frame, "Area de Anomalia", (area_x1, area_y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

    # Verificando as detecções
    for box in results.boxes:
        nome_classe = model.names[int(box.cls)]
        conf = float(box.conf)

        # Detecta "Tela" com confiança > 0.90
        if nome_classe == "Tela" and conf > CONFIDENCE_THRESHOLD_TELA:
            tela_detectada += 1
            # Desenha a caixa da Tela
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
            label = f"{nome_classe} ({conf:.2f})"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Detecta "Estribo" com confiança > 0.85
        elif nome_classe == "Estribo" and conf > CONFIDENCE_THRESHOLD_ESTRIBO:
            estribo_detectado += 1
            # Desenha a caixa do Estribo (verde por padrão)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cor_estribo = (0, 255, 0)  # Verde por padrão
            erros = verificar_ESTRIBO(x1, y1, x2, y2)
            
            # Se houver erros (anomalia), a caixa ficará vermelha
            if erros:
                cor_estribo = (0, 0, 255)  # Vermelho para anomalia
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), cor_estribo, 2)
            label = f"{nome_classe} ({conf:.2f})"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor_estribo, 2)

            # Exibe erros no terminal se houver anomalia
            if erros:
                print("⚠️ ANOMALIA DETECTADA:")
                for erro in erros:
                    print(f" - {erro}")

    # Verifica se há mais de 1 tela ou estribo
    if tela_detectada > 1:
        print("⚠️ Mais de 1 Tela detectada!")
    if estribo_detectado > 1:
        print("⚠️ Mais de 1 Estribo detectado!")

    # Exibe a imagem com as caixas desenhadas
    cv2.imshow("Monitoramento de Estribos", frame)

    if cv2.waitKey(1) & 0xFF == ord("p"):
        break

cap.release()
cv2.destroyAllWindows()
